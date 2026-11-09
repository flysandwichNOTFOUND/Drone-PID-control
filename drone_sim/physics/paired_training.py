import numpy as np

from drone_sim.configuration import parameters
from drone_sim.validation import positive, vector


STATE_COLUMNS = ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r")

# Regression inputs. Absolute position is left out (a model could memorise
# places); height enters as clearance above the ground, which ground effect needs.
FEATURE_COLUMNS = ("vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r", "clearance",
                   "command_plus_x", "command_plus_y", "command_minus_x", "command_minus_y",
                   "rotor_plus_x", "rotor_plus_y", "rotor_minus_x", "rotor_minus_y",
                   "force_x", "force_y", "force_z", "wind_x", "wind_y", "wind_z")
FEATURE_UNITS = ("m/s",) * 3 + ("rad",) * 3 + ("rad/s",) * 3 + ("m",) + ("rad/s",) * 8 + ("N",) * 3 + ("m/s",) * 3

# Targets: world linear-acceleration residual, then body angular-acceleration residual.
TARGET_COLUMNS = ("delta_ax", "delta_ay", "delta_az", "delta_alpha_p", "delta_alpha_q", "delta_alpha_r")
TARGET_UNITS = ("m/s^2",) * 3 + ("rad/s^2",) * 3


def state_vector(state):

    return np.concatenate((state.position, state.velocity, state.orientation, state.angular_velocity)).astype(np.float64)


class PairedAccelerationCollector:
    """Observe PyBullet steps and predict each from the same state with custom physics.

    The custom model receives the motor command and reaches it instantly; the
    PyBullet rotors lag it. The rotor speeds at the start of the step are an
    input, so that lag is learnable rather than hidden.
    """

    def __init__(self, custom_drone, min_clearance = None):

        self.custom_drone = custom_drone
        self.min_clearance = positive(parameters.TRAINING_MIN_CLEARANCE if min_clearance is None else min_clearance,
                                      "Training minimum clearance", allow_zero = True)
        self.rows = []


    def record(self, before, pybullet_after, dt, external_force, time_before, wind_velocity = None):
        """Collect identical-input transitions before landing logic disarms motors."""

        dt = positive(dt, "Paired time step")
        force = vector(external_force, 3, "Paired external force")
        wind = np.zeros(3) if wind_velocity is None else vector(wind_velocity, 3, "Paired wind velocity")
        command = vector(before.motor_speeds, 4, "Paired motor command")
        if np.any(command < 0):
            raise ValueError("Paired motor commands must be nonnegative.")
        maximum = np.sqrt(self.custom_drone.max_thrust_per_motor / self.custom_drone.thrust_coe)
        applied = np.minimum(command, maximum)
        # Both models must clip the command identically (the actual PyBullet
        # rotor speeds differ because of motor lag; that is reported separately).
        if not np.allclose(applied, pybullet_after.motor_speeds, rtol = 1e-12, atol = 1e-9):
            raise ValueError("Paired motor commands differ between the physics models.")

        predicted = before.copy()
        predicted.motor_speeds = applied.copy()
        nominal_acceleration = self.custom_drone.linear_acceleration(predicted, external_force = force)
        self.custom_drone.update_state(predicted, dt, external_force = force)
        custom_acceleration = (predicted.velocity - before.velocity) / dt
        pybullet_acceleration = (pybullet_after.velocity - before.velocity) / dt
        custom_angular = (predicted.angular_velocity - before.angular_velocity) / dt
        pybullet_angular = (pybullet_after.angular_velocity - before.angular_velocity) / dt
        residual = np.r_[pybullet_acceleration - custom_acceleration, pybullet_angular - custom_angular]

        clearance_before = self.custom_drone.bottom_clearance(before)
        rotor_before = vector(before.rotor_speeds, 4, "Paired rotor speeds")
        features = np.r_[before.velocity, before.orientation, before.angular_velocity, clearance_before,
                         applied, rotor_before, force, wind]

        # Exclude any step that touches the ground in either model (contact
        # flag, impact speed or normal force, including within PyBullet
        # substeps) and any step that starts or ends close to the ground.
        def touching(state):
            return bool(state.ground_contact or state.contact_speed > 0 or state.contact_force > 0)

        reference_contact = touching(before) or touching(pybullet_after)
        model_contact = touching(predicted)
        clearance = min(self.custom_drone.bottom_clearance(state) for state in (before, pybullet_after, predicted))
        near_ground = bool(clearance < self.min_clearance)
        values = np.r_[features, state_vector(predicted), state_vector(pybullet_after), residual, nominal_acceleration]
        if not np.all(np.isfinite(values)):
            raise ValueError("Paired training row contains nonfinite values.")

        self.rows.append({
            "features": features, "state_before": state_vector(before),
            "custom_state_after": state_vector(predicted), "pybullet_state_after": state_vector(pybullet_after),
            "raw_motor_command": command, "motor_command": applied, "rotor_speed_before": rotor_before,
            "rotor_speed_after": vector(pybullet_after.rotor_speeds, 4, "PyBullet rotor speeds"),
            "external_force": force, "wind_velocity": wind,
            "custom_acceleration": custom_acceleration, "pybullet_acceleration": pybullet_acceleration,
            "custom_angular_acceleration": custom_angular, "pybullet_angular_acceleration": pybullet_angular,
            "custom_nominal_acceleration": nominal_acceleration, "acceleration_residual": residual,
            "time_before": float(time_before), "time_after": float(time_before + dt),
            "reference_contact": reference_contact, "model_contact": model_contact,
            "reference_contact_force": float(max(before.contact_force, pybullet_after.contact_force)),
            "min_clearance": float(clearance), "near_ground": near_ground,
            "valid_training": not (reference_contact or model_contact or near_ground)
        })


    def arrays(self):
        """Return full diagnostic rows plus contact-filtered regression X and y."""

        widths = {"features": len(FEATURE_COLUMNS), "state_before": 12, "custom_state_after": 12, "pybullet_state_after": 12,
                  "raw_motor_command": 4, "motor_command": 4, "rotor_speed_before": 4, "rotor_speed_after": 4,
                  "external_force": 3, "wind_velocity": 3,
                  "custom_acceleration": 3, "pybullet_acceleration": 3,
                  "custom_angular_acceleration": 3, "pybullet_angular_acceleration": 3,
                  "custom_nominal_acceleration": 3, "acceleration_residual": len(TARGET_COLUMNS)}
        arrays = {}
        for key, width in widths.items():
            arrays[f"paired_{key}"] = np.asarray([row[key] for row in self.rows], dtype = np.float64).reshape(-1, width)
        flags = ("reference_contact", "model_contact", "near_ground", "valid_training")
        for key in ("time_before", "time_after", "reference_contact_force", "min_clearance") + flags:
            dtype = bool if key in flags else np.float64
            arrays[f"paired_{key}"] = np.asarray([row[key] for row in self.rows], dtype = dtype)

        valid = arrays["paired_valid_training"]
        arrays["training_row_indices"] = np.flatnonzero(valid)
        arrays["X"] = arrays["paired_features"][valid]
        arrays["y"] = arrays["paired_acceleration_residual"][valid]
        for key in ("custom_acceleration", "pybullet_acceleration", "custom_angular_acceleration",
                    "pybullet_angular_acceleration", "acceleration_residual"):
            arrays[key] = arrays[f"paired_{key}"][valid]
        return arrays
