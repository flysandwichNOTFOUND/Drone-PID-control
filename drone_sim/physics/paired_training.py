import numpy as np

from drone_sim.validation import positive, vector


STATE_COLUMNS = ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r")
FEATURE_COLUMNS = STATE_COLUMNS + ("motor_plus_x", "motor_plus_y", "motor_minus_x", "motor_minus_y", "force_x", "force_y", "force_z", "dt")
FEATURE_UNITS = ("m",) * 3 + ("m/s",) * 3 + ("rad",) * 3 + ("rad/s",) * 7 + ("N",) * 3 + ("s",)


def state_vector(state):

    return np.concatenate((state.position, state.velocity, state.orientation, state.angular_velocity)).astype(np.float64)


class PairedAccelerationCollector:
    """Observe PyBullet steps and predict each from the same state with custom physics."""

    def __init__(self, custom_drone):

        self.custom_drone = custom_drone
        self.rows = []


    def record(self, before, pybullet_after, dt, external_force, time_before):
        """Collect identical-input transitions before landing logic disarms motors."""

        dt = positive(dt, "Paired time step")
        force = vector(external_force, 3, "Paired external force")
        command = vector(before.motor_speeds, 4, "Paired motor command")
        if np.any(command < 0):
            raise ValueError("Paired motor commands must be nonnegative.")
        maximum = np.sqrt(self.custom_drone.max_thrust_per_motor / self.custom_drone.thrust_coe)
        applied = np.minimum(command, maximum)
        # PyBullet readout must confirm that it used the same effective command.
        if not np.allclose(applied, pybullet_after.motor_speeds, rtol = 1e-12, atol = 1e-9):
            raise ValueError("Paired motor commands differ between the physics models.")

        predicted = before.copy()
        predicted.motor_speeds = applied.copy()
        nominal_acceleration = self.custom_drone.linear_acceleration(predicted, external_force = force)
        self.custom_drone.update_state(predicted, dt, external_force = force)
        custom_acceleration = (predicted.velocity - before.velocity) / dt
        pybullet_acceleration = (pybullet_after.velocity - before.velocity) / dt
        residual = pybullet_acceleration - custom_acceleration
        features = np.r_[state_vector(before), applied, force, dt]

        reference_contact = bool(before.ground_contact or pybullet_after.ground_contact
                                 or before.contact_speed > 0 or pybullet_after.contact_speed > 0)
        model_contact = bool(predicted.ground_contact or predicted.contact_speed > 0)
        values = np.r_[features, state_vector(predicted), state_vector(pybullet_after), custom_acceleration, pybullet_acceleration, nominal_acceleration]
        if not np.all(np.isfinite(values)):
            raise ValueError("Paired training row contains nonfinite values.")

        self.rows.append({
            "features": features, "state_before": state_vector(before),
            "custom_state_after": state_vector(predicted), "pybullet_state_after": state_vector(pybullet_after),
            "raw_motor_command": command, "motor_command": applied, "external_force": force,
            "custom_acceleration": custom_acceleration, "pybullet_acceleration": pybullet_acceleration,
            "custom_nominal_acceleration": nominal_acceleration, "acceleration_residual": residual,
            "time_before": float(time_before), "time_after": float(time_before + dt),
            "reference_contact": reference_contact, "model_contact": model_contact,
            "valid_training": not (reference_contact or model_contact)
        })


    def arrays(self):
        """Return full diagnostic rows plus contact-filtered regression X and y."""

        widths = {"features": 20, "state_before": 12, "custom_state_after": 12, "pybullet_state_after": 12,
                  "raw_motor_command": 4, "motor_command": 4, "external_force": 3,
                  "custom_acceleration": 3, "pybullet_acceleration": 3,
                  "custom_nominal_acceleration": 3, "acceleration_residual": 3}
        arrays = {}
        for key, width in widths.items():
            arrays[f"paired_{key}"] = np.asarray([row[key] for row in self.rows], dtype = np.float64).reshape(-1, width)
        for key in ("time_before", "time_after", "reference_contact", "model_contact", "valid_training"):
            dtype = bool if key in ("reference_contact", "model_contact", "valid_training") else np.float64
            arrays[f"paired_{key}"] = np.asarray([row[key] for row in self.rows], dtype = dtype)

        valid = arrays["paired_valid_training"]
        arrays["training_row_indices"] = np.flatnonzero(valid)
        arrays["X"] = arrays["paired_features"][valid]
        arrays["y"] = arrays["paired_acceleration_residual"][valid]
        for key in ("custom_acceleration", "pybullet_acceleration", "acceleration_residual"):
            arrays[key] = arrays[f"paired_{key}"][valid]
        return arrays
