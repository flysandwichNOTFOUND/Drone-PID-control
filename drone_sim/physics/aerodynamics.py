"""Effects applied only inside the PyBullet reference model.

The custom model (dynamics.py) deliberately leaves these out; the difference
between the two is what the learned correction is meant to capture.

- Motor lag: each rotor speed follows its command with a first-order delay.
- Ground effect: per-rotor thrust gain near the ground (Cheeseman-Bennett),
  using each rotor's height above the shared collision surface.
- Rotor drag: force opposing body-frame airspeed, growing with total rotor
  speed (blade flapping / induced drag, as in Faessler et al. 2018).
- Body drag: quadratic parasitic drag on the airframe.
"""

import numpy as np

from drone_sim.validation import positive, vector


class RotorAerodynamics:

    def __init__(self, config, arm_length, thrust_coefficient):

        self.prop_radius = positive(config.PROP_RADIUS, "Propeller radius")
        self.motor_time_constant = positive(config.MOTOR_TIME_CONSTANT, "Motor time constant", allow_zero = True)
        self.ground_effect_coefficient = positive(config.GROUND_EFFECT_COEFFICIENT, "Ground effect coefficient", allow_zero = True)
        minimum = positive(config.GROUND_EFFECT_MIN_HEIGHT, "Ground effect minimum height")

        if minimum <= 0.25:
            raise ValueError("GROUND_EFFECT_MIN_HEIGHT must exceed 0.25 propeller radii.")

        self.ground_effect_min_height = minimum * self.prop_radius
        self.rotor_drag = vector(config.ROTOR_DRAG_COEFFICIENTS, 3, "Rotor drag coefficients")

        if np.any(self.rotor_drag < 0):
            raise ValueError("Rotor drag coefficients must be nonnegative.")

        self.body_drag_area = positive(config.BODY_DRAG_AREA, "Body drag area", allow_zero = True)
        self.air_density = positive(config.AIR_DENSITY, "Air density")
        self.thrust_coefficient = positive(thrust_coefficient, "Thrust coefficient")
        arm = positive(arm_length, "Arm length")
        # Project rotor order: +X, +Y, -X, -Y.
        self.rotor_offsets = arm * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


    def spin(self, rotor_speeds, command, dt):
        """Advance actual rotor speeds (rad/s) toward the command over dt seconds."""

        if self.motor_time_constant == 0:
            return np.asarray(command, dtype = float).copy()

        blend = 1.0 - np.exp(-dt / self.motor_time_constant)
        return rotor_speeds + blend * (np.asarray(command, dtype = float) - rotor_speeds)


    def rotor_heights(self, position, rotation, surface):
        """Height of each rotor above the ground directly beneath it, in metres."""

        points = np.asarray(position, dtype = float) + self.rotor_offsets @ rotation.T
        return points[:, 2] - surface.height(points[:, 0], points[:, 1])


    def ground_effect_gain(self, heights):
        """Extra thrust fraction per rotor: T_ground / T_free - 1."""

        if self.ground_effect_coefficient == 0:
            return np.zeros(len(heights))

        heights = np.maximum(np.asarray(heights, dtype = float), self.ground_effect_min_height)
        ratio = (self.prop_radius / (4.0 * heights))**2
        return self.ground_effect_coefficient * ratio / (1.0 - ratio)


    def drag_force(self, rotation, velocity, wind_velocity, rotor_speeds):
        """World-frame rotor drag plus body drag, in newtons."""

        air = np.asarray(velocity, dtype = float) - np.asarray(wind_velocity, dtype = float)
        rotor = rotation @ (-self.rotor_drag * float(np.sum(rotor_speeds)) * (rotation.T @ air))
        body = -0.5 * self.air_density * self.body_drag_area * float(np.linalg.norm(air)) * air
        return rotor + body
