import numpy as np
from attitude import euler_rates, wrap_angle
from validation import gains, positive, vector

class HeightPIDController:
    def __init__(self, kp, ki, kd, hover_speed, max_motor_speed):

        self.kp, self.ki, self.kd = gains([kp, ki, kd], "Height gains")

        self.hover_speed = positive(hover_speed, "Hover speed")
        self.max_motor_speed = positive(max_motor_speed, "Maximum motor speed")

        self.integral_error = 0.0


    def calculate_motor_speed(self, target_height, state, dt, target_vertical_velocity = 0.0):

        dt = positive(dt, "Time step")
        if not np.isfinite(target_height) or not np.isfinite(target_vertical_velocity):
            raise ValueError("Target height and vertical velocity must be finite.")

        current_height = state.position[2]
        error = target_height - current_height

        if abs(error) < 0.20:
            self.integral_error += error * dt
        else:
            self.integral_error = 0.0

        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)

        derivative_error = target_vertical_velocity - state.velocity[2]

        correction = (self.kp * error + self.ki * self.integral_error + self.kd * derivative_error)

        motor_speed = self.hover_speed + correction
        motor_speeds = np.full(4, motor_speed, dtype = float)
        motor_speeds = np.clip(motor_speeds, 0.0, self.max_motor_speed)


        return motor_speeds


class OrientationPIDController:
    def __init__(self, roll_gains, pitch_gains, yaw_gains, max_correction, max_motor_speed):

        #[Kp, Ki, Kd] for eaach

        self.roll_gains = gains(roll_gains, "Roll gains")
        self.pitch_gains = gains(pitch_gains, "Pitch gains")
        self.yaw_gains = gains(yaw_gains, "Yaw gains")

        self.gains = np.array([
            self.roll_gains,
            self.pitch_gains,
            self.yaw_gains
        ])

        self.max_correction = positive(max_correction, "Maximum correction")
        self.max_motor_speed = positive(max_motor_speed, "Maximum motor speed")

        self.integral_error = np.zeros(3, dtype = float)

    def calculate_corrections(self, target_orientation, state, dt):

        dt = positive(dt, "Time step")

        target_orientation = vector(target_orientation, 3, "Target orientation")

        # Roll error, pitch error, yaw error.
        error = (target_orientation - state.orientation)
        error[2] = wrap_angle(error[2])

        previous_integral = self.integral_error.copy()
        self.integral_error = np.clip(previous_integral + error * dt, -5.0, 5.0)

        kp = self.gains[:, 0]
        ki = self.gains[:, 1]
        kd = self.gains[:, 2]

        derivative_error = -euler_rates(state.orientation, state.angular_velocity)

        corrections = (kp * error + ki * self.integral_error + kd * derivative_error)
        limited = np.clip(corrections, -self.max_correction, self.max_correction)
        # Freeze only integral changes that drive further into saturation.
        blocked = ki * (self.integral_error - previous_integral) * (corrections - limited) > 0
        self.integral_error[blocked] = previous_integral[blocked]
        corrections = np.clip(kp * error + ki * self.integral_error + kd * derivative_error,
                              -self.max_correction, self.max_correction)

        return corrections

    def calculate_motor_speed(self, base_motor_speeds, target_orientation, state, dt):

        base_motor_speeds = vector(base_motor_speeds, 4, "Base motor speeds")
        previous_integral = self.integral_error.copy()
        corrections = self.calculate_corrections(target_orientation = target_orientation, state = state, dt = dt)

        roll_correction = corrections[0]
        pitch_correction = corrections[1]
        yaw_correction = corrections[2]

        # Need change into customized speed for each.
        base_speed = np.mean(base_motor_speeds)

        motor_speeds = np.array([
            base_speed - pitch_correction + yaw_correction,
            base_speed + roll_correction - yaw_correction,
            base_speed + pitch_correction + yaw_correction,
            base_speed - roll_correction - yaw_correction
        ])

        limited = np.clip(motor_speeds, 0.0, self.max_motor_speed)
        mixer = np.array([[0, -1, 1], [1, 0, -1], [0, 1, 1], [-1, 0, -1]], dtype = float)
        realized = np.linalg.pinv(mixer) @ (limited - base_speed)
        blocked = (self.gains[:, 1] * (self.integral_error - previous_integral)
 * (corrections - realized) > 1e-12)
        if np.any(blocked):
            self.integral_error[blocked] = previous_integral[blocked]
            error = vector(target_orientation, 3, "Target orientation") - state.orientation
            error[2] = wrap_angle(error[2])
            corrections = np.clip(self.gains[:, 0] * error + self.gains[:, 1] * self.integral_error
 - self.gains[:, 2] * euler_rates(state.orientation, state.angular_velocity),
                                  -self.max_correction, self.max_correction)
            limited = np.clip(base_speed + mixer @ corrections, 0.0, self.max_motor_speed)
        motor_speeds = limited

        return motor_speeds

class PositionPIDController:
    def __init__(self, x_gains, y_gains, max_tilt, tilt_smoothing_time = 0.0,
                 max_tilt_rate = None):

        self.x_gains = gains(x_gains, "X gains")
        self.y_gains = gains(y_gains, "Y gains")
        self.max_tilt = positive(max_tilt, "Maximum tilt")
        if self.max_tilt >= np.pi / 2:
            raise ValueError("Maximum tilt must be less than 90 degrees.")

        self.tilt_smoothing_time = positive(tilt_smoothing_time, "Tilt smoothing time", allow_zero = True)
        self.max_tilt_rate = None if max_tilt_rate is None else positive(max_tilt_rate, "Maximum tilt rate")
        self.previous_tilt_command = None

        self.x_integral = 0.0
        self.y_integral = 0.0

    def calculate_target_orientation(self, target_position, target_yaw, state, dt, target_velocity = None):

        dt = positive(dt, "Time step")
        target_position = vector(target_position, 2, "Target position")
        if not np.isfinite(target_yaw):
            raise ValueError("Target yaw must be finite.")
        target_velocity = np.zeros(2) if target_velocity is None else vector(target_velocity, 2, "Target velocity")
        x_error = target_position[0] - state.position[0]
        y_error = target_position[1] - state.position[1]

        if abs(x_error) < 0.5:
            self.x_integral += x_error * dt
        else:
            self.x_integral = 0.0

        if abs(y_error) < 0.5:
            self.y_integral += y_error * dt
        else:
            self.y_integral = 0.0

        # Prevent unlimited accumulation
        self.x_integral = np.clip(self.x_integral, -5.0, 5.0)
        self.y_integral = np.clip(self.y_integral, -5.0, 5.0)

        kp_x, ki_x, kd_x = self.x_gains
        kp_y, ki_y, kd_y = self.y_gains

        world_x = kp_x * x_error + ki_x * self.x_integral + kd_x * (target_velocity[0] - state.velocity[0])
        world_y = kp_y * y_error + ki_y * self.y_integral + kd_y * (target_velocity[1] - state.velocity[1])
        yaw = state.orientation[2]
        pitch_command = np.cos(yaw) * world_x + np.sin(yaw) * world_y
        roll_command = np.sin(yaw) * world_x - np.cos(yaw) * world_y
        roll_command = np.clip(roll_command, -self.max_tilt, self.max_tilt)

        pitch_command = np.clip(pitch_command, -self.max_tilt, self.max_tilt)

        tilt_command = np.array([roll_command, pitch_command])
        if self.tilt_smoothing_time > 0 or self.max_tilt_rate is not None:
            if self.previous_tilt_command is None:
                self.previous_tilt_command = np.clip(state.orientation[:2], -self.max_tilt, self.max_tilt).copy()
            # Filter from the previous applied command, then cap each axis's rate.
            alpha = -np.expm1(-dt / self.tilt_smoothing_time) if self.tilt_smoothing_time > 0 else 1.0
            change = alpha * (tilt_command - self.previous_tilt_command)
            if self.max_tilt_rate is not None:
                change = np.clip(change, -self.max_tilt_rate * dt, self.max_tilt_rate * dt)
            tilt_command = self.previous_tilt_command + change
            self.previous_tilt_command = tilt_command.copy()

        return np.r_[tilt_command, target_yaw]
