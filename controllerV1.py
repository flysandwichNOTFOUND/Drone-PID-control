import numpy as np

class HeightPIDController:
    def __init__(self, kp, ki, kd, hover_speed, max_motor_speed):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.hover_speed = hover_speed
        self.max_motor_speed = max_motor_speed

        self.integral_error = 0.0
        self.previous_error = None

    def calculate_motor_speed(self, target_height, state, dt):
        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")
        
        current_height = state.position[2]
        error = target_height - current_height

        self.integral_error += error * dt
        if self.previous_error is None:
            derivative_error = 0.0
        else:
            derivative_error = (error - self.previous_error) / dt

        correction = (self.kp * error
                    + self.ki * self.integral_error
                    + self.kd * derivative_error
                    )

        motor_speed = self.hover_speed +correction
        motor_speeds = np.full(4, motor_speed, dtype = float)
        motor_speeds = np.clip(motor_speeds, 0.0, self.max_motor_speed)

        self.previous_error = error

        return motor_speeds

class OrientationPIDController:
    def __init__ (self, roll_gains, pitch_gains, yaw_gains, max_correction, max_motor_speed):

        # Each gains array contains [Kp, Ki, Kd]

        self.roll_gains = np.asarray(roll_gains, dtype = float)
        self.pitch_gains = np.asarray(pitch_gains, dtype = float)
        self.yaw_gains = np.asarray(yaw_gains, dtype = float)

        self.gains = np.array([self.roll_gains, self.pitch_gains, self.yaw_gains])

        self.max_correction = max_correction
        self.max_motor_speed = max_motor_speed

        self.integral_error = np.zeros(3, dtype=float)
        self.previous_error = None

    def calculate_corrections(self, target_orientation, state, dt):

        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")

        target_orientation = np.asarray(target_orientation, dtype = float)
 
        # roll error, pitch error, yaw error
        error = target_orientation - state.orientation

        self.integral_error += error * dt

        if self.previous_error is None:
            derivative_error = np.zeros(3, dtype=float)
        else:
            derivative_error = (error - self.previous_error) / dt

        kp = self.gains[:, 0]
        ki = self.gains[:, 1]
        kd = self.gains[:, 2]

        corrections = (kp * error + ki * self.integral_error + kd * derivative_error)
        corrections = np.clip(corrections, -self.max_correction, self.max_correction)

        self.previous_error = error.copy()

        return corrections


    def calculate_motor_speed(self, base_motor_speeds, target_orientation, state, dt):

        corrections = self.calculate_corrections(target_orientation = target_orientation, state = state, dt = dt)

        roll_correction = corrections[0]
        pitch_correction = corrections[1]
        yaw_correction = corrections[2]

        # need change into custoumized speed for each
        base_speed = np.mean(base_motor_speeds)

        motor_speeds = np.array([
            base_speed - pitch_correction + yaw_correction,
            base_speed + roll_correction - yaw_correction,
            base_speed + pitch_correction + yaw_correction,
            base_speed - roll_correction - yaw_correction
        ])

        motor_speeds = np.clip(
            motor_speeds,
            0.0,
            self.max_motor_speed
        )

        return motor_speeds