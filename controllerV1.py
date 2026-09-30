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
 
        if abs(error) < 0.20:
            self.integral_error += error * dt
        else:
            self.integral_error = 0.0

        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)

        if self.previous_error is None: 
            derivative_error = 0.0 
        else: 
            derivative_error = (error - self.previous_error) / dt 
 
        correction = (self.kp * error  + self.ki * self.integral_error  + self.kd * derivative_error) 
 
        motor_speed = self.hover_speed + correction 
        motor_speeds = np.full(4, motor_speed, dtype=float) 
        motor_speeds = np.clip(motor_speeds, 0.0, self.max_motor_speed) 
 
        self.previous_error = error 
 
        return motor_speeds 
 
 
class OrientationPIDController: 
    def __init__(self, roll_gains, pitch_gains, yaw_gains, max_correction, max_motor_speed): 

        #[Kp, Ki, Kd] for eaach
        self.roll_gains = np.asarray(roll_gains, dtype = float) 
        self.pitch_gains = np.asarray(pitch_gains, dtype = float) 
        self.yaw_gains = np.asarray(yaw_gains, dtype = float) 
 
        self.gains = np.array([
            self.roll_gains,
            self.pitch_gains,
            self.yaw_gains
        ]) 
 
        self.max_correction = max_correction 
        self.max_motor_speed = max_motor_speed 
 
        self.integral_error = np.zeros(3, dtype = float) 

    def calculate_corrections(self, target_orientation, state, dt): 
 
        if dt <= 0: 
            raise ValueError("Time step must be greater than zero.") 
 
        target_orientation = np.asarray(target_orientation, dtype=float) 
  
        # Roll error, pitch error, yaw error.
        error = (target_orientation - state.orientation)
 
        self.integral_error += error * dt

        kp = self.gains[:, 0] 
        ki = self.gains[:, 1] 
        kd = self.gains[:, 2]
        
        derivative_error = -np.asarray(state.angular_velocity, dtype = float)
         
        corrections = (kp * error + ki * self.integral_error + kd * derivative_error) 
        corrections = np.clip(corrections, -self.max_correction, self.max_correction) 
  
        return corrections 
 
    def calculate_motor_speed(self, base_motor_speeds, target_orientation, state, dt): 
 
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
 
        motor_speeds = np.clip( motor_speeds, 0.0, self.max_motor_speed) 
 
        return motor_speeds

class PositionPIDController:
    def __init__(self, x_gains, y_gains, max_tilt):
        self.x_gains = x_gains
        self.y_gains = y_gains
        self.max_tilt = max_tilt

        self.x_integral = 0.0
        self.y_integral = 0.0

    def calculate_target_orientation(self, target_position, target_yaw, state, dt):
        
        x_error = target_position[0] - state.position[0]
        y_error = target_position[1] - state.position[1]

        self.x_integral += x_error * dt
        self.y_integral += y_error * dt

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

        pitch_command = (kp_x * x_error + ki_x * self.x_integral - kd_x * state.velocity[0])
        roll_command = -(kp_y * y_error + ki_y * self.y_integral - kd_y * state.velocity[1])
        roll_command = np.clip(roll_command, -self.max_tilt, self.max_tilt)

        pitch_command = np.clip(pitch_command, -self.max_tilt, self.max_tilt)

        return np.array([roll_command, pitch_command, target_yaw])
