import numpy as np

class Drone:
    def __init__(
        self,
        total_mass,
        arm_len,
        max_thrust_per_motor,
        moi,
        thrust_coe,
        torque_coe
    ):
        self.total_mass = total_mass
        self.gravity = -9.81
        self.arm_len = arm_len
        self.max_thrust_per_motor = max_thrust_per_motor
        self.moi = moi
        self.thrust_coe = thrust_coe
        self.torque_coe = torque_coe

    def motor_thrusts(self, state):
        raw_thrusts = self.thrust_coe * state.motor_speeds**2

        excess = np.maximum(raw_thrusts - self.max_thrust_per_motor, 0)
        thrusts = np.minimum(raw_thrusts, self.max_thrust_per_motor)
        limit_exceeded = np.any(excess > 0)

        return thrusts, limit_exceeded, excess
    
    def thrusts_force(self, state):

        thrusts = self.motor_thrusts(state)[0]

        forces = np.array([
            [0, 0, thrusts[0]],
            [0, 0, thrusts[1]],
            [0, 0, thrusts[2]],
            [0, 0, thrusts[3]]
        ])

        return forces
        

    def gravity_force(self):
        return np.array([0, 0, self.total_mass * self.gravity])

    def net_force(self, state, external_force = None): #explain

        roll = state.orientation[0]
        pitch = state.orientation[1]
        yaw = state.orientation[2]

        if external_force == None:
            external_force = np.zeros(3, dtype = float)
        else:
            external_force = np.asarray(external_force, dtype = float)

        if external_force.shape != (3,):
            raise ValueError("External force must be a three-element vector.")

        R_roll = np.array([[1,0,0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])
       
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])

        rotation_matrix = R_yaw @ R_pitch @ R_roll

        body_thrust = np.sum(self.thrusts_force(state),axis=0)

        world_thrust  = rotation_matrix @ body_thrust

        net_force = world_thrust + self.gravity_force() + external_force

        return net_force

    def net_torques(self, state):

        thrusts = self.motor_thrusts(state)[0]

        roll_torque = self.arm_len * (thrusts[1] - thrusts[3])
        pitch_torque = self.arm_len * (thrusts[2] - thrusts[0])

        effective_speed_squared = thrusts / self.thrust_coe
        yaw_torque = self.torque_coe * (
            effective_speed_squared[0]
            - effective_speed_squared[1]
            + effective_speed_squared[2]
            - effective_speed_squared[3]
        )

        torques = np.array([roll_torque, pitch_torque, yaw_torque])

        return torques
    
    def linear_acceleration(self, state, external_force = None):
        if self.total_mass <= 0:
            raise ValueError("Total mass must be greater than zero.")
        else:
            linear_acceleration = (self.net_force(state, external_force = None) / self.total_mass) 
            return linear_acceleration

    def angular_acceleration(self, state):
        if np.any(self.moi <= 0):
            raise ValueError("Total MOI must be greater than zero.")

        torque = self.net_torques(state)
        angular_momentum = self.moi * state.angular_velocity

        gyroscopic_term = np.cross(
            state.angular_velocity,
            angular_momentum
        )

        return (
            torque - gyroscopic_term
        ) / self.moi

    def update_state(self, state, dt, external_force = None):
        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")

        linear_acceleration = self.linear_acceleration(state, external_force)
        angular_acceleration = self.angular_acceleration(state)

        state.velocity += linear_acceleration * dt
        state.angular_velocity += angular_acceleration * dt

        state.position += state.velocity * dt
        state.orientation += state.angular_velocity * dt
        

class DroneState:
    def __init__(self):

        # Position
        self.position = np.zeros(3, dtype=float)          # x, y, z

        # Linear velocity
        self.velocity = np.zeros(3, dtype=float)          # vx, vy, vz

        # Orientation
        self.orientation = np.zeros(3, dtype=float)       # roll, pitch, yaw

        # Angular velocity
        self.angular_velocity = np.array([0,0,0], dtype=float)  # p, q, r

        # Motor speeds
        self.motor_speeds = np.array([0,0,0,0], dtype=float)      # w1, w2, w3, w4

    def update_state():
        pass  

    def copy(self):
        copied_state = DroneState()
    
        copied_state.position = self.position.copy()
        copied_state.velocity = self.velocity.copy()
        copied_state.orientation = self.orientation.copy()
        copied_state.angular_velocity = self.angular_velocity.copy()
        copied_state.motor_speeds = self.motor_speeds.copy()
    
        return copied_state 

    def reset(self):
        self.position.fill(0)
        self.velocity.fill(0)
        self.orientation.fill(0)
        self.angular_velocity.fill(0)
        self.motor_speeds.fill(0)

    def print_status(self):
        print("Drone Status")
        print("------------")
        print("Position:", self.position)
        print("Velocity:", self.velocity)
        print("Orientation:", self.orientation)
        print("Angular Velocity:", self.angular_velocity)
        print("Motor Speeds:", self.motor_speeds)


