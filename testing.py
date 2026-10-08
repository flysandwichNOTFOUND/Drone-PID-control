import numpy as np
from attitude import integrate_orientation
from validation import positive, vector
from terrain import FlatTerrain

class Drone:
    def __init__(self, total_mass, arm_len, max_thrust_per_motor, moi, thrust_coe, torque_coe,
                 gravity = 9.81, ground_height = 0.0, terrain = None):

        self.total_mass = positive(total_mass, "Total mass")
        self.gravity = -positive(abs(gravity), "Gravity")
        self.arm_len = positive(arm_len, "Arm length")
        self.max_thrust_per_motor = positive(max_thrust_per_motor, "Maximum thrust")
        self.moi = vector(moi, 3, "Moment of inertia")
        if np.any(self.moi <= 0):
            raise ValueError("Moment of inertia must be positive on each axis.")
        self.thrust_coe = positive(thrust_coe, "Thrust coefficient")
        self.torque_coe = positive(torque_coe, "Torque coefficient", allow_zero = True)
        self.ground_height = float(ground_height)
        if not np.isfinite(self.ground_height):
            raise ValueError("Ground height must be finite.")
        self.terrain = terrain if terrain is not None else FlatTerrain(self.ground_height)

    def get_ground_height(self, x, y):

        return float(self.terrain.height(x, y))

    def motor_thrusts(self, state):

        speeds = vector(state.motor_speeds, 4, "Motor speeds")
        if np.any(speeds < 0):
            raise ValueError("Motor speeds must be nonnegative.")
        raw_thrusts = self.thrust_coe * speeds**2

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

        if external_force is None:
            external_force = np.zeros(3, dtype = float)
        else:
            external_force = np.asarray(external_force, dtype = float)

        if external_force.shape != (3,):
            raise ValueError("External force must be a three-element vector.")

        R_roll = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])

        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])

        rotation_matrix = R_yaw @ R_pitch @ R_roll

        body_thrust = np.sum(self.thrusts_force(state), axis = 0)

        world_thrust = rotation_matrix @ body_thrust

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

        return (self.net_force(state, external_force = external_force) / self.total_mass)

    def angular_acceleration(self, state):

        if np.any(self.moi <= 0):
            raise ValueError("Total MOI must be greater than zero.")

        torque = self.net_torques(state)
        angular_momentum = self.moi * state.angular_velocity

        gyroscopic_term = np.cross(state.angular_velocity, angular_momentum)

        return (torque - gyroscopic_term) / self.moi

    def update_state(self, state, dt, external_force = None):

        dt = positive(dt, "Time step")

        linear_acceleration = self.linear_acceleration(state, external_force = external_force)
        angular_acceleration = self.angular_acceleration(state)

        state.velocity += linear_acceleration * dt
        state.angular_velocity += angular_acceleration * dt

        state.position += state.velocity * dt
        state.orientation[:] = integrate_orientation(state.orientation, state.angular_velocity, dt)
        ground = self.get_ground_height(*state.position[:2])
        state.ground_contact = state.position[2] <= ground
        state.contact_speed = 0.0
        if state.ground_contact:
            state.position[2] = ground
            normal = self.terrain.normal(*state.position[:2])
            normal_speed = np.dot(state.velocity, normal)
            state.contact_speed = max(-normal_speed, 0.0)
            if normal_speed < 0:
                state.velocity -= normal_speed * normal


class DroneState:
    def __init__(self):

        self.position = np.zeros(3, dtype = float)          # x, y, z

        self.velocity = np.zeros(3, dtype = float)          # vx, vy, vz

        self.orientation = np.zeros(3, dtype = float)       # roll, pitch, yaw

        self.angular_velocity = np.array([0, 0, 0], dtype = float)  # p, q, r

        self.motor_speeds = np.array([0, 0, 0, 0], dtype = float)      # w1, w2, w3, w4
        self.ground_contact = False
        self.contact_speed = 0.0

    def copy(self):

        copied_state = DroneState()

        copied_state.position = self.position.copy()
        copied_state.velocity = self.velocity.copy()
        copied_state.orientation = self.orientation.copy()
        copied_state.angular_velocity = self.angular_velocity.copy()
        copied_state.motor_speeds = self.motor_speeds.copy()
        copied_state.ground_contact = self.ground_contact
        copied_state.contact_speed = self.contact_speed

        return copied_state

    def reset(self):

        self.position.fill(0)
        self.velocity.fill(0)
        self.orientation.fill(0)
        self.angular_velocity.fill(0)
        self.motor_speeds.fill(0)
        self.ground_contact = False
        self.contact_speed = 0.0

    def print_status(self):

        print("Drone Status")
        print("------------")
        print("Position:", self.position)
        print("Velocity:", self.velocity)
        print("Orientation:", self.orientation)
        print("Angular Velocity:", self.angular_velocity)
        print("Motor Speeds:", self.motor_speeds)

