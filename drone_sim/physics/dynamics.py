import numpy as np
from drone_sim.physics.attitude import integrate_orientation, rotation_matrix
from drone_sim.validation import positive, vector
from drone_sim.world.terrain import FlatTerrain

class Drone:
    CONTACT_TOLERANCE = 1e-4   # m; bottom points this close to the deepest one share the load
    CONTACT_ITERATIONS = 10    # impulse solver passes per contact step

    def __init__(self, total_mass, arm_len, max_thrust_per_motor, moi, thrust_coe, torque_coe,
                 gravity = 9.81, ground_height = 0.0, terrain = None,
                 collision_surface = None, collision_body = None, contact_friction = None):

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

        # Same ground mesh and body cylinder as the PyBullet backend.
        from drone_sim.world import collision
        from drone_sim.configuration import parameters

        self.collision_surface = collision.collision_surface(self.terrain) if collision_surface is None else collision_surface
        self.collision_body = collision.collision_body() if collision_body is None else collision_body
        friction = parameters.CONTACT_FRICTION if contact_friction is None else contact_friction
        self.contact_friction = positive(friction, "Contact friction", allow_zero = True)

    def get_ground_height(self, x, y):

        return float(self.collision_surface.height(x, y))

    def resting_height(self, x, y):
        """Centre-of-mass height of a level drone sitting on the ground at (x, y)."""

        return self.collision_body.resting_height(self.collision_surface, x, y)

    def bottom_clearance(self, state):

        return self.collision_body.clearance(self.collision_surface, state.position, state.orientation)

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

        if external_force is None:
            external_force = np.zeros(3, dtype = float)
        else:
            external_force = np.asarray(external_force, dtype = float)

        if external_force.shape != (3,):
            raise ValueError("External force must be a three-element vector.")

        rotation = rotation_matrix(state.orientation)

        body_thrust = np.sum(self.thrusts_force(state), axis = 0)

        world_thrust = rotation @ body_thrust

        net_force = world_thrust + self.gravity_force() + external_force

        return net_force

    def net_torques(self, state):

        thrusts = self.motor_thrusts(state)[0]

        roll_torque = self.arm_len * (thrusts[1] - thrusts[3])
        pitch_torque = self.arm_len * (thrusts[2] - thrusts[0])

        effective_speed_squared = thrusts / self.thrust_coe
        yaw_torque = self.torque_coe * ( effective_speed_squared[0] - effective_speed_squared[1] + effective_speed_squared[2] - effective_speed_squared[3] )

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

    def update_state(self, state, dt, external_force = None, wind_velocity = None):
        """Advance one step. Rotors reach their command instantly and there is
        no rotor/body drag or ground effect; wind_velocity is accepted for a
        common interface but only external_force acts here."""

        dt = positive(dt, "Time step")
        state.rotor_speeds = np.minimum(vector(state.motor_speeds, 4, "Motor speeds"), np.sqrt(self.max_thrust_per_motor / self.thrust_coe))

        linear_acceleration = self.linear_acceleration(state, external_force = external_force)
        angular_acceleration = self.angular_acceleration(state)

        state.velocity += linear_acceleration * dt
        state.angular_velocity += angular_acceleration * dt

        state.position += state.velocity * dt
        state.orientation[:] = integrate_orientation(state.orientation, state.angular_velocity, dt)
        self.resolve_ground_contact(state, dt)

    def resolve_ground_contact(self, state, dt):
        """Rigid, non-bouncing contact between the body cylinder and the ground.

        If the bottom disk overlaps the ground, lift the body until its
        deepest point just touches. Every bottom point within CONTACT_TOLERANCE
        of the ground then shares inelastic impulses (normal plus Coulomb
        friction), solved iteratively like PyBullet's contact solver. Impulses
        change linear and angular velocity, so a body touching a slope with
        one edge tips onto it. contact_speed is the fastest closing speed of a
        contact point; contact_force is the total normal impulse divided by dt.
        """

        body, surface = self.collision_body, self.collision_surface
        points = body.bottom_points(state.position, state.orientation)
        gaps = points[:, 2] - surface.height(points[:, 0], points[:, 1])
        deepest = float(np.min(gaps))

        state.ground_contact = bool(deepest <= 0.0)
        state.contact_speed = 0.0
        state.contact_force = 0.0

        if not state.ground_contact:
            return

        state.position[2] -= deepest
        touching = np.flatnonzero(gaps - deepest <= self.CONTACT_TOLERANCE)
        rotation = rotation_matrix(state.orientation)
        inverse_inertia = rotation @ np.diag(1.0 / self.moi) @ rotation.T
        omega = rotation @ state.angular_velocity
        velocity = state.velocity.copy()

        def inverse_mass(offset, direction):
            return 1.0 / self.total_mass + float(np.dot(direction, np.cross(inverse_inertia @ np.cross(offset, direction), offset)))

        contacts = []
        for index in touching:
            offset = rotation @ body.bottom_body[index]
            normal = surface.normal(*points[index, :2])
            tangent = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
            tangent /= np.linalg.norm(tangent)
            directions = (normal, tangent, np.cross(normal, tangent))
            closing = -float(np.dot(velocity + np.cross(omega, offset), normal))
            state.contact_speed = max(state.contact_speed, closing)
            contacts.append([offset, directions, [inverse_mass(offset, d) for d in directions], 0.0, np.zeros(2)])

        def apply(offset, impulse):
            nonlocal velocity, omega
            velocity = velocity + impulse / self.total_mass
            omega = omega + inverse_inertia @ np.cross(offset, impulse)

        for _ in range(self.CONTACT_ITERATIONS):
            for contact in contacts:
                offset, (normal, t1, t2), (k_normal, k_t1, k_t2), total_normal, total_friction = contact
                point_velocity = velocity + np.cross(omega, offset)
                new_normal = max(total_normal - float(np.dot(point_velocity, normal)) / k_normal, 0.0)
                apply(offset, (new_normal - total_normal) * normal)
                contact[3] = new_normal

                point_velocity = velocity + np.cross(omega, offset)
                new_friction = total_friction - np.array([np.dot(point_velocity, t1) / k_t1, np.dot(point_velocity, t2) / k_t2])
                limit = self.contact_friction * new_normal
                size = float(np.linalg.norm(new_friction))
                if size > limit:
                    new_friction *= limit / size
                change = new_friction - total_friction
                apply(offset, change[0] * t1 + change[1] * t2)
                contact[4] = new_friction

        state.velocity = velocity
        state.angular_velocity = rotation.T @ omega
        state.contact_force = sum(contact[3] for contact in contacts) / dt


class DroneState:
    def __init__(self):

        self.position = np.zeros(3, dtype = float)          # x, y, z

        self.velocity = np.zeros(3, dtype = float)          # vx, vy, vz

        self.orientation = np.zeros(3, dtype = float)       # roll, pitch, yaw

        self.angular_velocity = np.array([0, 0, 0], dtype = float)  # p, q, r

        self.motor_speeds = np.array([0, 0, 0, 0], dtype = float)      # w1, w2, w3, w4 commanded (rad/s)
        self.rotor_speeds = np.zeros(4, dtype = float)      # actual rotor speeds (rad/s)
        self.ground_contact = False
        self.contact_speed = 0.0                            # m/s closing speed at contact
        self.contact_force = 0.0                            # N normal force from the ground

    def copy(self):

        copied_state = DroneState()

        copied_state.position = self.position.copy()
        copied_state.velocity = self.velocity.copy()
        copied_state.orientation = self.orientation.copy()
        copied_state.angular_velocity = self.angular_velocity.copy()
        copied_state.motor_speeds = self.motor_speeds.copy()
        copied_state.rotor_speeds = self.rotor_speeds.copy()
        copied_state.ground_contact = self.ground_contact
        copied_state.contact_speed = self.contact_speed
        copied_state.contact_force = self.contact_force

        return copied_state

    def reset(self):

        self.position.fill(0)
        self.velocity.fill(0)
        self.orientation.fill(0)
        self.angular_velocity.fill(0)
        self.motor_speeds.fill(0)
        self.rotor_speeds.fill(0)
        self.ground_contact = False
        self.contact_speed = 0.0
        self.contact_force = 0.0

    def print_status(self):

        print("Drone Status")
        print("------------")
        print("Position:", self.position)
        print("Velocity:", self.velocity)
        print("Orientation:", self.orientation)
        print("Angular Velocity:", self.angular_velocity)
        print("Motor Speeds:", self.motor_speeds)

