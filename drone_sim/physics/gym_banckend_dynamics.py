from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import xml.etree.ElementTree as ET

import numpy as np

from drone_sim.physics.dynamics import DroneState
from drone_sim.configuration.project_paths import OUTPUT_DIRECTORY
from drone_sim.validation import positive, vector
from drone_sim.world.terrain import FlatTerrain


@contextmanager
def _quiet_startup():
    """Capture Python/native startup messages; replay diagnostics on failure.

    File descriptors are redirected too because PyBullet's build banner is
    written by native code. Redirection is limited to backend construction.
    """

    sys.stdout.flush()
    sys.stderr.flush()

    with TemporaryFile(mode = "w+", encoding = "utf-8", errors = "replace") as captured:
        stdout_fd, stderr_fd = os.dup(1), os.dup(2)

        try:
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)

            with redirect_stdout(captured), redirect_stderr(captured):
                yield
        except BaseException:
            captured.flush()
            captured.seek(0)
            diagnostics = captured.read()
            os.write(stderr_fd, diagnostics.encode("utf-8", errors = "replace"))
            raise
        finally:
            captured.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


class GymPyBulletDrone:
    """Alternative to physics.dynamics.Drone, exposing the same stepping interface."""


    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self, config, terrain, gui = False):
        """Input: parameter module, terrain object, GUI flag. Output: backend.

        Source: main.create_drone() supplies parameters.py and create_terrain().
        Caller: main.create_drone(); future comparison runner.
        Calls the setup helpers below; closes owned resources if setup fails.
        """

        self.env = None
        self._model_directory = None
        self.config = config
        self.terrain = terrain
        self.gui = bool(gui)

        self._validate_settings()
        self._configure_timing()
        with _quiet_startup():
            self._load_dependencies()

            try:
                self._create_model_file()
                self._create_environment()
                self._configure_drone()
                self._create_terrain()
                self.env._updateAndStoreKinematicInformation()
            except BaseException:
                self.close()
                raise


    # ============================================================
    # CONFIGURATION AND DEPENDENCIES
    # ============================================================

    def _validate_settings(self):
        """Input: self.config/terrain. Output: validated, copied model settings.

        Source: parameters.py and main.create_terrain(). Caller: __init__.
        Reject bad values before opening a physics connection.
        """

        c = self.config

        self.total_mass = positive(c.TOTAL_MASS, "Total mass")
        self.arm_len = positive(c.ARM_LENGTH, "Arm length")
        self.max_thrust_per_motor = positive(c.MAX_THRUST_PER_MOTOR, "Maximum thrust")
        self.moi = vector(c.MOMENT_OF_INERTIA, 3, "Moment of inertia").copy()

        if np.any(self.moi <= 0) or 2 * np.max(self.moi) > np.sum(self.moi) + 1e-12:
            raise ValueError("Inertia must be positive and satisfy the triangle inequality.")

        self.gravity = positive(c.GRAVITY, "Gravity")
        self.thrust_coe = positive(c.THRUST_COEFFICIENT, "Thrust coefficient")
        self.torque_coe = positive(c.TORQUE_COEFFICIENT, "Torque coefficient", allow_zero = True)
        self.dt = positive(c.DT, "Time step")

        self.body_radius = positive(c.GYM_BODY_RADIUS, "Body collision radius")
        self.body_height = positive(c.GYM_BODY_HEIGHT, "Body collision height")
        self.terrain_resolution = positive(c.GYM_TERRAIN_RESOLUTION, "Terrain resolution")
        self.terrain_bounds = vector(c.GYM_TERRAIN_BOUNDS, 4, "Terrain bounds").copy()
        xmin, xmax, ymin, ymax = self.terrain_bounds

        if not (xmin < 0 < xmax and ymin < 0 < ymax):
            raise ValueError("Terrain bounds must enclose the starting XY position (0, 0).")

        nx = int(np.ceil((xmax - xmin) / self.terrain_resolution)) + 1
        ny = int(np.ceil((ymax - ymin) / self.terrain_resolution)) + 1

        if nx * ny > 250000:
            raise ValueError("Terrain mesh exceeds 250000 vertices; increase resolution spacing.")

        self._terrain_shape = (nx, ny)

        if not callable(getattr(self.terrain, "height", None)):
            raise ValueError("Terrain must provide height(x, y).")

        ground = float(self.terrain.height(0.0, 0.0))

        if not np.isfinite(ground):
            raise ValueError("Starting terrain height must be finite.")

        self.initial_position = np.array([0.0, 0.0, ground + self.body_height / 2 + 0.01])


    def _configure_timing(self):
        """Input: DT and substeps. Output: control/physics frequencies in Hz.

        Source: parameters.py. Caller: __init__. One environment step must
        advance exactly DT, with an integer number of smaller physics steps.
        """

        frequency = 1.0 / self.dt

        if not np.isclose(frequency, round(frequency), rtol = 0, atol = 1e-9):
            raise ValueError("Gym backend requires DT to be the reciprocal of an integer Hz.")

        substeps = self.config.GYM_PHYSICS_SUBSTEPS

        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError("GYM_PHYSICS_SUBSTEPS must be a positive integer.")

        self.control_frequency = int(round(frequency))
        self.physics_frequency = self.control_frequency * int(substeps)


    def _load_dependencies(self):
        """Input: none. Output: optional module/class references on self.

        Source: installed packages in the active environment. Caller: __init__.
        Delayed imports keep the simulation backend independent of Gym.
        """

        try:
            import pybullet
            from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
            from gym_pybullet_drones.utils.enums import DroneModel, Physics
        except ImportError as error:
            raise ImportError("The gym_pybullet backend needs a compatible gym-pybullet-drones " "installation and its dependencies in this Python environment. " "The simulation backend can still run without them.") from error

        self._p = pybullet
        self._aviary_class = CtrlAviary
        self._drone_model = DroneModel.CF2P
        self._physics = Physics.PYB  # No uncalibrated Crazyflie aerodynamic coefficients.


    # ============================================================
    # MODEL AND ENVIRONMENT SETUP
    # ============================================================

    def _create_model_file(self):
        """Input: validated drone properties. Output: temporary custom URDF path.

        Source: parameters.py. Caller: __init__. Base holds all mass; four
        massless fixed links mark motors (+X, +Y, -X, -Y), link 4 marks COM.
        """

        OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)
        self._model_directory = TemporaryDirectory(prefix = "gym-model-", dir = OUTPUT_DIRECTORY)
        self.model_path = Path(self._model_directory.name) / "drone.urdf"

        robot = ET.Element("robot", name = "project_drone")
        base = ET.SubElement(robot, "link", name = "base_link")
        inertial = ET.SubElement(base, "inertial")
        ET.SubElement(inertial, "mass", value = str(self.total_mass))
        ET.SubElement(inertial, "inertia", ixx = str(self.moi[0]), iyy = str(self.moi[1]),
                      izz = str(self.moi[2]), ixy = "0", ixz = "0", iyz = "0")

        for tag in ("visual", "collision"):
            geometry = ET.SubElement(ET.SubElement(base, tag), "geometry")
            ET.SubElement(geometry, "cylinder", radius = str(self.body_radius), length = str(self.body_height))

        locations = [(self.arm_len, 0, 0), (0, self.arm_len, 0), (-self.arm_len, 0, 0), (0, -self.arm_len, 0), (0, 0, 0)]

        for index, location in enumerate(locations):
            name = f"motor_{index}" if index < 4 else "center_of_mass"
            link = ET.SubElement(robot, "link", name = name)
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value = "0")
            ET.SubElement(inertial, "inertia", ixx = "0", iyy = "0", izz = "0", ixy = "0", ixz = "0", iyz = "0")

            joint = ET.SubElement(robot, "joint", name = name + "_joint", type = "fixed")
            ET.SubElement(joint, "parent", link = "base_link")
            ET.SubElement(joint, "child", link = name)
            ET.SubElement(joint, "origin", xyz = " ".join(map(str, location)))

        ET.ElementTree(robot).write(self.model_path, encoding = "utf-8", xml_declaration = True)


    def _create_environment(self):
        """Input: timing, GUI flag, starting pose. Output: self.env/client_id.

        Source: previous helpers. Caller: __init__. CtrlAviary initially loads
        its stock body; _configure_drone replaces it before any physics step.
        """

        self.env = self._aviary_class(
            drone_model = self._drone_model, num_drones = 1,
            initial_xyzs = self.initial_position.reshape(1, 3),
            initial_rpys = np.zeros((1, 3)), physics = self._physics,
            pyb_freq = self.physics_frequency, ctrl_freq = self.control_frequency,
            gui = self.gui, record = False, obstacles = False, user_debug_gui = False,
            output_folder = str(OUTPUT_DIRECTORY))

        self.client_id = self.env.getPyBulletClient()


    def _configure_drone(self):
        """Input: custom URDF and physical settings. Output: matching body/limits.

        Source: parameters.py and model file. Caller: __init__. Convert force
        coefficients from rad/s squared to RPM squared; replace stock limits.
        Uses CtrlAviary internals, so changes upstream need compatibility tests.
        """

        p, env = self._p, self.env

        old_body = int(env.DRONE_IDS[0])
        self.body_id = p.loadURDF(str(self.model_path), self.initial_position.tolist(), flags = p.URDF_USE_INERTIA_FROM_FILE, physicsClientId = self.client_id)
        p.removeBody(old_body, physicsClientId = self.client_id)
        env.DRONE_IDS[0] = self.body_id

        p.changeDynamics(self.body_id, -1, linearDamping = 0, angularDamping = 0, physicsClientId = self.client_id)

        env.M, env.L, env.G = self.total_mass, self.arm_len, self.gravity
        env.J = np.diag(self.moi)
        env.J_INV = np.diag(1.0 / self.moi)

        conversion = (2 * np.pi / 60)**2
        env.KF = self.thrust_coe * conversion
        env.KM = self.torque_coe * conversion

        env.GRAVITY = self.total_mass * self.gravity
        env.THRUST2WEIGHT_RATIO = 4 * self.max_thrust_per_motor / env.GRAVITY
        env.HOVER_RPM = np.sqrt(env.GRAVITY / (4 * env.KF))
        env.MAX_RPM = np.sqrt(self.max_thrust_per_motor / env.KF)
        env.MAX_THRUST = 4 * self.max_thrust_per_motor
        env.MAX_XY_TORQUE = self.arm_len * self.max_thrust_per_motor
        env.MAX_Z_TORQUE = 2 * env.KM * env.MAX_RPM**2

        env.COLLISION_H, env.COLLISION_R = self.body_height, self.body_radius
        env.COLLISION_Z_OFFSET = 0.0

        env.GND_EFF_COEFF = 0.0
        env.DRAG_COEFF = np.zeros(3)
        env.DW_COEFF_1 = 0.0

        env.action_space = env._actionSpace()
        env.observation_space = env._observationSpace()
        p.setGravity(0, 0, -self.gravity, physicsClientId = self.client_id)

        # Motor links already match project order (+X, +Y, -X, -Y).
        # Stock CF2P yaw signs oppose our dynamics.py convention.
        self.yaw_torque_sign = -1
        self._motor_speeds = np.zeros(4)
        self._contact_velocity = None


    # ============================================================
    # TERRAIN SETUP
    # ============================================================

    def _create_terrain(self):
        """Input: shared terrain/bounds. Output: self.terrain_body_id in physics.

        Source: main.create_terrain() and configuration. Caller: __init__.
        Flat ground is exact; other surfaces use a finite triangle mesh whose
        heights approximate the analytical surface between sampled vertices.
        """

        p = self._p
        p.removeBody(self.env.PLANE_ID, physicsClientId = self.client_id)

        if type(self.terrain) is FlatTerrain:
            shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId = self.client_id)
            origin = [0, 0, float(self.terrain.height(0, 0))]
        else:
            xmin, xmax, ymin, ymax = self.terrain_bounds
            nx, ny = self._terrain_shape
            x, y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
            z = np.asarray(self.terrain.height(x, y), dtype = float)

            if z.shape != x.shape or not np.all(np.isfinite(z)):
                raise ValueError("Terrain must return finite heights for the mesh grid.")

            vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            indices = []

            for row in range(ny - 1):
                for col in range(nx - 1):
                    a = row * nx + col
                    indices.extend([a, a + 1, a + nx, a + 1, a + nx + 1, a + nx])

            # Large inline vertex buffers exceed PyBullet's command capacity.
            # Load an OBJ from our owned temporary directory instead.
            mesh_path = Path(self._model_directory.name) / "terrain.obj"

            with mesh_path.open("w", encoding = "utf-8") as mesh:
                for vx, vy, vz in vertices:
                    mesh.write(f"v {vx} {vy} {vz}\n")

                for index in range(0, len(indices), 3):
                    a, b, c = indices[index:index + 3]
                    mesh.write(f"f {a + 1} {b + 1} {c + 1}\n")

            shape = p.createCollisionShape(p.GEOM_MESH, fileName = str(mesh_path), flags = p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId = self.client_id)
            origin = [0, 0, 0]

        self.terrain_body_id = p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = shape, basePosition = origin, physicsClientId = self.client_id)

        self.env.PLANE_ID = self.terrain_body_id


    # ============================================================
    # STATE AND PHYSICS INTERFACE
    # ============================================================

    def reset(self, state: DroneState):
        """Input: initial DroneState. Output: None; initialize the simulator.

        Source: mission/comparison initial position, orientation and velocities.
        Caller: future mission setup and comparison runner, before first step.
        """

        if self.env is None:
            raise RuntimeError("The Gym backend is closed.")

        position = vector(state.position, 3, "Position")
        orientation = vector(state.orientation, 3, "Orientation")
        velocity = vector(state.velocity, 3, "Velocity")
        angular_velocity = vector(state.angular_velocity, 3, "Angular velocity")
        speeds = vector(state.motor_speeds, 4, "Motor speeds")

        if np.any(speeds < 0):
            raise ValueError("Motor speeds must be nonnegative.")

        p, env = self._p, self.env
        quaternion = p.getQuaternionFromEuler(orientation.tolist())
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        p.resetBasePositionAndOrientation(self.body_id, position.tolist(), quaternion, physicsClientId = self.client_id)
        p.resetBaseVelocity(self.body_id, linearVelocity = velocity.tolist(), angularVelocity = (rotation @ angular_velocity).tolist(), physicsClientId = self.client_id)

        self._motor_speeds = np.minimum(speeds, np.sqrt(self.max_thrust_per_motor / self.thrust_coe))
        self._contact_velocity = None
        env.step_counter = 0
        env.last_clipped_action = (self._motor_speeds * 60 / (2 * np.pi)).reshape(1, 4)
        p.performCollisionDetection(physicsClientId = self.client_id)
        env._updateAndStoreKinematicInformation()
        self.read_state(state)


    def get_ground_height(self, x, y):
        """Input: world x/y in metres. Output: terrain elevation in metres.

        Source: terrain passed to __init__, also used for collision geometry.
        Caller: main.run_simulation() during initialization and logging.
        """

        coordinates = vector([x, y], 2, "Ground coordinates")
        height = float(self.terrain.height(*coordinates))

        if not np.isfinite(height):
            raise ValueError("Terrain height must be finite.")

        return height


    def update_state(self, state: DroneState, dt, external_force = None):
        """Input: mutable state, seconds, optional world force in newtons.

        Output: None; update the supplied DroneState in place after stepping.
        Source: state.motor_speeds (rad/s) from PID; wind force from environment.
        Caller: main.control_step(); future same-command comparison loop.
        Motor links use project order. Convert rad/s to RPM, apply wind and
        the yaw correction on every physics substep, then call read_state().
        """

        if self.env is None:
            raise RuntimeError("The Gym backend is closed.")

        dt = positive(dt, "Time step")

        if not np.isclose(dt, self.dt, rtol = 0, atol = 1e-12):
            raise ValueError("Time step must equal the configured Gym DT.")

        speeds = vector(state.motor_speeds, 4, "Motor speeds")
        force = np.zeros(3) if external_force is None else vector(external_force, 3, "External force")

        if np.any(speeds < 0):
            raise ValueError("Motor speeds must be nonnegative.")

        self._motor_speeds = np.minimum(speeds, np.sqrt(self.max_thrust_per_motor / self.thrust_coe))
        rpm = (self._motor_speeds * 60 / (2 * np.pi)).reshape(1, 4)
        p, env = self._p, self.env
        original_physics = env._physics
        impact_speed = 0.0
        substep_started = False

        def physics_with_external_forces(motor_rpm, nth_drone):
            nonlocal impact_speed, substep_started
            # CtrlAviary calls this once before each PyBullet substep.
            if substep_started:
                self.read_state(state)
                impact_speed = max(impact_speed, state.contact_speed)

            substep_started = True
            self._contact_velocity = p.getBaseVelocity(self.body_id, physicsClientId = self.client_id)
            original_physics(motor_rpm, nth_drone)
            position = p.getBasePositionAndOrientation(self.body_id, physicsClientId = self.client_id)[0]
            p.applyExternalForce(self.body_id, -1, force.tolist(), position, p.WORLD_FRAME, physicsClientId = self.client_id)

            desired_yaw = self.torque_coe * np.dot(self._motor_speeds**2, [1, -1, 1, -1])
            correction = (1 - self.yaw_torque_sign) * desired_yaw
            p.applyExternalTorque(self.body_id, -1, [0, 0, correction], p.LINK_FRAME, physicsClientId = self.client_id)

        # Keep Gym's action processing, substep timing and observation updates.
        had_override = "_physics" in vars(env)
        previous_override = vars(env).get("_physics")
        env._physics = physics_with_external_forces

        try:
            env.step(rpm)
        finally:
            if had_override:
                env._physics = previous_override
            else:
                del env._physics

        self.read_state(state)
        state.contact_speed = max(impact_speed, state.contact_speed)


    def read_state(self, state: DroneState):
        """Input: state to fill. Output: None; replace its measured fields.

        Source: CtrlAviary observation and PyBullet contact information.
        Caller: update_state() and reset() inside this adapter.
        Position/velocity in world frame, Euler angles in radians,
        angular velocity in body frame, motor speeds in rad/s, contact fields.
        """

        if self.env is None:
            raise RuntimeError("The Gym backend is closed.")

        p = self._p
        position, quaternion = p.getBasePositionAndOrientation(self.body_id, physicsClientId = self.client_id)
        velocity, angular_velocity = p.getBaseVelocity(self.body_id, physicsClientId = self.client_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

        state.position = np.asarray(position, dtype = float).copy()
        state.velocity = np.asarray(velocity, dtype = float).copy()
        state.orientation = np.asarray(p.getEulerFromQuaternion(quaternion), dtype = float)
        state.angular_velocity = rotation.T @ np.asarray(angular_velocity)
        state.motor_speeds = self._motor_speeds.copy()

        contacts = p.getContactPoints(bodyA = self.body_id, bodyB = self.terrain_body_id, physicsClientId = self.client_id)
        contacts = [contact for contact in contacts if contact[8] <= 0]
        state.ground_contact = bool(contacts)
        state.contact_speed = 0.0

        # Use velocity before the last solver step so an impact is not hidden
        # by the collision impulse. Include rotational speed at the contact.
        linear, angular = self._contact_velocity if self._contact_velocity is not None else (velocity, angular_velocity)

        for contact in contacts:
            offset = np.asarray(contact[5]) - state.position
            contact_velocity = np.asarray(linear) + np.cross(angular, offset)
            closing_speed = -float(np.dot(contact_velocity, contact[7]))
            state.contact_speed = max(state.contact_speed, closing_speed)


    # ============================================================
    # RESOURCE CLEANUP
    # ============================================================

    def close(self):
        """Input: none. Output: None; release the physics connection and GUI.

        Source: CtrlAviary instance owned by this backend.
        Caller: future mission/comparison runner in a finally block.
        """

        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None

            if self._model_directory is not None:
                self._model_directory.cleanup()
                self._model_directory = None
