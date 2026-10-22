import numpy as np
from copy import copy
from itertools import count
from drone_sim.configuration import parameters
from drone_sim.planning.pathfinding import NoPathError

from drone_sim.physics.dynamics import Drone, DroneState
from drone_sim.control.controllerV1 import HeightPIDController, OrientationPIDController, PositionPIDController
from drone_sim.world.environment import BasicEnvironment
from drone_sim.control.navigation import PathFollower
from drone_sim.planning.routeplanner import RoutePlanner
from drone_sim.validation import positive, vector
from drone_sim.world.terrain import FlatTerrain, HillTerrain, SlopedTerrain, LandscapeTerrain
from drone_sim.control.landing import LandingController

TARGET_YAW = 0
MINIMUM_FLIGHT_ALTITUDE = 2.0     # metres in world Z; terrain clearance can require a higher altitude
TARGET_X = 80
TARGET_Y = 90

# MODE SELECTION
PHYSICS_BACKEND = "compare"      # "simulation", "gym_pybullet", or "compare"
ROUTE_PLANNING_MODE = "astar"     # "astar" or "straight"
TERRAIN_MODE = "landscape"  # "flat", "hill", "slope", "landscape"
WIND_MODE = "none"  #"none", "constant", "gust"
LANDING_MODE = "safe_nearby"  # "strict", "safe_nearby", "exact"


def current_gains():
    """Read PID gains from parameters.py; the tuner uses this same source."""

    return {"height": np.array([parameters.KP, parameters.KI, parameters.KD], dtype = float),
            "roll": parameters.ROLL_GAINS.copy(), "pitch": parameters.PITCH_GAINS.copy(),
            "yaw": parameters.YAW_GAINS.copy(), "x": parameters.X_POSITION_GAINS.copy(),
            "y": parameters.Y_POSITION_GAINS.copy()}


def create_terrain():

    if TERRAIN_MODE == "landscape":
        return LandscapeTerrain(parameters.GROUND_HEIGHT, parameters.TERRAIN_SEED)

    if TERRAIN_MODE == "flat":
        return FlatTerrain(parameters.GROUND_HEIGHT)

    if TERRAIN_MODE == "hill":
        return HillTerrain(parameters.GROUND_HEIGHT, parameters.HILL_CENTER, parameters.HILL_HEIGHT, parameters.HILL_WIDTH)

    if TERRAIN_MODE == "slope":
        return SlopedTerrain(parameters.GROUND_HEIGHT, parameters.TERRAIN_SLOPE)

    raise ValueError("TERRAIN_MODE must be flat, hill, slope, or landscape.")


def create_drone(terrain = None, physics_backend = None):
    """Input: optional terrain. Output: selected physics backend.

    Source: PHYSICS_BACKEND above and drone properties in parameters.py.
    Caller: run_simulation(), tuning tools, and tests.
    """

    physics_backend = PHYSICS_BACKEND if physics_backend is None else physics_backend

    if physics_backend not in ("simulation", "gym_pybullet"):
        raise ValueError("A single drone requires simulation or gym_pybullet; use run_simulation() for compare.")

    if physics_backend == "gym_pybullet":
        from drone_sim.physics.gym_banckend_dynamics import GymPyBulletDrone

        return GymPyBulletDrone(
            config = parameters,
            terrain = create_terrain() if terrain is None else terrain)

    return Drone(parameters.TOTAL_MASS, parameters.ARM_LENGTH, parameters.MAX_THRUST_PER_MOTOR,
                 parameters.MOMENT_OF_INERTIA, parameters.THRUST_COEFFICIENT, parameters.TORQUE_COEFFICIENT,
                 gravity = parameters.GRAVITY, ground_height = parameters.GROUND_HEIGHT,
                 terrain = create_terrain() if terrain is None else terrain)


def create_controllers(gains = None):

    gains = current_gains() if gains is None else gains
    hover = np.sqrt(parameters.TOTAL_MASS * parameters.GRAVITY / (4 * parameters.THRUST_COEFFICIENT))
    maximum = np.sqrt(parameters.MAX_THRUST_PER_MOTOR / parameters.THRUST_COEFFICIENT)

    return (HeightPIDController(*gains["height"], hover, maximum),
            OrientationPIDController(gains["roll"], gains["pitch"], gains["yaw"],
                                     parameters.MAX_ORIENTATION_CORRECTION, maximum),
            PositionPIDController(gains["x"], gains["y"], np.radians(parameters.MAX_TILT_DEGREES),
                                  tilt_smoothing_time = parameters.TILT_SMOOTHING_TIME,
                                  max_tilt_rate = np.radians(parameters.MAX_TILT_RATE_DEGREES)))


def create_environment():

    return BasicEnvironment(WIND_MODE, parameters.WIND_VELOCITY, parameters.WIND_FORCE_COEFFICIENT, parameters.GUST_START, parameters.GUST_DURATION, parameters.GUST_PERIOD)


def control_step(drone, state, controllers, environment, target_height,
                 target_position, target_yaw, dt, current_time,
                 fixed_orientation = None, disarmed = False, target_vertical_velocity = 0.0, target_horizontal_velocity = None, step_observer = None):
    """Shared control/physics step used by the mission and PID trials."""

    height, orientation, position = controllers

    if disarmed:
        state.motor_speeds.fill(0.0)
        return np.array([0.0, 0.0, target_yaw])

    base = height.calculate_motor_speed(target_height, state, dt, target_vertical_velocity)
    target_orientation = (position.calculate_target_orientation(target_position, target_yaw, state, dt, target_horizontal_velocity) if fixed_orientation is None
    else np.asarray(fixed_orientation, dtype = float).copy())

    factor = max(np.cos(state.orientation[0]) * np.cos(state.orientation[1]), 0.5)

    state.motor_speeds = orientation.calculate_motor_speed(base / np.sqrt(factor), target_orientation, state, dt)
    force = environment.calculate_wind_force(state, current_time)
    before = state.copy() if step_observer is not None else None
    drone.update_state(state, dt, external_force = force)

    values = np.concatenate([state.position, state.velocity, state.orientation, state.angular_velocity, state.motor_speeds])

    if not np.all(np.isfinite(values)) or np.max(np.abs(state.position)) > 100.0:
        raise FloatingPointError("Simulation became unstable.")

    if step_observer is not None:
        step_observer(before, state, dt, force, current_time)

    return target_orientation


def run_simulation(gains = None, duration = None, include_landing = None, verbose = True,
                   target_yaw = None, terrain = None, goal_position = None, physics_backend = None, initial_state = None, environment = None, step_observer = None):
    """Run one mission, or return (final_states, comparison_data) in compare mode.

    Each invocation creates fresh controllers, navigation, wind and histories.
    Owned physics resources are closed even when a mission fails.
    """

    physics_backend = PHYSICS_BACKEND if physics_backend is None else physics_backend

    if physics_backend == "compare":
        if step_observer is not None:
            raise ValueError("An external step observer requires a single physics backend.")
        from drone_sim.physics.comparison import run_comparison

        comparison_data = run_comparison(
            duration = duration, gains = gains, include_landing = include_landing,
            verbose = verbose, target_yaw = target_yaw, terrain = terrain,
            goal_position = goal_position, initial_state = initial_state, environment = environment)
        return comparison_data["final_states"], comparison_data

    terrain = create_terrain() if terrain is None else terrain
    drone = create_drone(terrain, physics_backend = physics_backend)

    try:
        return _run_mission(drone, physics_backend, gains, duration, include_landing, verbose, target_yaw, terrain, goal_position, initial_state, environment, step_observer)
    finally:
        close = getattr(drone, "close", None)

        if close is not None:
            close()


def _run_mission(drone, physics_backend, gains, duration, include_landing, verbose, target_yaw, terrain, goal_position, initial_state, environment, step_observer):

    include_landing = parameters.INCLUDE_LANDING if include_landing is None else include_landing
    target_yaw = np.radians(TARGET_YAW) if target_yaw is None else target_yaw
    positive(parameters.DT, "Time step")
    if duration is not None:
        duration = positive(duration, "Simulation duration")
        if duration < parameters.DT:
            raise ValueError("Simulation duration must include at least one time step.")

    arrival_wait = positive(parameters.ARRIVAL_WAIT_TIME, "Arrival wait time", allow_zero = True)

    if not np.isfinite(target_yaw):
        raise ValueError("Target yaw must be finite.")

    terrain = create_terrain() if terrain is None else terrain
    goal = vector([TARGET_X, TARGET_Y, MINIMUM_FLIGHT_ALTITUDE] if goal_position is None else goal_position, 3, "Goal position")
    requested_goal = goal.copy()

    state = DroneState() if initial_state is None else initial_state.copy()

    if initial_state is None:
        state.position[2] = drone.get_ground_height(*state.position[:2])

        if physics_backend == "gym_pybullet":
            state.position[:] = drone.initial_position

    if physics_backend == "gym_pybullet":
        drone.reset(state)

    landing_controller = (LandingController(
        terrain, goal[:2], parameters.LANDING_DESCENT_SPEED, parameters.LANDING_FLARE_SPEED,
        parameters.LANDING_FLARE_HEIGHT, speed_tolerance = parameters.SPEED_TOLERANCE,
        max_slope_degrees = parameters.MAX_LANDING_SLOPE_DEGREES,
        footprint_radius = parameters.ARM_LENGTH, mode = LANDING_MODE,
        search_radius = parameters.LANDING_SEARCH_RADIUS,
        search_resolution = parameters.LANDING_SEARCH_RESOLUTION,
        max_acceleration = parameters.LANDING_MAX_ACCELERATION,
        max_jerk = parameters.LANDING_MAX_JERK,
        approach_gain = parameters.LANDING_APPROACH_GAIN,
        touchdown_speed = parameters.LANDING_TOUCHDOWN_SPEED)
        if include_landing else None)

    if landing_controller is not None:
        goal[:2] = landing_controller.target_xy

    landing_active = False
    hover_speed = np.sqrt(parameters.TOTAL_MASS * parameters.GRAVITY / (4.0 * parameters.THRUST_COEFFICIENT))
    controllers = create_controllers(gains)
    environment = create_environment() if environment is None else copy(environment)
    landed = False

    # Lists for recording simulation data
    time_history = []
    height_history = []
    velocity_history = []
    position_history = []
    motor_speed_history = []
    error_history = []
    orientation_history = []
    angular_velocity_history = []
    target_orientation_history = []
    target_position_history = []
    waypoint_index_history = []
    distance_to_target_history = []
    ground_height_history = []
    landing_phase_history = []
    commanded_position_history = []
    target_velocity_history = []
    velocity_vector_history = []
    contact_speed_history = []
    ground_contact_history = []
    wind_force_history = []
    physics_step_history = []
    flight_stage_history = []
    stage_start_times = {}

    number_of_steps = None if duration is None else int(duration / parameters.DT)
    arrival_time = None
    stop_reason = None

    if ROUTE_PLANNING_MODE not in ("astar", "straight"):
        raise ValueError("ROUTE_PLANNING_MODE must be astar or straight.")

    route_planner = RoutePlanner(
        start_position = state.position.copy(), goal_position = goal,
        ground_height = parameters.GROUND_HEIGHT, minimum_clearance = parameters.MINIMUM_CLEARANCE,
        waypoint_spacing = parameters.WAYPOINT_SPACING, terrain = terrain,
        use_astar = ROUTE_PLANNING_MODE == "astar", grid_resolution = parameters.ASTAR_GRID_RESOLUTION,
        search_margin = parameters.ASTAR_SEARCH_MARGIN, climb_weight = parameters.ASTAR_CLIMB_WEIGHT,
        maximum_altitude = parameters.ASTAR_MAXIMUM_ALTITUDE, clearance_margin = parameters.ASTAR_CLEARANCE_MARGIN)
    waypoints = route_planner.generate_route(include_landing = include_landing)
    requested_waypoints = None
    requested_route_error = None

    if include_landing and LANDING_MODE == "safe_nearby":
        requested_planner = copy(route_planner)
        requested_planner.goal_position = requested_goal.copy()

        try:
            requested_waypoints = np.asarray(requested_planner.generate_route(include_landing = True), dtype = float)
        except NoPathError as error:
            requested_route_error = str(error)

    stop_index = route_planner.landing_start_index - 1 if include_landing else len(waypoints) - 1
    navigation = PathFollower(
        waypoints, state.position.copy(), parameters.ARRIVAL_TOLERANCE, parameters.SPEED_TOLERANCE,
        max_speed = parameters.MAX_FLIGHT_SPEED, max_acceleration = parameters.MAX_FLIGHT_ACCELERATION,
        max_climb_rate = parameters.MAX_CLIMB_RATE, max_descent_rate = parameters.MAX_DESCENT_RATE,
        lookahead_distance = parameters.PATH_LOOKAHEAD_DISTANCE, stop_index = max(0, stop_index),
        max_jerk = parameters.MAX_FLIGHT_JERK)

    #simulation loop

    for step in count():

        if number_of_steps is not None and step >= number_of_steps:
            stop_reason = "duration_limit"
            break

        current_time = step * parameters.DT

        reference_velocity = np.zeros(3)

        if landing_active:
            current_target = np.asarray(navigation.get_current_target()).copy()
            commanded_target = current_target.copy()
        else:
            commanded_target = navigation.update(state, parameters.DT)
            reference_velocity = navigation.target_velocity.copy()
            current_target = np.asarray(navigation.get_current_target()).copy()

            if include_landing and navigation.mission_complete:
                landing_active = True
                navigation.mission_complete = False
                navigation.current_waypoint_index = len(waypoints) - 1
                current_target = np.asarray(navigation.get_current_target()).copy()
                commanded_target = current_target.copy()
                reference_velocity.fill(0.0)

        active_waypoint_index = navigation.current_waypoint_index
        flight_stage = ("landed" if landed else "landing" if landing_active else
                        "climbing" if active_waypoint_index < route_planner.takeoff_waypoint_count
                        else "flying")
        stage_start_times.setdefault(flight_stage, current_time)
        target_height = current_target[2]
        vertical_velocity = reference_velocity[2]
        failed = landing_controller.failed if landing_controller is not None else False

        if landing_active and not landed and not failed:
            commanded_target, vertical_velocity = landing_controller.target(state, parameters.DT)
            reference_velocity[2] = vertical_velocity

        physics_advanced = not (landed or failed)
        wind_force_history.append(environment.calculate_wind_force(state, current_time).copy() if physics_advanced else np.zeros(3))
        physics_step_history.append(physics_advanced)
        target_orientation = control_step(
            drone, state, controllers, environment, commanded_target[2],
            commanded_target[:2], target_yaw, parameters.DT, current_time, disarmed = landed or failed,
            target_vertical_velocity = vertical_velocity,
            target_horizontal_velocity = reference_velocity[:2], step_observer = step_observer)
        if landing_active and not landed and not failed:
            landed = landing_controller.check_touchdown(state)

            if landed:
                navigation.advance_waypoint()

        recorded_time = (step + 1) * parameters.DT

        if landed:
            flight_stage = "landed"
            stage_start_times.setdefault("landed", recorded_time)

        height_error = target_height - state.position[2]

        target_position_history.append(current_target.copy())
        waypoint_index_history.append(active_waypoint_index)
        distance_to_target_history.append(float(np.linalg.norm(current_target - state.position)))
        ground_height_history.append(drone.get_ground_height(*state.position[:2]))
        landing_phase_history.append(landing_controller.phase if landing_active else "cruise")
        commanded_position_history.append(commanded_target.copy())
        target_velocity_history.append(reference_velocity.copy())
        velocity_vector_history.append(state.velocity.copy())
        contact_speed_history.append(state.contact_speed)
        ground_contact_history.append(state.ground_contact)
        flight_stage_history.append(flight_stage)
        time_history.append(recorded_time)
        height_history.append(state.position[2])
        velocity_history.append(state.velocity[2])
        position_history.append(state.position.copy())
        motor_speed_history.append(state.motor_speeds.copy())
        error_history.append(height_error)
        orientation_history.append(np.degrees(state.orientation.copy()))
        target_orientation_history.append(np.degrees(target_orientation.copy()))
        angular_velocity_history.append(state.angular_velocity.copy())

        if landing_controller is not None and landing_controller.failed:
            stop_reason = "landing_failed"
            break

        arrived = landed if include_landing else navigation.mission_complete and navigation.check_arrival(state)
        if arrived:
            if arrival_time is None:
                arrival_time = recorded_time
            if recorded_time - arrival_time >= arrival_wait - 1e-12:
                stop_reason = "destination_reached"
                break
        else:
            arrival_time = None

    # Put all recorded information into one dictionary
    simulation_data = {
        "physics_backend": physics_backend,
        "stop_reason": stop_reason,
        "arrival_time": arrival_time,
        "arrival_wait_time": arrival_wait,
        "terrain_mode": TERRAIN_MODE,
        "wind_mode": environment.wind_mode,
        "time": time_history,
        "height": height_history,
        "velocity": velocity_history,
        "position": position_history,
        "motor_speed": motor_speed_history,
        "error": error_history,
        "orientation": orientation_history,
        "angular_velocity": angular_velocity_history,
        "target_orientation": target_orientation_history,
        "target_height": goal[2],
        "target_x": goal[0],
        "target_y": goal[1],
        "hover_speed": hover_speed,
        "max_motor_speed": np.sqrt(parameters.MAX_THRUST_PER_MOTOR / parameters.THRUST_COEFFICIENT),
        "arrival_tolerance": parameters.ARRIVAL_TOLERANCE,
        "ground_height": terrain.base_height,
        "terrain": terrain,
        "ground_height_history": ground_height_history,
        "clearance": np.asarray(height_history) - np.asarray(ground_height_history),
        "landing_phase": landing_phase_history,
        "flight_stage": flight_stage_history,
        "stage_start_times": stage_start_times,
        "commanded_position_history": commanded_position_history,
        "target_velocity_history": target_velocity_history,
        "velocity_vector": velocity_vector_history,
        "contact_speed": contact_speed_history,
        "ground_contact": ground_contact_history,
        "external_force": wind_force_history,
        "physics_advanced": physics_step_history,
        "landing_failed": landing_controller.failed if landing_controller else False,
        "touchdown_speed": landing_controller.touchdown_speed if landing_controller else None,
        "target_position_history": target_position_history,
        "waypoint_index": waypoint_index_history,
        "distance_to_target": distance_to_target_history,
        "waypoints": np.asarray(waypoints, dtype = float),
        "route_planning_mode": ROUTE_PLANNING_MODE,
        "mission_complete": navigation.mission_complete,
        "landed": landed,
        "landing_mode": LANDING_MODE if landing_controller is not None else None,
        "requested_destination": requested_goal.copy(),
        "requested_waypoints": requested_waypoints,
        "requested_route_error": requested_route_error,
        "selected_destination": goal.copy(),
        "landing_site_offset": float(np.linalg.norm(goal[:2] - requested_goal[:2]))
    }

    if verbose:
        print("Simulation complete")
        print(f'PHYSICS_BACKEND = "{physics_backend}"')
        print(f'ROUTE_PLANNING_MODE = "{ROUTE_PLANNING_MODE}"')
        print(f'TERRAIN_MODE = "{TERRAIN_MODE}"')
        print(f'WIND_MODE = "{WIND_MODE}"')
        print(f'LANDING_MODE = "{LANDING_MODE}"')
        print(f"Final location (x, y, z): {np.array2string(state.position, precision = 3)} m")
        print(f"Mission complete: {navigation.mission_complete}")

    return state, simulation_data


if __name__ == "__main__":
    from drone_sim.visualization import plotting
    plotting.use_backend("Agg")
    final_state, simulation_data = run_simulation()
    # Refresh the terrain preview in compare mode too. Until combined graphs
    # are added, show the simulation run and retain both histories in memory.
    graph_data = simulation_data["simulation"] if PHYSICS_BACKEND == "compare" else simulation_data
    plotting.plot_data(graph_data, show = False)
    plotting.plot_3d_trajectory(graph_data, show = False)
    plotting.plot_2d_trajectory(graph_data, show = False)
    plotting.close_all()
