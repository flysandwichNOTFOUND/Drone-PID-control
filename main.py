import numpy as np
import parameters

from testing import Drone, DroneState
from controllerV1 import HeightPIDController, OrientationPIDController, PositionPIDController
from environment import BasicEnvironment
from navigation import PathFollower
from routeplanner import RoutePlanner
from validation import positive, vector
from terrain import FlatTerrain, HillTerrain, SlopedTerrain, LandscapeTerrain
from landing import LandingController


# MISSION TARGETS
TARGET_YAW = 0
TARGET_HEIGHT = 2.0
TARGET_X = 12
TARGET_Y = 6

# MODE SELECTION
ROUTE_PLANNING_MODE = "astar"     # "astar" or "straight"
TERRAIN_MODE = "hill"  # "flat", "hill", "slope", "landscape"
WIND_MODE = "none"  #"none", "constant", "gust"


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


def create_drone(terrain = None):

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

    return BasicEnvironment(WIND_MODE, parameters.WIND_VELOCITY, parameters.WIND_FORCE_COEFFICIENT,
                            parameters.GUST_START, parameters.GUST_DURATION, parameters.GUST_PERIOD)


def control_step(drone, state, controllers, environment, target_height,
                 target_position, target_yaw, dt, current_time,
                 fixed_orientation = None, disarmed = False, target_vertical_velocity = 0.0, target_horizontal_velocity = None):
    """Shared control/physics step used by the mission and PID trials."""

    height, orientation, position = controllers
    if disarmed:
        state.motor_speeds.fill(0.0)
        return np.array([0.0, 0.0, target_yaw])
    base = height.calculate_motor_speed(target_height, state, dt, target_vertical_velocity)
    target_orientation = (position.calculate_target_orientation(
        target_position, target_yaw, state, dt, target_horizontal_velocity) if fixed_orientation is None
        else np.asarray(fixed_orientation, dtype = float).copy())
    factor = max(np.cos(state.orientation[0]) * np.cos(state.orientation[1]), 0.5)
    state.motor_speeds = orientation.calculate_motor_speed(
        base / np.sqrt(factor), target_orientation, state, dt)
    force = environment.calculate_wind_force(state, current_time)
    drone.update_state(state, dt, external_force = force)
    values = np.concatenate([state.position, state.velocity, state.orientation,
                             state.angular_velocity, state.motor_speeds])
    if not np.all(np.isfinite(values)) or np.linalg.norm(state.position) > 100.0:
        raise FloatingPointError("Simulation became unstable.")
    return target_orientation


def run_simulation(gains = None, duration = None, include_landing = None, verbose = True,
                   target_yaw = None, terrain = None, goal_position = None):

    duration = parameters.SIMULATION_TIME if duration is None else duration
    include_landing = parameters.INCLUDE_LANDING if include_landing is None else include_landing
    target_yaw = np.radians(TARGET_YAW) if target_yaw is None else target_yaw
    positive(parameters.DT, "Time step")
    positive(duration, "Simulation duration")
    if duration < parameters.DT:
        raise ValueError("Simulation duration must include at least one time step.")
    if not np.isfinite(target_yaw):
        raise ValueError("Target yaw must be finite.")
    terrain = create_terrain() if terrain is None else terrain
    goal = vector([TARGET_X, TARGET_Y, TARGET_HEIGHT] if goal_position is None else goal_position,
                  3, "Goal position")
    drone = create_drone(terrain)
    state = DroneState()
    state.position[2] = drone.get_ground_height(*state.position[:2])
    landing_controller = (LandingController(
        terrain, goal[:2], parameters.LANDING_DESCENT_SPEED, parameters.LANDING_FLARE_SPEED,
        parameters.LANDING_FLARE_HEIGHT, speed_tolerance = parameters.SPEED_TOLERANCE,
        max_slope_degrees = parameters.MAX_LANDING_SLOPE_DEGREES, footprint_radius = parameters.ARM_LENGTH)
        if include_landing else None)
    landing_active = False
    hover_speed = np.sqrt(parameters.TOTAL_MASS * parameters.GRAVITY / (4.0 * parameters.THRUST_COEFFICIENT))
    controllers = create_controllers(gains)
    environment = create_environment()
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
    flight_stage_history = []
    stage_start_times = {}

    number_of_steps = int(duration / parameters.DT)

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
    stop_index = route_planner.landing_start_index - 1 if include_landing else len(waypoints) - 1
    navigation = PathFollower(
        waypoints, state.position.copy(), parameters.ARRIVAL_TOLERANCE, parameters.SPEED_TOLERANCE,
        max_speed = parameters.MAX_FLIGHT_SPEED, max_acceleration = parameters.MAX_FLIGHT_ACCELERATION,
        max_climb_rate = parameters.MAX_CLIMB_RATE, max_descent_rate = parameters.MAX_DESCENT_RATE,
        lookahead_distance = parameters.PATH_LOOKAHEAD_DISTANCE, stop_index = max(0, stop_index),
        max_jerk = parameters.MAX_FLIGHT_JERK)

    #simulation loop
    for step in range(number_of_steps):

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
        target_orientation = control_step(
            drone, state, controllers, environment, commanded_target[2],
            commanded_target[:2], target_yaw, parameters.DT, current_time, disarmed = landed or failed,
            target_vertical_velocity = vertical_velocity,
            target_horizontal_velocity = reference_velocity[:2])
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

    # Put all recorded information into one dictionary
    simulation_data = {
        "time": time_history,
        "height": height_history,
        "velocity": velocity_history,
        "position": position_history,
        "motor_speed": motor_speed_history,
        "error": error_history,
        "orientation": orientation_history,
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
        "landing_failed": landing_controller.failed if landing_controller else False,
        "touchdown_speed": landing_controller.touchdown_speed if landing_controller else None,
        "target_position_history": target_position_history,
        "waypoint_index": waypoint_index_history,
        "distance_to_target": distance_to_target_history,
        "waypoints": np.asarray(waypoints, dtype = float),
        "route_planning_mode": ROUTE_PLANNING_MODE,
        "mission_complete": navigation.mission_complete,
        "landed": landed
    }

    if verbose:
        print("Simulation complete")
        print(f"Route planner: {ROUTE_PLANNING_MODE}")
        print(f"Hover motor speed: {hover_speed:.3f} rad/s")
        print(f"Final route target: {waypoints[-1]}")
        print(f"Current waypoint: {navigation.current_waypoint_index + 1}/{len(waypoints)}")
        print(f"Mission complete: {navigation.mission_complete}")
        if landing_controller:
            print(f"Landing status: {landing_controller.phase if landing_active else 'not started'}")
        state.print_status()

    return state, simulation_data


if __name__ == "__main__":
    import plotting

    plotting.use_backend("Agg")
    final_state, simulation_data = run_simulation()
    plotting.plot_data(simulation_data, show = False)
    plotting.plot_3d_trajectory(simulation_data, show = False)
    plotting.close_all()
