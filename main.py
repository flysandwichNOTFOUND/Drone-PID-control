import numpy as np
import matplotlib.pyplot as plt

from testing import Drone, DroneState
from controllerV1 import HeightPIDController, OrientationPIDController, PositionPIDController
from environment import BasicEnvironment
from navigation import Navigation
from routeplanner import RoutePlanner


# ============================================================
# 1. DRONE PARAMETERS
# Change these values to test different drones
# ============================================================

TOTAL_MASS = 1.0                  # kg
ARM_LENGTH = 0.20                  # m
MAX_THRUST_PER_MOTOR = 10.0        # N

MOMENT_OF_INERTIA = np.array([
    0.005,                          # roll inertia
    0.005,                          # pitch inertia
    0.009                           # yaw inertia
], dtype = float)

THRUST_COEFFICIENT = 1.0e-5
TORQUE_COEFFICIENT = 2.0e-7
GRAVITY = 9.81


# ============================================================
# 2. CONTROLLER PARAMETERS
# Change these values to tune the PID controller
# ============================================================
#height PID
KP = 49.967
KI = 1
KD = 45.189
# Orientation PID gains: [Kp, Ki, Kd]
ROLL_GAINS = np.array([13.893, 0,  6.789])
PITCH_GAINS = np.array([13.757, 0.026, 6.772])
YAW_GAINS = np.array([15.158408 , 0.006  ,  21.907])
X_POSITION_GAINS = np.array([0.056, 0.003, 0.11])
Y_POSITION_GAINS = np.array([0.056, 0.003, 0.11])

# Target orientation in degrees


TARGET_YAW = 0

TARGET_HEIGHT = 2.0
TARGET_X = 1.5
TARGET_Y = -0.5

MAX_ORIENTATION_CORRECTION = 20.0
MAX_TILT_DEGREES = 15.0

# Route planning and waypoint arrival settings
GROUND_HEIGHT = 0.0
MINIMUM_CLEARANCE = 2.0
WAYPOINT_SPACING = 1.0
ARRIVAL_TOLERANCE = 0.2
SPEED_TOLERANCE = 0.15
INCLUDE_LANDING = False

MAX_MOTOR_SPEED = np.sqrt(
    MAX_THRUST_PER_MOTOR
    / THRUST_COEFFICIENT
)

HOVER_SPEED = np.sqrt(
    TOTAL_MASS * GRAVITY
    / (4 * THRUST_COEFFICIENT)
)


# ============================================================
# 3. ENVIRONMENT PARAMETERS
# Change these values to test different wind conditions
# ============================================================

WIND_MODE = "gust"  #"none", "constant", "gust"

WIND_VELOCITY = np.array([0.0, 0.0, 0.0], dtype=float)

WIND_FORCE_COEFFICIENT = 0.2

GUST_START = 10.0
GUST_DURATION = 2.0
GUST_PERIOD = 6.0


# ============================================================
# 4. SIMULATION PARAMETERS
# ============================================================

DT = 0.01                          # seconds per step
SIMULATION_TIME = 40.0             # total simulation time



# ============================================================
# 5. CREATE THE OBJECTS
# ============================================================
def run_simulation():

    drone = Drone(total_mass=TOTAL_MASS, arm_len=ARM_LENGTH, max_thrust_per_motor=MAX_THRUST_PER_MOTOR, moi=MOMENT_OF_INERTIA, thrust_coe=THRUST_COEFFICIENT, torque_coe=TORQUE_COEFFICIENT)
    state = DroneState()

    hover_speed = np.sqrt(TOTAL_MASS * abs(drone.gravity)/ (4.0 * THRUST_COEFFICIENT))
    max_motor_speed = np.sqrt(MAX_THRUST_PER_MOTOR / THRUST_COEFFICIENT)

    #three controllers
    height_controller = HeightPIDController(kp = KP, ki = KI, kd = KD, hover_speed = hover_speed, max_motor_speed = max_motor_speed)
    orientation_controller = OrientationPIDController(roll_gains = ROLL_GAINS, pitch_gains = PITCH_GAINS, yaw_gains = YAW_GAINS, max_correction = MAX_ORIENTATION_CORRECTION, max_motor_speed = max_motor_speed)
    position_controller = PositionPIDController(x_gains = X_POSITION_GAINS, y_gains = Y_POSITION_GAINS, max_tilt = np.radians(MAX_TILT_DEGREES))

    environment = BasicEnvironment(wind_mode = WIND_MODE, wind_velocity = WIND_VELOCITY, wind_force_coefficient = WIND_FORCE_COEFFICIENT, gust_start = GUST_START, gust_duration = GUST_DURATION, gust_period = GUST_PERIOD) 
    
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

    number_of_steps = int(SIMULATION_TIME / DT)
    

    route_planner = RoutePlanner(start_position=state.position.copy(), goal_position=[TARGET_X, TARGET_Y, TARGET_HEIGHT], ground_height=GROUND_HEIGHT, minimum_clearance=MINIMUM_CLEARANCE, waypoint_spacing=WAYPOINT_SPACING)
    waypoints = route_planner.generate_route(include_landing=INCLUDE_LANDING)
    navigation = Navigation(waypoints=waypoints, current_waypoint_index=0, arrival_tolerance=ARRIVAL_TOLERANCE, speed_tolerance=SPEED_TOLERANCE)
    
    #simulation loop
    for step in range(number_of_steps):

        current_time = step * DT

        #if step == int(10.0 / DT):
        #    position_controller.x_integral = 0.0
        #    position_controller.y_integral = 0.0

        current_target = np.asarray(navigation.update(state), dtype=float).copy()
        target_position = current_target[:2]
        target_height = current_target[2]

        base_motor_speeds = height_controller.calculate_motor_speed(target_height=target_height, state=state, dt=DT)
        base_motor_speeds = np.clip(base_motor_speeds, 0.0, max_motor_speed)

        target_orientation = (position_controller.calculate_target_orientation(target_position = target_position,target_yaw = np.radians(TARGET_YAW),state = state,dt = DT))

        # Compensate for vertical thrust lost when the drone tilts
        current_roll = state.orientation[0]
        current_pitch = state.orientation[1]

        vertical_thrust_factor = (np.cos(current_roll) * np.cos(current_pitch))
        vertical_thrust_factor = max(vertical_thrust_factor, 0.5)

        # Motor thrust is proportional to motor speed squared
        base_motor_speeds = (base_motor_speeds / np.sqrt(vertical_thrust_factor))

        state.motor_speeds = (orientation_controller.calculate_motor_speed(base_motor_speeds = base_motor_speeds, target_orientation = target_orientation, state = state, dt = DT))
        external_force = environment.calculate_wind_force(state = state, current_time = current_time)

        drone.update_state(state = state, dt = DT, external_force = external_force)

        recorded_time = (step + 1) * DT
        height_error = target_height - state.position[2]

        target_position_history.append(current_target.copy())
        waypoint_index_history.append(navigation.current_waypoint_index)
        distance_to_target_history.append(navigation.distance_to_target(state))
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
        "target_height": TARGET_HEIGHT,
        "target_x": TARGET_X,
        "target_y": TARGET_Y,
        "hover_speed": hover_speed,
        "target_position_history": target_position_history,
        "waypoint_index": waypoint_index_history,
        "distance_to_target": distance_to_target_history,
        "waypoints": np.asarray(waypoints, dtype=float),
        "mission_complete": navigation.mission_complete
    }

    print("Simulation complete")
    print(f"Hover motor speed: {hover_speed:.3f} rad/s")
    print(f"Final route target: {waypoints[-1]}")
    print(f"Current waypoint: {navigation.current_waypoint_index + 1}/{len(waypoints)}")
    print(f"Mission complete: {navigation.mission_complete}")
    state.print_status()

    return state, simulation_data


def plot_data(simulation_data):
    time_history = np.asarray(simulation_data["time"], dtype=float)
    height_history = np.asarray(simulation_data["height"], dtype=float)
    velocity_history = np.asarray(simulation_data["velocity"], dtype=float)
    position_history = np.asarray(simulation_data["position"], dtype=float)
    motor_speed_history = np.asarray(simulation_data["motor_speed"], dtype=float)
    orientation_history = np.asarray(simulation_data["orientation"], dtype=float)
    target_orientation_history = np.asarray(simulation_data["target_orientation"], dtype=float)
    target_position_history = np.asarray(simulation_data["target_position_history"], dtype=float)

    hover_speed = simulation_data["hover_speed"]

    if target_orientation_history.ndim == 1:
        target_orientation_history = np.tile(target_orientation_history, (len(time_history), 1))

    if orientation_history.ndim != 2:
        raise ValueError("orientation_history must contain one [roll, pitch, yaw] array for every timestep.")

    plt.figure(figsize=(12, 12))

    # Plot 1: Altitude
    plt.subplot(3, 2, 1)
    plt.plot(time_history, height_history, label="Actual height")
    plt.step(time_history, target_position_history[:, 2], color="red", linestyle="--", where="post", label="Target height")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Drone Altitude")
    plt.grid(True)
    plt.legend()

    # Plot 2: Vertical velocity
    plt.subplot(3, 2, 2)
    plt.plot(time_history, velocity_history, color="orange")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Vertical velocity (m/s)")
    plt.title("Vertical Velocity")
    plt.grid(True)

    # Plot 3: Four motor speeds
    plt.subplot(3, 2, 3)

    for motor_index in range(4):
        plt.plot(time_history, motor_speed_history[:, motor_index], label=f"Motor {motor_index + 1}")

    plt.axhline(y=hover_speed, color="red", linestyle="--", label="Hover speed")
    plt.axhline(y=MAX_MOTOR_SPEED, color="purple", linestyle="--", linewidth=1.2, label="Maximum motor speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor speed (rad/s)")
    plt.title("Motor Speeds")
    plt.grid(True)
    plt.legend(fontsize=8)

    # Plot 4: Roll, pitch and yaw
    plt.subplot(3, 2, 4)
    axis_names = ["Roll", "Pitch", "Yaw"]
    axis_colors = ["tab:blue", "tab:orange", "tab:green"]

    for axis_index in range(3):
        plt.plot(time_history, orientation_history[:, axis_index], color=axis_colors[axis_index], label=f"Actual {axis_names[axis_index]}")
        plt.plot(time_history, target_orientation_history[:, axis_index], color=axis_colors[axis_index], linestyle="--", label=f"Target {axis_names[axis_index]}")

    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (degrees)")
    plt.title("Orientation Tracking")
    plt.grid(True)
    plt.legend(fontsize=7)

    # Plot 5: Horizontal position
    plt.subplot(3, 2, 5)
    plt.plot(time_history, position_history[:, 0], color="tab:blue", label="X position")
    plt.plot(time_history, position_history[:, 1], color="tab:orange", label="Y position")
    plt.step(time_history, target_position_history[:, 0], color="tab:blue", linestyle="--", where="post", label="Target X")
    plt.step(time_history, target_position_history[:, 1], color="tab:orange", linestyle="--", where="post", label="Target Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Horizontal position (m)")
    plt.title("Horizontal Position and Drift")
    plt.grid(True)
    plt.legend(fontsize=8)

    # Plot 6: Distance to the active waypoint
    plt.subplot(3, 2, 6)
    plt.plot(time_history, simulation_data["distance_to_target"], label="Distance to active waypoint")
    plt.axhline(y=ARRIVAL_TOLERANCE, color="red", linestyle="--", label="Arrival tolerance")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Waypoint Tracking")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("altitude_control_results.png", dpi=300)
    plt.show()

def plot_3d_trajectory(simulation_data):
    positions = np.asarray(simulation_data["position"], dtype=float)
    waypoints = np.asarray(simulation_data["waypoints"], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:blue", label="Actual trajectory")
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="tab:orange", linestyle="--", marker="o", label="Planned route")
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="green", s=80, label="First recorded position")
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], color="red", marker="*", s=150, label="Endpoint")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="purple", marker="x", s=80, label="Final position")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("Drone 3D Trajectory")

    all_points = np.vstack((positions, waypoints))
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = (lower + upper) / 2
    half_range = max(float(np.max(upper - lower)) / 2, 0.5) * 1.1

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))

    ax.legend()
    plt.tight_layout()
    plt.savefig("drone_3d_trajectory.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    final_state, simulation_data = run_simulation()
    plot_data(simulation_data)
    plot_3d_trajectory(simulation_data)