import numpy as np
import matplotlib.pyplot as plt

from testing import Drone, DroneState
from controllerV1 import HeightPIDController, OrientationPIDController
from environment import BasicEnvironment


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

KP = 37
KI = 0
KD = 42

TARGET_HEIGHT = 2.0                # m

# Orientation PID gains: [Kp, Ki, Kd]
ROLL_GAINS = np.array([5.0, 0.0, 2.0])
PITCH_GAINS = np.array([5.0, 0.0, 2.0])
YAW_GAINS = np.array([5.0, 0.0, 3.0])

# Target orientation in degrees
TARGET_ROLL = 0.0
TARGET_PITCH = 0.0
TARGET_YAW = 0.0

MAX_ORIENTATION_CORRECTION = 20.0

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

WIND_VELOCITY = np.array([
    0.0,                            # x-direction wind
    0.0,                            # y-direction wind
    0.0                             # vertical wind
], dtype = float)

WIND_FORCE_COEFFICIENT = 0.2


# ============================================================
# 4. SIMULATION PARAMETERS
# ============================================================

DT = 0.01                          # seconds per step
SIMULATION_TIME = 10.0             # total simulation time



# ============================================================
# 5. CREATE THE OBJECTS
# ============================================================
def run_simulation():
    drone = Drone(
        total_mass=TOTAL_MASS,
        arm_len=ARM_LENGTH,
        max_thrust_per_motor=MAX_THRUST_PER_MOTOR,
        moi=MOMENT_OF_INERTIA,
        thrust_coe=THRUST_COEFFICIENT,
        torque_coe=TORQUE_COEFFICIENT
    )

    state = DroneState()

    hover_speed = np.sqrt(
        TOTAL_MASS * abs(drone.gravity)
        / (4.0 * THRUST_COEFFICIENT)
    )

    max_motor_speed = np.sqrt(
        MAX_THRUST_PER_MOTOR
        / THRUST_COEFFICIENT
    )

    height_controller = HeightPIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        hover_speed=hover_speed,
        max_motor_speed=max_motor_speed
    )

    orientation_controller = OrientationPIDController(
        roll_gains=ROLL_GAINS,
        pitch_gains=PITCH_GAINS,
        yaw_gains=YAW_GAINS,
        max_correction=MAX_ORIENTATION_CORRECTION,
        max_motor_speed=max_motor_speed
    )

    environment = BasicEnvironment(
        wind_velocity=WIND_VELOCITY,
        wind_force_coefficient=WIND_FORCE_COEFFICIENT
    )

    number_of_steps = int(SIMULATION_TIME / DT)

    target_orientation = np.radians([TARGET_ROLL, TARGET_PITCH, TARGET_YAW])
    # Lists for recording simulation data
    time_history = []
    height_history = []
    velocity_history = []
    motor_speed_history = []
    error_history = []
    orientation_history = []
    angular_velocity_history = []

    for step in range(number_of_steps):

        base_motor_speeds = (
            height_controller.calculate_motor_speed(
            target_height=TARGET_HEIGHT,
            state=state,
            dt=DT
            )
        )

        state.motor_speeds = (
            orientation_controller.calculate_motor_speed(
            base_motor_speeds=base_motor_speeds,
            target_orientation=target_orientation,
            state=state,
            dt=DT
            )
        )

        external_force = environment.calculate_wind_force(state)

        drone.update_state(
            state=state,
            dt=DT,
            external_force=external_force
        )

        current_time = (step + 1) * DT
        height_error = TARGET_HEIGHT - state.position[2]

        time_history.append(current_time)
        height_history.append(state.position[2])
        velocity_history.append(state.velocity[2])
        motor_speed_history.append(state.motor_speeds.copy())
        error_history.append(height_error)
        orientation_history.append(state.orientation.copy())
        angular_velocity_history.append(state.angular_velocity.copy())

    # Put all recorded information into one dictionary
    simulation_data = {
        "time": time_history,
        "height": height_history,
        "velocity": velocity_history,
        "motor_speed": motor_speed_history,
        "error": error_history,
        "orientation": orientation_history,
        "angular_velocity": angular_velocity_history,
        "target_height": TARGET_HEIGHT,
        "target_orientation": target_orientation,
        "hover_speed": hover_speed
    }

    print("Simulation complete")
    print(f"Hover motor speed: {hover_speed:.3f} rad/s")
    print(f"Target height: {TARGET_HEIGHT:.3f} m")
    state.print_status()

    return state, simulation_data

def plot_data(simulation_data):
    time_history = simulation_data["time"]
    height_history = simulation_data["height"]
    velocity_history = simulation_data["velocity"]
    motor_speed_history = np.array(simulation_data["motor_speed"])
    error_history = simulation_data["error"]

    target_height = simulation_data["target_height"]
    hover_speed = simulation_data["hover_speed"]

    plt.figure(figsize=(10, 8))

    # Plot 1: Altitude
    plt.subplot(2, 2, 1)

    plt.plot(
        time_history,
        height_history,
        label="Actual height"
    )

    plt.axhline(
        y=target_height,
        color="red",
        linestyle="--",
        label="Target height"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Drone Altitude")
    plt.grid(True)
    plt.legend()

    # Plot 2: Vertical velocity
    plt.subplot(2, 2, 2)

    plt.plot(
        time_history,
        velocity_history,
        color="orange"
    )

    plt.axhline(
        y=0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Vertical velocity (m/s)")
    plt.title("Vertical Velocity")
    plt.grid(True)

    # Plot 3: Motor speed
    plt.subplot(2, 2, 3)

    for motor_index in range(4):
        plt.plot(
        time_history,
        motor_speed_history[:, motor_index],
        label=f"Motor {motor_index + 1}"
    )

    plt.axhline(
        y=hover_speed,
        color="red",
        linestyle="--",
        label="Hover speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Motor speed (rad/s)")
    plt.title("Motor Speed")
    plt.grid(True)
    plt.legend()

    # Plot 4: Altitude error
    plt.subplot(2, 2, 4)

    plt.plot(
        time_history,
        error_history,
        color="purple"
    )

    plt.axhline(
        y=0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Altitude error (m)")
    plt.title("Altitude Error")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "altitude_control_results.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    final_state, simulation_data = run_simulation()
    plot_data(simulation_data)





