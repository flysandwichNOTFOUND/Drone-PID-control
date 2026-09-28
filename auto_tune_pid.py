"""Automatically tune the PID controllers used by the drone prototype.

Place this file beside:
    testing.py
    controllerV1.py

Run:
    python auto_tune_pid.py

The tuner works in stages:
    1. Height
    2. Roll
    3. Pitch
    4. Yaw
    5. X position
    6. Y position
    7. Combined X/Y validation

It does not modify main.py or controllerV1.py. Results are written to
optimized_pid_gains.json and auto_tune_results.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from testing import Drone, DroneState
from controllerV1 import (
    HeightPIDController,
    OrientationPIDController,
    PositionPIDController,
)


# ============================================================
# 1. DRONE PARAMETERS -- keep these equal to main.py
# ============================================================

TOTAL_MASS = 1.0
ARM_LENGTH = 0.20
MAX_THRUST_PER_MOTOR = 10.0

MOMENT_OF_INERTIA = np.array(
    [0.005, 0.005, 0.009],
    dtype=float,
)

THRUST_COEFFICIENT = 1.0e-5
TORQUE_COEFFICIENT = 2.0e-7
GRAVITY = 9.81


# ============================================================
# 2. SIMULATION AND CONTROLLER SETTINGS
# ============================================================

DT = 0.01
TARGET_HEIGHT = 2.0
TARGET_YAW_DEGREES = 0.0
POSITION_TEST_DISTANCE = 1.0

MAX_ORIENTATION_CORRECTION = 20.0
MAX_TILT_DEGREES = 5.0

# Increase this value for a more thorough but slower search.
TRIALS_PER_CONTROLLER = 90
RANDOM_SEED = 7


# ============================================================
# 3. CURRENT GAINS FROM main.py
# ============================================================

INITIAL_HEIGHT_GAINS = np.array(
    [44.801684, 0.052019, 47.593191],
    dtype=float,
)

INITIAL_ROLL_GAINS = np.array(
    [9.884521, 0.0, 4.714994],
    dtype=float,
)

INITIAL_PITCH_GAINS = np.array(
    [9.963423, 0.029695, 4.678016],
    dtype=float,
)

INITIAL_YAW_GAINS = np.array(
    [15.158408, 0.013277, 14.900141],
    dtype=float,
)

INITIAL_X_GAINS = np.array(
    [0.01, 0.0, 0.08],
    dtype=float,
)

INITIAL_Y_GAINS = np.array(
    [0.01, 0.0, 0.08],
    dtype=float,
)


# Search limits: [minimum, maximum] for Kp, Ki, and Kd.
HEIGHT_BOUNDS = np.array(
    [[20.0, 50.0], [0.0, 0.5], [25.0, 65.0]],
    dtype=float,
)

ROLL_BOUNDS = np.array(
    [[3.0, 14.0], [0.0, 0.20], [2.0, 10.0]],
    dtype=float,
)

PITCH_BOUNDS = np.array(
    [[3.0, 14.0], [0.0, 0.20], [2.0, 10.0]],
    dtype=float,
)

YAW_BOUNDS = np.array(
    [[8.0, 24.0], [0.0, 0.20], [5.0, 22.0]],
    dtype=float,
)

POSITION_BOUNDS = np.array(
    [[0.002, 0.060], [0.0, 0.010], [0.010, 0.180]],
    dtype=float,
)


MAX_MOTOR_SPEED = np.sqrt(
    MAX_THRUST_PER_MOTOR / THRUST_COEFFICIENT
)

HOVER_SPEED = np.sqrt(
    TOTAL_MASS * GRAVITY
    / (4.0 * THRUST_COEFFICIENT)
)


def create_drone():
    return Drone(
        total_mass=TOTAL_MASS,
        arm_len=ARM_LENGTH,
        max_thrust_per_motor=MAX_THRUST_PER_MOTOR,
        moi=MOMENT_OF_INERTIA.copy(),
        thrust_coe=THRUST_COEFFICIENT,
        torque_coe=TORQUE_COEFFICIENT,
    )


def create_controllers(gains):
    height_controller = HeightPIDController(
        kp=gains["height"][0],
        ki=gains["height"][1],
        kd=gains["height"][2],
        hover_speed=HOVER_SPEED,
        max_motor_speed=MAX_MOTOR_SPEED,
    )

    orientation_controller = OrientationPIDController(
        roll_gains=gains["roll"],
        pitch_gains=gains["pitch"],
        yaw_gains=gains["yaw"],
        max_correction=MAX_ORIENTATION_CORRECTION,
        max_motor_speed=MAX_MOTOR_SPEED,
    )

    position_controller = PositionPIDController(
        x_gains=gains["x"],
        y_gains=gains["y"],
        max_tilt=np.radians(MAX_TILT_DEGREES),
    )

    return (
        height_controller,
        orientation_controller,
        position_controller,
    )


def empty_history():
    return {
        "time": [],
        "position": [],
        "velocity": [],
        "orientation": [],
        "angular_velocity": [],
        "motor_speeds": [],
        "target_height": [],
        "target_orientation": [],
        "target_position": [],
        "saturation_fraction": 0.0,
    }


def run_simulation(
    gains,
    duration,
    fixed_orientation=None,
    target_position=None,
):
    """Run one no-wind simulation using the current main.py control flow."""

    if fixed_orientation is not None and target_position is not None:
        raise ValueError(
            "Use either fixed_orientation or target_position, not both."
        )

    drone = create_drone()
    state = DroneState()

    (
        height_controller,
        orientation_controller,
        position_controller,
    ) = create_controllers(gains)

    if fixed_orientation is None:
        fixed_orientation = np.zeros(3, dtype=float)
    else:
        fixed_orientation = np.asarray(
            fixed_orientation,
            dtype=float,
        )

    if target_position is None:
        target_position_array = np.zeros(2, dtype=float)
    else:
        target_position_array = np.asarray(
            target_position,
            dtype=float,
        )

    history = empty_history()
    saturated_steps = 0
    number_of_steps = int(duration / DT)

    for step in range(number_of_steps):
        base_motor_speeds = (
            height_controller.calculate_motor_speed(
                target_height=TARGET_HEIGHT,
                state=state,
                dt=DT,
            )
        )

        if target_position is None:
            target_orientation = fixed_orientation.copy()
        else:
            target_orientation = (
                position_controller.calculate_target_orientation(
                    target_position=target_position_array,
                    target_yaw=np.radians(TARGET_YAW_DEGREES),
                    state=state,
                    dt=DT,
                )
            )

        # Match main.py's compensation for vertical thrust lost during tilt.
        vertical_thrust_factor = (
            np.cos(state.orientation[0])
            * np.cos(state.orientation[1])
        )

        vertical_thrust_factor = max(
            vertical_thrust_factor,
            0.5,
        )

        base_motor_speeds = (
            base_motor_speeds
            / np.sqrt(vertical_thrust_factor)
        )

        base_motor_speeds = np.clip(
            base_motor_speeds,
            0.0,
            MAX_MOTOR_SPEED,
        )

        state.motor_speeds = (
            orientation_controller.calculate_motor_speed(
                base_motor_speeds=base_motor_speeds,
                target_orientation=target_orientation,
                state=state,
                dt=DT,
            )
        )

        if np.any(
            (state.motor_speeds <= 1.0e-9)
            | (
                state.motor_speeds
                >= MAX_MOTOR_SPEED - 1.0e-9
            )
        ):
            saturated_steps += 1

        drone.update_state(
            state=state,
            dt=DT,
            external_force=np.zeros(3, dtype=float),
        )

        all_state_values = np.concatenate(
            [
                state.position,
                state.velocity,
                state.orientation,
                state.angular_velocity,
                state.motor_speeds,
            ]
        )

        if (
            not np.all(np.isfinite(all_state_values))
            or np.linalg.norm(state.position) > 100.0
            or np.linalg.norm(state.orientation) > 4.0 * np.pi
        ):
            return None

        history["time"].append((step + 1) * DT)
        history["position"].append(state.position.copy())
        history["velocity"].append(state.velocity.copy())
        history["orientation"].append(
            state.orientation.copy()
        )
        history["angular_velocity"].append(
            state.angular_velocity.copy()
        )
        history["motor_speeds"].append(
            state.motor_speeds.copy()
        )
        history["target_height"].append(TARGET_HEIGHT)
        history["target_orientation"].append(
            target_orientation.copy()
        )
        history["target_position"].append(
            target_position_array.copy()
        )

    history["saturation_fraction"] = (
        saturated_steps / number_of_steps
    )

    return history


def history_arrays(history):
    return {
        "time": np.asarray(history["time"], dtype=float),
        "position": np.asarray(
            history["position"],
            dtype=float,
        ),
        "velocity": np.asarray(
            history["velocity"],
            dtype=float,
        ),
        "orientation": np.asarray(
            history["orientation"],
            dtype=float,
        ),
        "angular_velocity": np.asarray(
            history["angular_velocity"],
            dtype=float,
        ),
        "motor_speeds": np.asarray(
            history["motor_speeds"],
            dtype=float,
        ),
        "target_height": np.asarray(
            history["target_height"],
            dtype=float,
        ),
        "target_orientation": np.asarray(
            history["target_orientation"],
            dtype=float,
        ),
        "target_position": np.asarray(
            history["target_position"],
            dtype=float,
        ),
    }


def score_height(history):
    if history is None:
        return float("inf")

    data = history_arrays(history)
    height_error = TARGET_HEIGHT - data["position"][:, 2]
    vertical_velocity = data["velocity"][:, 2]

    overshoot = max(
        0.0,
        np.max(data["position"][:, 2]) - TARGET_HEIGHT,
    )

    saturation_penalty = (
        250.0 * history["saturation_fraction"]
    )

    return (
        8.0 * np.mean(height_error**2)
        + 10.0 * height_error[-1] ** 2
        + 2.0 * overshoot**2
        + 0.30 * np.mean(vertical_velocity**2)
        + 1.0 * vertical_velocity[-1] ** 2
        + saturation_penalty
    )


def score_orientation(history, target_orientation, axis):
    if history is None:
        return float("inf")

    data = history_arrays(history)
    target_orientation = np.asarray(
        target_orientation,
        dtype=float,
    )

    angle = data["orientation"][:, axis]
    angle_error = target_orientation[axis] - angle
    angular_velocity = data["angular_velocity"][:, axis]
    height_error = TARGET_HEIGHT - data["position"][:, 2]

    target_angle = target_orientation[axis]

    if target_angle >= 0.0:
        overshoot = max(0.0, np.max(angle) - target_angle)
    else:
        overshoot = max(0.0, target_angle - np.min(angle))

    other_axes = [index for index in range(3) if index != axis]

    return (
        250.0 * np.mean(angle_error**2)
        + 700.0 * angle_error[-1] ** 2
        + 1200.0 * overshoot**2
        + 12.0 * np.mean(angular_velocity**2)
        + 80.0 * angular_velocity[-1] ** 2
        + 80.0 * np.mean(
            data["orientation"][:, other_axes] ** 2
        )
        + 4.0 * np.mean(height_error**2)
        + 250.0 * history["saturation_fraction"]
    )


def score_position(history, target_position, axis):
    if history is None:
        return float("inf")

    data = history_arrays(history)
    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    position = data["position"]
    velocity = data["velocity"]

    axis_error = target_position[axis] - position[:, axis]
    other_axis = 1 - axis
    other_error = (
        target_position[other_axis]
        - position[:, other_axis]
    )

    # Weight the final half more heavily so the tuner rewards settling.
    second_half = len(axis_error) // 2
    late_error = axis_error[second_half:]

    attitude_tracking_error = (
        data["target_orientation"]
        - data["orientation"]
    )

    height_error = TARGET_HEIGHT - position[:, 2]

    return (
        15.0 * np.mean(axis_error**2)
        + 35.0 * np.mean(late_error**2)
        + 120.0 * axis_error[-1] ** 2
        + 4.0 * np.mean(velocity[:, axis] ** 2)
        + 35.0 * velocity[-1, axis] ** 2
        + 30.0 * np.mean(other_error**2)
        + 80.0 * other_error[-1] ** 2
        + 8.0 * np.mean(attitude_tracking_error**2)
        + 5.0 * np.mean(height_error**2)
        + 300.0 * history["saturation_fraction"]
    )


def search_gains(
    name,
    initial_gains,
    bounds,
    evaluate,
    rng,
):
    """Random search followed by two narrower local searches."""

    best_gains = np.asarray(
        initial_gains,
        dtype=float,
    ).copy()

    best_score = evaluate(best_gains)
    low = bounds[:, 0].copy()
    high = bounds[:, 1].copy()

    trials_per_round = max(
        1,
        TRIALS_PER_CONTROLLER // 3,
    )

    for round_index in range(3):
        for _ in range(trials_per_round):
            candidate = rng.uniform(low, high)

            # Regularly test candidates without integral gain.
            if rng.random() < 0.35:
                candidate[1] = 0.0

            candidate_score = evaluate(candidate)

            if candidate_score < best_score:
                best_score = candidate_score
                best_gains = candidate.copy()

        full_width = bounds[:, 1] - bounds[:, 0]
        search_width = full_width * (
            0.30 / (round_index + 1)
        )

        low = np.maximum(
            bounds[:, 0],
            best_gains - search_width,
        )

        high = np.minimum(
            bounds[:, 1],
            best_gains + search_width,
        )

        print(
            f"{name}: round {round_index + 1}/3, "
            f"score={best_score:.6f}, "
            f"gains={np.round(best_gains, 6)}"
        )

    return best_gains, best_score


def initial_gain_dictionary():
    return {
        "height": INITIAL_HEIGHT_GAINS.copy(),
        "roll": INITIAL_ROLL_GAINS.copy(),
        "pitch": INITIAL_PITCH_GAINS.copy(),
        "yaw": INITIAL_YAW_GAINS.copy(),
        "x": INITIAL_X_GAINS.copy(),
        "y": INITIAL_Y_GAINS.copy(),
    }


def tune_controllers():
    rng = np.random.default_rng(RANDOM_SEED)
    gains = initial_gain_dictionary()
    scores = {}

    def evaluate_height(candidate):
        trial_gains = {
            name: value.copy()
            for name, value in gains.items()
        }
        trial_gains["height"] = candidate

        history = run_simulation(
            gains=trial_gains,
            duration=6.0,
            fixed_orientation=np.zeros(3, dtype=float),
        )

        return score_height(history)

    gains["height"], scores["height"] = search_gains(
        "Height",
        gains["height"],
        HEIGHT_BOUNDS,
        evaluate_height,
        rng,
    )

    orientation_settings = [
        (
            "roll",
            0,
            np.radians([8.0, 0.0, 0.0]),
            ROLL_BOUNDS,
        ),
        (
            "pitch",
            1,
            np.radians([0.0, 8.0, 0.0]),
            PITCH_BOUNDS,
        ),
        (
            "yaw",
            2,
            np.radians([0.0, 0.0, 12.0]),
            YAW_BOUNDS,
        ),
    ]

    for name, axis, target_orientation, bounds in orientation_settings:
        def evaluate_orientation(
            candidate,
            controller_name=name,
            controller_axis=axis,
            controller_target=target_orientation,
        ):
            trial_gains = {
                key: value.copy()
                for key, value in gains.items()
            }
            trial_gains[controller_name] = candidate

            history = run_simulation(
                gains=trial_gains,
                duration=5.0,
                fixed_orientation=controller_target,
            )

            return score_orientation(
                history,
                controller_target,
                controller_axis,
            )

        gains[name], scores[name] = search_gains(
            name.capitalize(),
            gains[name],
            bounds,
            evaluate_orientation,
            rng,
        )

    position_settings = [
        (
            "x",
            0,
            np.array([POSITION_TEST_DISTANCE, 0.0]),
        ),
        (
            "y",
            1,
            np.array([0.0, POSITION_TEST_DISTANCE]),
        ),
    ]

    for name, axis, target_position in position_settings:
        def evaluate_position(
            candidate,
            controller_name=name,
            controller_axis=axis,
            controller_target=target_position,
        ):
            trial_gains = {
                key: value.copy()
                for key, value in gains.items()
            }
            trial_gains[controller_name] = candidate

            history = run_simulation(
                gains=trial_gains,
                duration=12.0,
                target_position=controller_target,
            )

            return score_position(
                history,
                controller_target,
                controller_axis,
            )

        gains[name], scores[name] = search_gains(
            f"{name.upper()} position",
            gains[name],
            POSITION_BOUNDS,
            evaluate_position,
            rng,
        )

    gains["scores"] = scores
    return gains


def combined_validation_score(history, target_position):
    if history is None:
        return float("inf")

    return (
        score_position(history, target_position, axis=0)
        + score_position(history, target_position, axis=1)
    )


def print_recommendations(gains):
    current = initial_gain_dictionary()

    print("\nPID recommendations [Kp, Ki, Kd]")
    print("----------------------------------")

    for name in current:
        recommended = np.asarray(gains[name], dtype=float)
        adjustment = recommended - current[name]

        print(f"\n{name.capitalize()}")
        print("  Current:     ", np.round(current[name], 6))
        print("  Adjustment:  ", np.round(adjustment, 6))
        print("  Recommended: ", np.round(recommended, 6))


def save_json(gains, validation_score):
    current = initial_gain_dictionary()

    output = {
        "gain_order": ["Kp", "Ki", "Kd"],
        "current_gains": {
            name: values.tolist()
            for name, values in current.items()
        },
        "recommended_adjustments": {
            name: (
                np.asarray(gains[name]) - current[name]
            ).tolist()
            for name in current
        },
        "recommended_new_gains": {
            name: np.asarray(gains[name]).tolist()
            for name in current
        },
        "stage_scores": {
            name: float(score)
            for name, score in gains["scores"].items()
        },
        "combined_position_validation_score": float(
            validation_score
        ),
        "settings": {
            "dt": DT,
            "target_height": TARGET_HEIGHT,
            "max_tilt_degrees": MAX_TILT_DEGREES,
            "trials_per_controller": TRIALS_PER_CONTROLLER,
            "random_seed": RANDOM_SEED,
        },
    }

    Path("optimized_pid_gains.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )


def plot_validation(history):
    data = history_arrays(history)

    time = data["time"]
    position = data["position"]
    velocity = data["velocity"]
    motor_speeds = data["motor_speeds"]

    orientation_degrees = np.degrees(
        data["orientation"]
    )

    target_orientation_degrees = np.degrees(
        data["target_orientation"]
    )

    target_position = data["target_position"]

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12, 12),
    )

    axes[0, 0].plot(
        time,
        position[:, 2],
        label="Actual height",
    )
    axes[0, 0].axhline(
        TARGET_HEIGHT,
        color="red",
        linestyle="--",
        label="Target height",
    )
    axes[0, 0].set_title("Drone Altitude")
    axes[0, 0].set_ylabel("Height (m)")
    axes[0, 0].legend()

    axes[0, 1].plot(
        time,
        velocity[:, 2],
        color="tab:orange",
    )
    axes[0, 1].axhline(
        0.0,
        color="black",
        linestyle="--",
    )
    axes[0, 1].set_title("Vertical Velocity")
    axes[0, 1].set_ylabel("Vertical velocity (m/s)")

    for motor_index in range(4):
        axes[1, 0].plot(
            time,
            motor_speeds[:, motor_index],
            label=f"Motor {motor_index + 1}",
        )

    axes[1, 0].axhline(
        HOVER_SPEED,
        color="red",
        linestyle="--",
        label="Hover speed",
    )
    axes[1, 0].axhline(
        MAX_MOTOR_SPEED,
        color="purple",
        linestyle="--",
        label="Maximum motor speed",
    )
    axes[1, 0].set_title("Motor Speeds")
    axes[1, 0].set_ylabel("Motor speed (rad/s)")
    axes[1, 0].legend(fontsize=8)

    labels = ["Roll", "Pitch", "Yaw"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for axis_index, label in enumerate(labels):
        axes[1, 1].plot(
            time,
            orientation_degrees[:, axis_index],
            color=colors[axis_index],
            label=f"Actual {label}",
        )
        axes[1, 1].plot(
            time,
            target_orientation_degrees[:, axis_index],
            color=colors[axis_index],
            linestyle="--",
            label=f"Target {label}",
        )

    axes[1, 1].set_title("Orientation Tracking")
    axes[1, 1].set_ylabel("Orientation (degrees)")
    axes[1, 1].legend(fontsize=7)

    axes[2, 0].plot(
        time,
        position[:, 0],
        color="tab:blue",
        label="X position",
    )
    axes[2, 0].plot(
        time,
        position[:, 1],
        color="tab:orange",
        label="Y position",
    )
    axes[2, 0].plot(
        time,
        target_position[:, 0],
        color="tab:blue",
        linestyle="--",
        label="Target X",
    )
    axes[2, 0].plot(
        time,
        target_position[:, 1],
        color="tab:orange",
        linestyle="--",
        label="Target Y",
    )
    axes[2, 0].set_title("Horizontal Position")
    axes[2, 0].set_ylabel("Position (m)")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].axis("off")

    for axis in axes.flat:
        if axis.axison:
            axis.set_xlabel("Time (s)")
            axis.grid(True)

    figure.tight_layout()
    figure.savefig(
        "auto_tune_results.png",
        dpi=200,
    )
    plt.show()


def main():
    print("Starting staged PID search...")
    print(
        "Stages: height, roll, pitch, yaw, "
        "X position, Y position"
    )
    print(
        f"Trials per controller: {TRIALS_PER_CONTROLLER}"
    )

    gains = tune_controllers()
    print_recommendations(gains)

    validation_target = np.array(
        [POSITION_TEST_DISTANCE, POSITION_TEST_DISTANCE],
        dtype=float,
    )

    validation_history = run_simulation(
        gains=gains,
        duration=15.0,
        target_position=validation_target,
    )

    if validation_history is None:
        raise RuntimeError(
            "The combined X/Y validation became unstable. "
            "Reduce the search bounds or maximum tilt."
        )

    validation_score = combined_validation_score(
        validation_history,
        validation_target,
    )

    print(
        "\nCombined X/Y validation score: "
        f"{validation_score:.6f}"
    )

    save_json(gains, validation_score)
    plot_validation(validation_history)

    print("\nSaved optimized_pid_gains.json")
    print("Saved auto_tune_results.png")


if __name__ == "__main__":
    main()
