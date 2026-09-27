"""Automatically tune the prototype drone PID controllers.

Place this file beside:
    testing.py
    controllerV1.py

Run:
    python auto_tune_pid.py

The script tunes altitude, roll, pitch, and yaw in stages, validates the
result with a combined maneuver, saves the gains to optimized_pid_gains.json,
and saves diagnostic plots to auto_tune_results.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from testing import Drone, DroneState
from controllerV1 import HeightPIDController, OrientationPIDController


# ---------------------------------------------------------------------------
# Drone and search settings -- change these to match main.py when necessary.
# ---------------------------------------------------------------------------
TOTAL_MASS = 2.0
ARM_LENGTH = 0.20
MAX_THRUST_PER_MOTOR = 10.0
MOMENT_OF_INERTIA = np.array([0.005, 0.005, 0.009], dtype=float)
THRUST_COEFFICIENT = 1.0e-5
TORQUE_COEFFICIENT = 2.0e-7
GRAVITY = 9.81

DT = 0.02
TARGET_HEIGHT = 2.0
MAX_ORIENTATION_CORRECTION = 20.0

# More trials usually improve the result but take longer.
TRIALS_PER_CONTROLLER = 150
RANDOM_SEED = 7

# Starting points and search ranges: (minimum, maximum).
INITIAL_HEIGHT_GAINS = np.array([32.0, 0.0, 45.0])
INITIAL_ROLL_GAINS = np.array([5.0, 0.0, 2.0])
INITIAL_PITCH_GAINS = np.array([5.0, 0.0, 2.0])
INITIAL_YAW_GAINS = np.array([10.0, 0.0, 3.0])

# Keep the automatic recommendations within practical adjustment ranges around
# the current controller settings.
HEIGHT_BOUNDS = np.array([[20.0, 45.0], [0.0, 0.8], [30.0, 60.0]])
ROLL_BOUNDS = np.array([[2.0, 10.0], [0.0, 0.3], [0.5, 5.0]])
PITCH_BOUNDS = np.array([[2.0, 10.0], [0.0, 0.3], [0.5, 5.0]])
YAW_BOUNDS = np.array([[5.0, 20.0], [0.0, 0.3], [1.0, 10.0]])


MAX_MOTOR_SPEED = np.sqrt(
    MAX_THRUST_PER_MOTOR / THRUST_COEFFICIENT
)

HOVER_SPEED = np.sqrt(
    TOTAL_MASS * GRAVITY / (4.0 * THRUST_COEFFICIENT)
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


def orientation_motor_speeds(
    controller,
    base_motor_speeds,
    target_orientation,
    state,
    dt,
):
    """Support either singular or plural method names in controllerV1.py."""
    method = getattr(controller, "calculate_motor_speeds", None)
    if method is None:
        method = getattr(controller, "calculate_motor_speed")

    return method(
        base_motor_speeds=base_motor_speeds,
        target_orientation=target_orientation,
        state=state,
        dt=dt,
    )


def run_simulation(
    height_gains,
    roll_gains,
    pitch_gains,
    yaw_gains,
    duration,
    target_function,
    record=False,
):
    """Run one deterministic, no-wind simulation for a set of gains."""
    drone = create_drone()
    state = DroneState()

    height_controller = HeightPIDController(
        kp=height_gains[0],
        ki=height_gains[1],
        kd=height_gains[2],
        hover_speed=HOVER_SPEED,
        max_motor_speed=MAX_MOTOR_SPEED,
    )

    orientation_controller = OrientationPIDController(
        roll_gains=roll_gains,
        pitch_gains=pitch_gains,
        yaw_gains=yaw_gains,
        max_correction=MAX_ORIENTATION_CORRECTION,
        max_motor_speed=MAX_MOTOR_SPEED,
    )

    history = {
        "time": [],
        "position": [],
        "velocity": [],
        "orientation": [],
        "angular_velocity": [],
        "motor_speeds": [],
        "target_height": [],
        "target_orientation": [],
    }

    saturated_steps = 0
    steps = int(duration / DT)

    for step in range(steps):
        time = step * DT
        target_height, target_orientation = target_function(time)
        target_orientation = np.asarray(target_orientation, dtype=float)

        base_motor_speeds = height_controller.calculate_motor_speed(
            target_height=target_height,
            state=state,
            dt=DT,
        )

        state.motor_speeds = orientation_motor_speeds(
            controller=orientation_controller,
            base_motor_speeds=base_motor_speeds,
            target_orientation=target_orientation,
            state=state,
            dt=DT,
        )

        if np.any(
            (state.motor_speeds <= 1.0e-9)
            | (state.motor_speeds >= MAX_MOTOR_SPEED - 1.0e-9)
        ):
            saturated_steps += 1

        drone.update_state(state=state, dt=DT)

        state_values = np.concatenate(
            [
                state.position,
                state.velocity,
                state.orientation,
                state.angular_velocity,
            ]
        )

        if (
            not np.all(np.isfinite(state_values))
            or np.linalg.norm(state.position) > 1.0e3
            or np.linalg.norm(state.orientation) > 20.0
        ):
            return None

        if record:
            history["time"].append(time + DT)
            history["position"].append(state.position.copy())
            history["velocity"].append(state.velocity.copy())
            history["orientation"].append(state.orientation.copy())
            history["angular_velocity"].append(
                state.angular_velocity.copy()
            )
            history["motor_speeds"].append(state.motor_speeds.copy())
            history["target_height"].append(float(target_height))
            history["target_orientation"].append(
                target_orientation.copy()
            )

    if not record:
        # Record only the final values needed by a caller that requested no
        # detailed history.
        history["position"] = [state.position.copy()]
        history["velocity"] = [state.velocity.copy()]
        history["orientation"] = [state.orientation.copy()]
        history["angular_velocity"] = [state.angular_velocity.copy()]

    history["saturation_fraction"] = saturated_steps / steps
    return history


def constant_target(height, orientation):
    orientation = np.asarray(orientation, dtype=float)

    def target_function(_time):
        return height, orientation

    return target_function


def score_case(history, target_height, target_orientation, axis=None):
    if history is None:
        return float("inf")

    position = np.asarray(history["position"])
    velocity = np.asarray(history["velocity"])
    orientation = np.asarray(history["orientation"])
    angular_velocity = np.asarray(history["angular_velocity"])

    height_error = target_height - position[:, 2]
    saturation_penalty = 200.0 * history["saturation_fraction"]

    if axis is None:
        # Altitude tuning: reward small error, little overshoot, low vertical
        # velocity, and a settled final state.
        overshoot = max(0.0, np.max(position[:, 2]) - target_height)
        return (
            8.0 * np.mean(height_error**2)
            + 2.0 * abs(height_error[-1])
            + 1.5 * overshoot**2
            + 0.25 * np.mean(velocity[:, 2] ** 2)
            + 0.5 * abs(velocity[-1, 2])
            + saturation_penalty
        )

    target_orientation = np.asarray(target_orientation, dtype=float)
    angle_error = target_orientation[axis] - orientation[:, axis]

    # Orientation tuning also penalizes altitude loss and motion in the other
    # axes so an axis cannot look good by destabilizing the rest of the drone.
    other_axes = [index for index in range(3) if index != axis]

    return (
        250.0 * np.mean(angle_error**2)
        + 500.0 * abs(angle_error[-1]) ** 2
        + 2.0 * np.mean(angular_velocity[:, axis] ** 2)
        + 80.0 * np.mean(orientation[:, other_axes] ** 2)
        + 3.0 * np.mean(height_error**2)
        + saturation_penalty
    )


def search_gains(name, initial, bounds, evaluate, rng, trials):
    """Random search followed by two progressively smaller local searches."""
    best_gains = np.asarray(initial, dtype=float).copy()
    best_score = evaluate(best_gains)

    low = bounds[:, 0].astype(float).copy()
    high = bounds[:, 1].astype(float).copy()
    trials_per_round = max(1, trials // 3)

    for round_number in range(3):
        for _ in range(trials_per_round):
            candidate = rng.uniform(low, high)

            # Explicitly test PI-disabled candidates frequently. Many simple
            # undisturbed simulations do not need an integral term.
            if rng.random() < 0.35:
                candidate[1] = 0.0

            candidate_score = evaluate(candidate)

            if candidate_score < best_score:
                best_score = candidate_score
                best_gains = candidate.copy()

        full_width = bounds[:, 1] - bounds[:, 0]
        new_width = full_width * (0.30 / (round_number + 1))
        low = np.maximum(bounds[:, 0], best_gains - new_width)
        high = np.minimum(bounds[:, 1], best_gains + new_width)

        print(
            f"{name}: round {round_number + 1}/3, "
            f"best score={best_score:.6f}, "
            f"gains={np.round(best_gains, 5)}"
        )

    return best_gains, best_score


def tune_controllers():
    rng = np.random.default_rng(RANDOM_SEED)
    zero = np.zeros(3, dtype=float)

    def evaluate_height(candidate):
        history = run_simulation(
            height_gains=candidate,
            roll_gains=INITIAL_ROLL_GAINS,
            pitch_gains=INITIAL_PITCH_GAINS,
            yaw_gains=INITIAL_YAW_GAINS,
            duration=6.0,
            target_function=constant_target(TARGET_HEIGHT, zero),
            record=True,
        )
        return score_case(history, TARGET_HEIGHT, zero)

    height_gains, height_score = search_gains(
        "Height",
        INITIAL_HEIGHT_GAINS,
        HEIGHT_BOUNDS,
        evaluate_height,
        rng,
        TRIALS_PER_CONTROLLER,
    )

    optimized = {
        "roll": INITIAL_ROLL_GAINS.copy(),
        "pitch": INITIAL_PITCH_GAINS.copy(),
        "yaw": INITIAL_YAW_GAINS.copy(),
    }

    axis_settings = [
        ("roll", 0, np.radians([10.0, 0.0, 0.0]), ROLL_BOUNDS),
        ("pitch", 1, np.radians([0.0, 10.0, 0.0]), PITCH_BOUNDS),
        ("yaw", 2, np.radians([0.0, 0.0, 15.0]), YAW_BOUNDS),
    ]

    axis_scores = {}

    for name, axis, target_orientation, bounds in axis_settings:
        def evaluate_axis(candidate, name=name, axis=axis,
                          target_orientation=target_orientation):
            gains = {key: value.copy() for key, value in optimized.items()}
            gains[name] = candidate

            history = run_simulation(
                height_gains=height_gains,
                roll_gains=gains["roll"],
                pitch_gains=gains["pitch"],
                yaw_gains=gains["yaw"],
                duration=5.0,
                target_function=constant_target(
                    TARGET_HEIGHT,
                    target_orientation,
                ),
                record=True,
            )

            return score_case(
                history,
                TARGET_HEIGHT,
                target_orientation,
                axis=axis,
            )

        gains, score = search_gains(
            name.capitalize(),
            optimized[name],
            bounds,
            evaluate_axis,
            rng,
            TRIALS_PER_CONTROLLER,
        )

        optimized[name] = gains
        axis_scores[name] = score

    return {
        "height": height_gains,
        "roll": optimized["roll"],
        "pitch": optimized["pitch"],
        "yaw": optimized["yaw"],
        "scores": {
            "height": height_score,
            **axis_scores,
        },
    }


def maneuver_target(time):
    """Exercise roll, pitch, and yaw one at a time after takeoff."""
    if time < 2.0:
        angles = [0.0, 0.0, 0.0]
    elif time < 4.0:
        angles = [10.0, 0.0, 0.0]
    elif time < 6.0:
        angles = [0.0, -8.0, 0.0]
    elif time < 8.0:
        angles = [0.0, 0.0, 15.0]
    else:
        angles = [0.0, 0.0, 0.0]

    return TARGET_HEIGHT, np.radians(angles)


def save_results(gains, history):
    current_gains = {
        "height": INITIAL_HEIGHT_GAINS,
        "roll": INITIAL_ROLL_GAINS,
        "pitch": INITIAL_PITCH_GAINS,
        "yaw": INITIAL_YAW_GAINS,
    }

    recommended_gains = {
        name: np.asarray(gains[name], dtype=float)
        for name in current_gains
    }

    adjustments = {
        name: recommended_gains[name] - current_gains[name]
        for name in current_gains
    }

    output = {
        "gain_order": ["Kp", "Ki", "Kd"],
        "current_gains": {
            name: values.tolist()
            for name, values in current_gains.items()
        },
        "recommended_adjustments": {
            name: values.tolist()
            for name, values in adjustments.items()
        },
        "recommended_new_gains": {
            name: values.tolist()
            for name, values in recommended_gains.items()
        },
        "scores": {
            name: float(score)
            for name, score in gains["scores"].items()
        },
    }

    Path("optimized_pid_gains.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    time = np.asarray(history["time"])
    position = np.asarray(history["position"])
    velocity = np.asarray(history["velocity"])
    orientation = np.degrees(np.asarray(history["orientation"]))
    target_orientation = np.degrees(
        np.asarray(history["target_orientation"])
    )
    motor_speeds = np.asarray(history["motor_speeds"])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(time, position[:, 2], label="Height")
    axes[0, 0].axhline(
        TARGET_HEIGHT,
        color="black",
        linestyle="--",
        label="Target",
    )
    axes[0, 0].set_title("Altitude")
    axes[0, 0].set_ylabel("Meters")
    axes[0, 0].legend()

    labels = ["Roll", "Pitch", "Yaw"]
    for axis, label in enumerate(labels):
        axes[0, 1].plot(time, orientation[:, axis], label=label)
        axes[0, 1].plot(
            time,
            target_orientation[:, axis],
            linestyle="--",
            alpha=0.7,
        )
    axes[0, 1].set_title("Orientation and Targets")
    axes[0, 1].set_ylabel("Degrees")
    axes[0, 1].legend()

    for motor in range(4):
        axes[1, 0].plot(
            time,
            motor_speeds[:, motor],
            label=f"Motor {motor + 1}",
        )
    axes[1, 0].set_title("Motor Speeds")
    axes[1, 0].set_ylabel("rad/s")
    axes[1, 0].legend()

    axes[1, 1].plot(time, velocity[:, 2])
    axes[1, 1].axhline(0.0, color="black", linestyle="--")
    axes[1, 1].set_title("Vertical Velocity")
    axes[1, 1].set_ylabel("m/s")

    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(True)

    figure.tight_layout()
    figure.savefig("auto_tune_results.png", dpi=200)
    plt.show()


def main():
    print("Starting staged PID search...")
    print(f"Trials per controller: {TRIALS_PER_CONTROLLER}")

    gains = tune_controllers()

    current_gains = {
        "Height": INITIAL_HEIGHT_GAINS,
        "Roll": INITIAL_ROLL_GAINS,
        "Pitch": INITIAL_PITCH_GAINS,
        "Yaw": INITIAL_YAW_GAINS,
    }

    recommended_gains = {
        "Height": gains["height"],
        "Roll": gains["roll"],
        "Pitch": gains["pitch"],
        "Yaw": gains["yaw"],
    }

    print("\nPID recommendations [Kp, Ki, Kd]")
    print("----------------------------------")

    for name in current_gains:
        current = current_gains[name]
        recommended = recommended_gains[name]
        adjustment = recommended - current

        print(f"\n{name}")
        print("  Current:     ", np.round(current, 6))
        print("  Adjustment:  ", np.round(adjustment, 6))
        print("  Recommended: ", np.round(recommended, 6))

    history = run_simulation(
        height_gains=gains["height"],
        roll_gains=gains["roll"],
        pitch_gains=gains["pitch"],
        yaw_gains=gains["yaw"],
        duration=10.0,
        target_function=maneuver_target,
        record=True,
    )

    if history is None:
        raise RuntimeError(
            "The combined validation maneuver became unstable. "
            "Reduce the search ranges or controller correction limit."
        )

    save_results(gains, history)
    print("\nSaved optimized_pid_gains.json")
    print("Saved auto_tune_results.png")


if __name__ == "__main__":
    main()