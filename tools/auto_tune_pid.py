import json
from pathlib import Path
import sys

# Support both module execution and running this file directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_sim.configuration.project_paths import output_path

import numpy as np

from drone_sim.physics.dynamics import DroneState
from drone_sim.physics.attitude import wrap_angle


# main.py supplies mission targets and modes; parameters.py supplies gains and shared settings.
from drone_sim import main as mission
from drone_sim.configuration import parameters
DT = parameters.DT
TARGET_HEIGHT = mission.MINIMUM_FLIGHT_ALTITUDE
TARGET_YAW_DEGREES = mission.TARGET_YAW
POSITION_TEST_DISTANCE = 1.0
MAX_TILT_DEGREES = parameters.MAX_TILT_DEGREES
MAX_MOTOR_SPEED = parameters.MAX_MOTOR_SPEED
HOVER_SPEED = parameters.HOVER_SPEED
TRIALS_PER_CONTROLLER = 90
RANDOM_SEED = 7


# Search limits: [minimum, maximum] for Kp, Ki, and Kd.
HEIGHT_BOUNDS = np.array(
    [[20.0, 50.0], [0.0, 2.0], [25.0, 65.0]],
    dtype = float,
)

ROLL_BOUNDS = np.array(
    [[3.0, 14.0], [0.0, 0.20], [2.0, 10.0]],
    dtype = float,
)

PITCH_BOUNDS = np.array(
    [[3.0, 14.0], [0.0, 0.20], [2.0, 10.0]],
    dtype = float,
)

YAW_BOUNDS = np.array(
    [[8.0, 24.0], [0.0, 0.20], [5.0, 22.0]],
    dtype = float,
)

POSITION_BOUNDS = np.array(
    [[0.002, 0.060], [0.0, 0.010], [0.010, 0.180]],
    dtype = float,
)


def create_drone(physics_backend = "simulation", terrain = None):

    from drone_sim.world.terrain import FlatTerrain
    return mission.create_drone(FlatTerrain() if terrain is None else terrain, physics_backend = physics_backend)


def create_controllers(gains):

    return mission.create_controllers(gains)


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


def run_simulation(gains, duration, fixed_orientation = None, target_position = None,
                   physics_backend = "simulation", start_hover = True, wind = (0.0, 0.0, 0.0), yaw = 0.0):
    """Fixed-duration excitation trial with explicit backend and owned cleanup."""

    from drone_sim.validation import positive, vector
    from drone_sim.world.environment import BasicEnvironment

    if fixed_orientation is not None and target_position is not None:
        raise ValueError("Use fixed_orientation or target_position, not both.")
    dt = positive(DT, "Time step")
    duration = positive(duration, "Trial duration")
    steps = int(duration / dt)
    if steps < 1:
        raise ValueError("Trial duration must include at least one time step.")
    fixed = np.zeros(3) if fixed_orientation is None else vector(fixed_orientation, 3, "Target orientation")
    target = np.zeros(2) if target_position is None else vector(target_position, 2, "Target position")
    wind = vector(wind, 3, "Trial wind")
    environment = BasicEnvironment("constant" if np.any(wind) else "none", wind, parameters.WIND_FORCE_COEFFICIENT)
    drone = create_drone(physics_backend)
    try:
        state = DroneState()
        state.position[2] = TARGET_HEIGHT if start_hover else drone.get_ground_height(0, 0)
        if physics_backend == "gym_pybullet" and not start_hover:
            state.position[:] = drone.initial_position
        if start_hover:
            state.motor_speeds.fill(HOVER_SPEED)
        if physics_backend == "gym_pybullet":
            drone.reset(state)
        controllers = create_controllers(gains)
        history = empty_history()
        for step in range(steps):
            try:
                desired = mission.control_step(
                    drone, state, controllers, environment, TARGET_HEIGHT, target,
                    yaw, dt, step * dt, fixed_orientation = fixed if target_position is None else None)
            except (FloatingPointError, ValueError):
                return None
            for key, value in {
                "time": (step + 1) * dt, "position": state.position.copy(),
                "velocity": state.velocity.copy(), "orientation": state.orientation.copy(),
                "angular_velocity": state.angular_velocity.copy(), "motor_speeds": state.motor_speeds.copy(),
                "target_height": TARGET_HEIGHT, "target_orientation": desired.copy(), "target_position": target.copy()
            }.items():
                history[key].append(value)
        from tools.pid_scenarios import control_metrics
        history["saturation_fraction"], history["motor_slew_rms_per_s"] = control_metrics(history["motor_speeds"], history["time"])
        return history
    finally:
        if hasattr(drone, "close"):
            drone.close()


def history_arrays(history):

    return {
        "time": np.asarray(history["time"], dtype = float),
        "position": np.asarray(
            history["position"],
            dtype = float,
        ),
        "velocity": np.asarray(
            history["velocity"],
            dtype = float,
        ),
        "orientation": np.asarray(
            history["orientation"],
            dtype = float,
        ),
        "angular_velocity": np.asarray(
            history["angular_velocity"],
            dtype = float,
        ),
        "motor_speeds": np.asarray(
            history["motor_speeds"],
            dtype = float,
        ),
        "target_height": np.asarray(
            history["target_height"],
            dtype = float,
        ),
        "target_orientation": np.asarray(
            history["target_orientation"],
            dtype = float,
        ),
        "target_position": np.asarray(
            history["target_position"],
            dtype = float,
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
 + 10.0 * height_error[-1]**2
 + 2.0 * overshoot**2
 + 0.30 * np.mean(vertical_velocity**2)
 + 1.0 * vertical_velocity[-1]**2
 + saturation_penalty
    )


def score_orientation(history, target_orientation, axis):

    if history is None:
        return float("inf")

    data = history_arrays(history)
    target_orientation = np.asarray(
        target_orientation,
        dtype = float,
    )

    angle = data["orientation"][:, axis]
    angle_error = target_orientation[axis] - angle
    if axis == 2:
        angle_error = wrap_angle(angle_error)
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
 + 700.0 * angle_error[-1]**2
 + 1200.0 * overshoot**2
 + 12.0 * np.mean(angular_velocity**2)
 + 80.0 * angular_velocity[-1]**2
 + 80.0 * np.mean(
            data["orientation"][:, other_axes]**2
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
        dtype = float,
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
    attitude_tracking_error[:, 2] = wrap_angle(attitude_tracking_error[:, 2])

    height_error = TARGET_HEIGHT - position[:, 2]

    return (
        15.0 * np.mean(axis_error**2)
 + 35.0 * np.mean(late_error**2)
 + 120.0 * axis_error[-1]**2
 + 4.0 * np.mean(velocity[:, axis]**2)
 + 35.0 * velocity[-1, axis]**2
 + 30.0 * np.mean(other_error**2)
 + 80.0 * other_error[-1]**2
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
        dtype = float,
    ).copy()

    # Retuning may start from gains produced by a wider earlier search.
    # Include that baseline so narrowing cannot create an invalid interval.
    bounds = np.asarray(bounds, dtype = float).copy()
    bounds[:, 0] = np.minimum(bounds[:, 0], best_gains)
    bounds[:, 1] = np.maximum(bounds[:, 1], best_gains)

    best_score = evaluate(best_gains)
    low = bounds[:, 0].copy()
    high = bounds[:, 1].copy()

    trials_per_round, extra_trials = divmod(TRIALS_PER_CONTROLLER, 3)

    for round_index in range(3):
        for _ in range(trials_per_round + (round_index < extra_trials)):
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

    return mission.current_gains()


def trial_penalty(history, errors, speeds, tolerance):

    from tools.pid_scenarios import settling_time
    if history is None:
        return float("inf")
    time = np.asarray(history["time"])
    settled = settling_time(time, (np.abs(errors) <= tolerance) & (np.abs(speeds) <= 0.15))
    return 2 * (settled if settled is not None else 2 * time[-1]) + 0.05 * history["motor_slew_rms_per_s"]**2


def tune_controllers(physics_backend = "simulation"):
    """Tune attitude first, then altitude and position using both command signs."""

    gains = initial_gain_dictionary()
    rng = np.random.default_rng(RANDOM_SEED)
    scores = {}
    settings = [("roll", 0, ROLL_BOUNDS), ("pitch", 1, PITCH_BOUNDS),
                ("yaw", 2, YAW_BOUNDS), ("height", 2, HEIGHT_BOUNDS),
                ("x", 0, POSITION_BOUNDS), ("y", 1, POSITION_BOUNDS)]

    for name, axis, bounds in settings:
        def evaluate(candidate):
            trial = {key: np.asarray(value).copy() for key, value in gains.items()}
            trial[name] = candidate
            results = []
            for sign in (1, -1):
                wind = (0.0, 0.0, 0.0) if sign == 1 else (0.3, -0.2, 0.0)
                if name in ("roll", "pitch", "yaw"):
                    target = np.zeros(3)
                    target[axis] = np.radians(sign * (12 if name == "yaw" else 8))
                    history = run_simulation(trial, 5.0, fixed_orientation = target, physics_backend = physics_backend, wind = wind)
                    score = score_orientation(history, target, axis)
                    if history is not None:
                        errors = wrap_angle(np.asarray(history["orientation"])[:, axis] - target[axis])
                        score += trial_penalty(history, errors, np.asarray(history["angular_velocity"])[:, axis], np.radians(1))
                elif name == "height":
                    # Ground launch and hover recovery exercise different altitude errors.
                    history = run_simulation(trial, 6.0, physics_backend = physics_backend, start_hover = sign == -1, wind = (0, 0, sign * 0.3))
                    score = score_height(history)
                    if history is not None:
                        score += trial_penalty(history, np.asarray(history["position"])[:, 2] - TARGET_HEIGHT, np.asarray(history["velocity"])[:, 2], 0.1)
                else:
                    target = np.zeros(2)
                    target[axis] = sign * POSITION_TEST_DISTANCE
                    history = run_simulation(trial, 12.0, target_position = target, physics_backend = physics_backend, wind = wind, yaw = np.radians(sign * 30))
                    score = score_position(history, target, axis)
                    if history is not None:
                        score += trial_penalty(history, np.asarray(history["position"])[:, axis] - target[axis], np.asarray(history["velocity"])[:, axis], 0.1)
                results.append(score)
            return float(np.mean(results) + 0.25 * np.max(results))

        gains[name], scores[name] = search_gains(name, gains[name], bounds, evaluate, rng)

    return {**gains, "scores": scores}


def json_safe(value):

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def metric_summary(suite):

    metrics = ("tracking_rmse_m", "overshoot_m", "settling_time_s", "saturation_fraction",
               "motor_slew_rms_per_s", "touchdown_speed_m_s", "final_error_m")
    summary = {"score": suite["score"], "success_rate": suite["success_rate"]}
    for metric in metrics:
        values = [case[metric] for case in suite["cases"] if case.get(metric) is not None]
        summary[metric] = float(np.mean(values)) if values else None
    summary["unsettled_or_failed_cases"] = sum(case.get("settling_time_s") is None for case in suite["cases"])
    return summary


def refine_joint(gains, training, baseline, evaluate, trials, rng):
    """Refine all 18 gains against training missions; never inspect holdouts."""

    names = ("roll", "pitch", "yaw", "height", "x", "y")
    configured = (ROLL_BOUNDS, PITCH_BOUNDS, YAW_BOUNDS, HEIGHT_BOUNDS, POSITION_BOUNDS, POSITION_BOUNDS)
    reference = np.concatenate([baseline[name] for name in names])
    # Allow local exploration above earlier stage bounds that baseline exceeded.
    lower = np.concatenate([np.minimum(bounds[:, 0], baseline[name] * 0.5) for name, bounds in zip(names, configured)])
    upper = np.concatenate([np.maximum(bounds[:, 1], baseline[name] * 1.5) for name, bounds in zip(names, configured)])
    best = np.concatenate([gains[name] for name in names])
    scale = np.maximum(np.abs(reference), (upper - lower) * 0.1)
    cache = {tuple(best): training}
    trace = []

    for index in range(trials):
        candidate = best.copy()
        if index < 4:
            # Coupled horizontal P/D moves exercise mission tracking directly.
            proportional, derivative = ((1.15, 1.05), (1.05, 1.15), (0.9, 1.1), (1.1, 0.9))[index]
            for controller in (4, 5):
                candidate[controller * 3] *= proportional
                candidate[controller * 3 + 2] *= derivative
        else:
            # Simultaneous perturbations explore interactions across controllers.
            radius = 0.15 * (1 - 0.65 * (index - 4) / max(1, trials - 4))
            candidate += rng.normal(size = len(best)) * scale * radius
        candidate = np.clip(candidate, lower, upper)
        proposal = {name: candidate[3 * i:3 * i + 3].copy() for i, name in enumerate(names)}
        cached = tuple(candidate) in cache
        result = cache.get(tuple(candidate))
        if result is None:
            result = evaluate(proposal)
            cache[tuple(candidate)] = result
        improved = result["score"] < training["score"] and result["success_rate"] >= training["success_rate"]
        previous_score = training["score"]
        if improved:
            best, training = candidate.copy(), result
        trace.append({"trial": index + 1, "gains": proposal, "training": result,
                      "accepted": bool(improved), "cached": cached, "previous_best_score": previous_score})
        print(f"Joint mission refinement {index + 1}/{trials}: candidate={result['score']:.6f}, best={training['score']:.6f}, accepted={improved}", flush = True)

    refined = {name: best[3 * i:3 * i + 3].copy() for i, name in enumerate(names)}
    bounds = {name: np.column_stack((lower[3 * i:3 * i + 3], upper[3 * i:3 * i + 3])) for i, name in enumerate(names)}
    return refined, training, {"trials": trace, "bounds": bounds, "controller_order": names}


def run_workflow(physics_backend = "simulation", validation_backends = ("simulation", "gym_pybullet"), seeds = None, joint_trials = 12, reuse_staged = None):
    """Search training cases, freeze one candidate, then evaluate unseen cases."""

    from dataclasses import asdict
    from tools.pid_scenarios import TRAINING_SCENARIOS, VALIDATION_SCENARIOS, evaluate_suite, validate_candidate

    if physics_backend not in ("simulation", "gym_pybullet"):
        raise ValueError("Tuning requires one explicit backend.")
    if not validation_backends or any(name not in ("simulation", "gym_pybullet") for name in validation_backends):
        raise ValueError("Validation backends must be simulation and/or gym_pybullet.")
    if isinstance(joint_trials, bool) or not isinstance(joint_trials, int) or joint_trials < 0:
        raise ValueError("Joint trials must be a nonnegative integer.")
    # Fail early on missing optional dependencies, before an expensive search.
    for backend in set(validation_backends) | {physics_backend}:
        drone = create_drone(backend)
        if hasattr(drone, "close"):
            drone.close()

    baseline = initial_gain_dictionary()
    baseline_training = evaluate_suite(baseline, TRAINING_SCENARIOS, (physics_backend,))
    best, best_training = baseline, baseline_training
    searches = []
    global RANDOM_SEED
    original_seed = RANDOM_SEED
    seeds = [RANDOM_SEED] if seeds is None else list(seeds)
    if not seeds:
        raise ValueError("At least one search seed is required.")
    reused = None
    if reuse_staged is not None:
        reused = json.loads(Path(reuse_staged).read_text(encoding = "utf-8"))
        checks = {"tuning_backend": physics_backend, "baseline_gains": json_safe(baseline),
                  "parameters": json_safe({key: value for key, value in vars(parameters).items() if key.isupper()}),
                  "training_scenarios": json_safe([asdict(case) for case in TRAINING_SCENARIOS]), "random_seeds": seeds}
        if any(reused.get(key) != value for key, value in checks.items()):
            raise ValueError("Saved staged search does not match the current backend, gains, settings, scenarios or seeds. Run without --reuse-staged.")
    try:
        for seed in seeds:
            RANDOM_SEED = seed
            staged = None if reused is not None else tune_controllers(physics_backend)
            for fraction in (1.0, 0.5):
                if reused is not None:
                    previous = next(entry for entry in reused["searches"] if entry["seed"] == seed and entry["fraction"] == fraction)
                    candidate = {key: np.asarray(previous["gains"][key], dtype = float) for key in baseline}
                    stage_scores = previous["stage_scores"]
                else:
                    candidate = {key: baseline[key] + fraction * (staged[key] - baseline[key]) for key in baseline}
                    stage_scores = staged["scores"]
                training = evaluate_suite(candidate, TRAINING_SCENARIOS, (physics_backend,))
                searches.append({"seed": seed, "fraction": fraction, "gains": candidate, "training": training, "stage_scores": stage_scores})
                if training["score"] < best_training["score"] and training["success_rate"] >= baseline_training["success_rate"]:
                    best, best_training = candidate, training
    finally:
        RANDOM_SEED = original_seed

    before_joint_score = best_training["score"]
    best, best_training, joint = refine_joint(
        best, best_training, baseline,
        lambda gains: evaluate_suite(gains, TRAINING_SCENARIOS, (physics_backend,)),
        joint_trials, np.random.default_rng(seeds[0] + 1009))
    joint["starting_score"] = before_joint_score
    joint["final_score"] = best_training["score"]

    # No further candidate selection follows this held-out evaluation.
    validation = validate_candidate(baseline, best, validation_backends)
    changed = any(not np.array_equal(best[key], baseline[key]) for key in baseline)
    accepted = bool(changed and validation["candidate_accepted"])
    validation["heldout_gate_passed"] = validation["candidate_accepted"]
    validation["candidate_accepted"] = accepted
    selected = best if accepted else baseline
    before = metric_summary(validation["baseline"])
    after = metric_summary(validation["candidate"])
    report = {
        "schema_version": 2, "gain_order": ["Kp", "Ki", "Kd"],
        "stage_order": ["roll", "pitch", "yaw", "height", "x", "y"],
        "tuning_backend": physics_backend, "validation_backends": list(validation_backends),
        "random_seeds": seeds, "trials_per_controller": reused["trials_per_controller"] if reused is not None else TRIALS_PER_CONTROLLER,
        "training_scenarios": [asdict(case) for case in TRAINING_SCENARIOS],
        "validation_scenarios": [asdict(case) for case in VALIDATION_SCENARIOS],
        "baseline_gains": baseline, "candidate_gains": best, "selected_gains": selected,
        "baseline_training": baseline_training, "candidate_training": best_training,
        "before_after": {"baseline": before, "candidate": after,
                         "candidate_minus_baseline": {key: after[key] - before[key] if after[key] is not None and before[key] is not None else None for key in before}},
        "candidate_changed": changed,
        "joint_refinement": joint,
        "reused_staged_report": str(reuse_staged) if reuse_staged is not None else None,
        "validation": validation, "candidate_accepted": accepted, "searches": searches,
        "search_bounds": {"roll": ROLL_BOUNDS, "pitch": PITCH_BOUNDS, "yaw": YAW_BOUNDS,
                          "height": HEIGHT_BOUNDS, "x": POSITION_BOUNDS, "y": POSITION_BOUNDS},
        "parameters": {key: value for key, value in vars(parameters).items() if key.isupper()},
        "score_weights": {"tracking_rmse_squared": 15, "final_error_squared": 120,
                          "overshoot_squared": 20, "settling_seconds": 0.1, "saturation_fraction": 50,
                          "normalized_motor_slew_squared": 0.05, "impact_above_0_15_m_s_squared": 200,
                          "incomplete_or_failed_mission": 10000},
        "scope": "Recommendations only; held-out validation never guides search. No global optimality claim."
    }
    path = output_path("pid_tuning_report.json")
    path.write_text(json.dumps(json_safe(report), indent = 2, allow_nan = False), encoding = "utf-8")
    output_path("optimized_pid_gains.json").write_text(json.dumps(json_safe({
        "gain_order": ["Kp", "Ki", "Kd"], "recommended_new_gains": selected,
        "candidate_accepted": accepted, "source_report": path.name}), indent = 2, allow_nan = False), encoding = "utf-8")
    print(f"Candidate accepted: {accepted}; current parameters have not been changed.")
    if not changed:
        print("No training candidate improved on the baseline; baseline retained before validation.")
    for label, result in (("Baseline", validation["baseline"]), ("Candidate" if changed else "Retained baseline", validation["candidate"])):
        print(f"{label}: held-out score={result['score']:.3f}, success={result['success_rate']:.0%}")
    print(f"Saved {path}")
    return report


def main():

    global TRIALS_PER_CONTROLLER
    import argparse
    parser = argparse.ArgumentParser(description = "Attitude-first PID tuning with training/validation scenarios.")
    parser.add_argument("--backend", choices = ("simulation", "gym_pybullet"), default = "simulation")
    parser.add_argument("--validation-backends", choices = ("simulation", "gym_pybullet"), nargs = "+", default = ["simulation", "gym_pybullet"])
    parser.add_argument("--trials", type = int, default = TRIALS_PER_CONTROLLER)
    parser.add_argument("--seed", type = int, default = RANDOM_SEED)
    parser.add_argument("--joint-trials", type = int, default = 12, help = "Mission-level joint refinement candidates; 0 disables refinement.")
    parser.add_argument("--reuse-staged", type = Path, help = "Reuse staged candidates from a matching report, then rerun training and joint refinement.")
    args = parser.parse_args()
    if args.trials < 3:
        parser.error("--trials must be at least 3")
    if args.joint_trials < 0:
        parser.error("--joint-trials must be nonnegative")
    TRIALS_PER_CONTROLLER = args.trials
    run_workflow(args.backend, tuple(args.validation_backends), [args.seed], args.joint_trials, args.reuse_staged)


if __name__ == "__main__":
    main()
