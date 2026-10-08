"""Broader reproducible PID search, followed by baseline and mission checks.

Run: python retune_pid.py
Writes a tuning report and applies the best validated gains to main.py.
"""
import json
import argparse
from pathlib import Path
import re

import plotting
plotting.use_backend("Agg")
import numpy as np

import auto_tune_pid as tuner
import main as mission


def flight_metrics(gains, yaw = 0.0, landing = False):

    state, data = mission.run_simulation(
        gains = gains, target_yaw = yaw, include_landing = landing,
        duration = 70.0 if landing else mission.SIMULATION_TIME, verbose = False)
    speeds = np.asarray(data["motor_speed"])
    targets = np.asarray(data["target_position_history"])
    positions = np.asarray(data["position"])
    indices = np.asarray(data["waypoint_index"])
    final_index = len(data["waypoints"]) - 1
    arrived = ((indices == final_index)
               & (np.asarray(data["distance_to_target"]) <= mission.ARRIVAL_TOLERANCE))
    return {"mission_complete": bool(data["mission_complete"]),
            "landed": bool(data["landed"]),
            "final_error_m": float(np.linalg.norm(state.position - data["waypoints"][-1])),
            "final_speed_m_s": float(np.linalg.norm(state.velocity)),
            "tracking_rmse_m": float(np.sqrt(np.mean(np.sum((targets - positions)**2, axis = 1)))),
            "first_final_target_arrival_s": float(np.asarray(data["time"])[arrived][0]) if np.any(arrived) else None,
            "saturation_fraction": float(np.mean(np.any((speeds <= 0) | (speeds >= mission.MAX_MOTOR_SPEED), axis = 1))),
            "max_height_m": float(np.max(positions[:, 2]))}


def apply_gains(gains):

    path = Path(__file__).with_name("main.py")
    source = path.read_text(encoding = "utf-8")
    for name, value in zip(["KP", "KI", "KD"], gains["height"]):
        source, count = re.subn(rf"^{name} = .*?$", f"{name} = {float(value):.9f}", source, count = 1, flags = re.M)
        if count != 1:
            raise RuntimeError(f"Could not locate {name} in main.py.")
    for key, name in [("roll", "ROLL_GAINS"), ("pitch", "PITCH_GAINS"),
                      ("yaw", "YAW_GAINS"), ("x", "X_POSITION_GAINS"), ("y", "Y_POSITION_GAINS")]:
        values = ", ".join(f"{float(value):.9f}" for value in gains[key])
        source, count = re.subn(rf"^{name} = .*?$", f"{name} = np.array([{values}])", source, count = 1, flags = re.M)
        if count != 1:
            raise RuntimeError(f"Could not locate {name} in main.py.")
    path.write_bytes(source.encode("utf-8"))


def main(expanded = False):

    baseline = mission.current_gains()
    # Broaden the original ranges: several existing gains were near their upper bounds.
    tuner.HEIGHT_BOUNDS = np.array([[20, 100], [0, 2], [15, 80]], dtype = float)
    tuner.ROLL_BOUNDS = np.array([[3, 30], [0, 0.2], [2, 16]], dtype = float)
    tuner.PITCH_BOUNDS = tuner.ROLL_BOUNDS.copy()
    tuner.YAW_BOUNDS = np.array([[8, 40], [0, 0.2], [5, 35]], dtype = float)
    tuner.POSITION_BOUNDS = np.array([[0.005, 0.18], [0, 0.01], [0.03, 0.35]], dtype = float)
    tuner.TRIALS_PER_CONTROLLER = 120
    seeds = [7, 29]
    if expanded:
        seeds = [73]
        tuner.HEIGHT_BOUNDS = np.array([[50, 250], [0, 4], [25, 120]], dtype = float)
        tuner.ROLL_BOUNDS = np.array([[15, 70], [0, 0.2], [4, 25]], dtype = float)
        tuner.PITCH_BOUNDS = tuner.ROLL_BOUNDS.copy()
        tuner.YAW_BOUNDS = np.array([[20, 90], [0, 0.2], [10, 65]], dtype = float)
        tuner.POSITION_BOUNDS = np.array([[0.02, 0.35], [0, 0.01], [0.06, 0.6]], dtype = float)
        previous = Path("pid_retuning_report.json")
        if previous.exists():
            Path("pid_retuning_report_initial.json").write_bytes(previous.read_bytes())
    target = np.array([tuner.POSITION_TEST_DISTANCE] * 2)
    baseline_history = tuner.run_simulation(baseline, 15, target_position = target)
    baseline_combined = float(tuner.combined_validation_score(baseline_history, target))
    baseline_route = tuner.validate_route(baseline)
    best = {**baseline, "scores": {}}
    best_quality = 2.0
    best_history = baseline_history
    best_validation = None
    trials = []
    for seed in seeds:
        tuner.RANDOM_SEED = seed
        print(f"Starting broad search, seed {seed}", flush = True)
        candidate = tuner.tune_controllers()
        # Check both the full candidate and interpolations to avoid losing useful
        # stage improvements when their combination regresses the route.
        for fraction in [1.0, 0.75, 0.5, 0.25]:
            blended = {key: baseline[key] + fraction * (candidate[key] - baseline[key]) for key in baseline}
            blended["scores"] = candidate["scores"]
            selected, history, score, validation = tuner.validate_recommendations(blended)
            entry = {"seed": seed, "candidate_fraction": fraction, "validation": validation,
                     "candidate_gains": {key: blended[key].tolist() for key in baseline}}
            trials.append(entry)
            print(f"Seed {seed}, fraction {fraction}: accepted={validation['candidate_accepted']}, "
                  f"combined={validation['candidate_combined_score']:.6f}, "
                  f"route={validation['candidate_route']['score']:.6f}", flush = True)
            if not validation["candidate_accepted"]:
                continue
            quality = score / baseline_combined + validation["candidate_route"]["score"] / baseline_route["score"]
            if quality >= best_quality:
                continue
            try:
                checks = {"yaw_90": flight_metrics(selected, np.pi / 2),
                          "landing": flight_metrics(selected, landing = True)}
            except (ValueError, FloatingPointError):
                entry["additional_checks_passed"] = False
                continue
            passed = checks["yaw_90"]["mission_complete"] and checks["landing"]["landed"]
            entry["additional_checks"] = checks
            entry["additional_checks_passed"] = passed
            if passed:
                best, best_quality, best_history, best_validation = selected, quality, history, validation
    report = {"gain_order": ["Kp", "Ki", "Kd"],
              "baseline_gains": {key: baseline[key].tolist() for key in baseline},
              "selected_gains": {key: best[key].tolist() for key in baseline},
              "search": {"seeds": seeds, "trials_per_controller_per_seed": 120,
                         "bounds": {key: value.tolist() for key, value in {
                             "height": tuner.HEIGHT_BOUNDS, "roll": tuner.ROLL_BOUNDS,
                             "pitch": tuner.PITCH_BOUNDS, "yaw": tuner.YAW_BOUNDS,
                             "x": tuner.POSITION_BOUNDS, "y": tuner.POSITION_BOUNDS}.items()}},
              "baseline_metrics": flight_metrics(baseline),
              "selected_metrics": flight_metrics(best),
              "selected_validation": best_validation, "trials": trials,
              "scope": "Best tested gains for configured model and scoring; no global optimality guarantee."}
    Path("pid_retuning_report.json").write_text(
        json.dumps(tuner.json_safe(report), indent = 2, allow_nan = False), encoding = "utf-8")
    if best_validation is None:
        print("No candidate passed every check; current gains retained.", flush = True)
        return
    # Save the tuner output before applying gains so its baseline remains accurate.
    tuner.save_json(best, float(tuner.combined_validation_score(best_history, target)), best_validation)
    plotting.plot_validation(best_history, target_height = tuner.TARGET_HEIGHT,
                             hover_speed = tuner.HOVER_SPEED, max_motor_speed = tuner.MAX_MOTOR_SPEED,
                             show = False)
    apply_gains(best)
    _, data = mission.run_simulation(gains = best, verbose = False)
    plotting.plot_data(data, show = False)
    plotting.plot_3d_trajectory(data, show = False)
    plotting.close_all()
    print("Applied validated gains to main.py", flush = True)
    print(json.dumps(report["selected_gains"], indent = 2), flush = True)
    print("METRICS", report["baseline_metrics"], report["selected_metrics"], flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--expanded", action = "store_true", help = "Search wider ranges around a previously tuned baseline.")
    main(expanded = parser.parse_args().expanded)
