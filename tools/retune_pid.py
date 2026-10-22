import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from drone_sim.configuration.project_paths import PROJECT_ROOT
from tools import auto_tune_pid as tuner


def apply_gains(gains):

    path = PROJECT_ROOT / "drone_sim" / "configuration" / "parameters.py"
    source = path.read_text(encoding = "utf-8")
    for name, value in zip(["KP", "KI", "KD"], gains["height"]):
        source, count = re.subn(rf"^{name} = .*?$", f"{name} = {float(value):.9f}", source, count = 1, flags = re.M)
        if count != 1:
            raise RuntimeError(f"Could not locate {name} in parameters.py.")
    for key, name in [("roll", "ROLL_GAINS"), ("pitch", "PITCH_GAINS"),
                      ("yaw", "YAW_GAINS"), ("x", "X_POSITION_GAINS"), ("y", "Y_POSITION_GAINS")]:
        values = ", ".join(f"{float(value):.9f}" for value in gains[key])
        source, count = re.subn(rf"^{name} = .*?$", f"{name} = np.array([{values}])", source, count = 1, flags = re.M)
        if count != 1:
            raise RuntimeError(f"Could not locate {name} in parameters.py.")
    path.write_bytes(source.encode("utf-8"))


def main(expanded = False, physics_backend = "simulation", trials = 120,
         validation_backends = ("simulation", "gym_pybullet"), apply = False, seeds = (7, 29), joint_trials = 12):

    if trials < 3:
        raise ValueError("At least 3 trials per controller are required.")
    tuner.TRIALS_PER_CONTROLLER = trials
    if expanded:
        tuner.HEIGHT_BOUNDS = np.array([[50, 250], [0, 4], [25, 120]], dtype = float)
        tuner.ROLL_BOUNDS = np.array([[15, 70], [0, 0.2], [4, 25]], dtype = float)
        tuner.PITCH_BOUNDS = tuner.ROLL_BOUNDS.copy()
        tuner.YAW_BOUNDS = np.array([[20, 90], [0, 0.2], [10, 65]], dtype = float)
        tuner.POSITION_BOUNDS = np.array([[0.02, 0.35], [0, 0.01], [0.06, 0.6]], dtype = float)
    report = tuner.run_workflow(physics_backend, validation_backends, seeds, joint_trials = joint_trials)
    if apply and report["candidate_accepted"]:
        apply_gains(report["selected_gains"])
        print("Applied validated gains to parameters.py.")
    elif apply:
        print("Candidate failed validation or did not improve training; current gains retained.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Broader PID tuning with independent held-out validation.")
    parser.add_argument("--expanded", action = "store_true")
    parser.add_argument("--backend", choices = ("simulation", "gym_pybullet"), default = "simulation")
    parser.add_argument("--validation-backends", choices = ("simulation", "gym_pybullet"), nargs = "+", default = ["simulation", "gym_pybullet"])
    parser.add_argument("--trials", type = int, default = 120)
    parser.add_argument("--seeds", type = int, nargs = "+", default = [7, 29])
    parser.add_argument("--apply", action = "store_true", help = "Apply gains only if held-out validation passes.")
    parser.add_argument("--joint-trials", type = int, default = 12)
    args = parser.parse_args()
    main(args.expanded, args.backend, args.trials, tuple(args.validation_backends), args.apply, tuple(args.seeds), args.joint_trials)
