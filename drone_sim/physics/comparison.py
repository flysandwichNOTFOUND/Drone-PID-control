"""Scaffold for comparing custom physics with gym-pybullet-drones.

Implement after gym_backend.py. Both models must share physical parameters,
initial conditions and timestamps. Each closed-loop run needs its own PID
controllers and navigation state. This module does not change the main switch.
"""


def run_comparison(duration, mode = "same_commands"):
    """Input: duration in seconds; same_commands or same_mission mode.

    Output (planned): dict with time, simulation/PyBullet histories and errors.
    Source: parameters.py, main's mission settings, and both backend states.
    Caller: future comparison entry point or an interactive Python session.
    Planned: same_commands sends one motor-command stream to both models;
    same_mission uses independent controllers to follow the same mission.
    """
    raise NotImplementedError("Implement after the gym backend and model mapping.")
