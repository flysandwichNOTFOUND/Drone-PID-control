import numpy as np

from drone_sim.configuration import parameters
from drone_sim.physics.dynamics import DroneState


def run_comparison(duration = None, mode = "same_mission", gains = None, include_landing = None,
                   verbose = True, target_yaw = None, terrain = None, goal_position = None, initial_state = None, environment = None):
    """Run simulation first, then Gym from the same initial conditions.

    Return both complete histories and final states for future shared graphs.
    Final-position difference is Gym minus simulation, in world-frame metres.
    """

    from drone_sim.main import create_drone, create_terrain, run_simulation
    from drone_sim.physics.paired_training import PairedAccelerationCollector

    if mode != "same_mission":
        raise ValueError("Comparison currently supports same_mission mode only.")

    terrain = create_terrain() if terrain is None else terrain
    starting_state = DroneState() if initial_state is None else initial_state.copy()

    if initial_state is None:
        starting_state.position[2] = float(terrain.height(0, 0)) + parameters.GYM_BODY_HEIGHT / 2 + 0.01

    final_states, histories = {}, {}
    paired = PairedAccelerationCollector(create_drone(terrain, physics_backend = "simulation"))

    for backend in ("simulation", "gym_pybullet"):
        # A fresh run resets controllers, navigation, wind and logging.
        state, history = run_simulation(
            gains = gains, duration = duration, include_landing = include_landing,
            verbose = False, target_yaw = target_yaw, terrain = terrain,
            goal_position = goal_position, physics_backend = backend,
            initial_state = starting_state.copy(), environment = environment,
            step_observer = paired.record if backend == "gym_pybullet" else None)
        final_states[backend] = state
        histories[backend] = history

    difference = final_states["gym_pybullet"].position - final_states["simulation"].position
    distance = float(np.linalg.norm(difference))

    if verbose:
        print("See compare in graph")
        print(f"Final location difference (gym_pybullet - simulation), (x, y, z): {np.array2string(difference, precision = 3)} m")
        print(f"Final location distance: {distance:.3f} m")

    comparison = {"mode": mode, "simulation": histories["simulation"],
            "gym_pybullet": histories["gym_pybullet"], "final_states": final_states,
            "final_position_difference": difference, "final_position_distance": distance,
            "paired_training": paired.arrays()}

    from drone_sim.visualization.training_export import export_comparison
    comparison["dataset_path"] = export_comparison(comparison, starting_state, gains = gains, target_yaw = target_yaw)
    return comparison
