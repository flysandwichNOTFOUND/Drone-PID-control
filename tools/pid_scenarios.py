from dataclasses import asdict, dataclass

import numpy as np

from drone_sim import main as mission
from drone_sim.configuration import parameters
from drone_sim.world.environment import BasicEnvironment
from drone_sim.world.terrain import FlatTerrain, HillTerrain, LandscapeTerrain


@dataclass(frozen = True)
class Scenario:
    name: str
    terrain: str
    seed: int
    destination: tuple
    yaw_degrees: float = 0.0
    wind_mode: str = "none"
    wind_velocity: tuple = (0.0, 0.0, 0.0)
    duration: float = 60.0


TRAINING_SCENARIOS = (
    Scenario("train_flat", "flat", 11, (2.0, 0.0)),
    Scenario("train_hill_wind", "hill", 12, (1.0, 1.5), 30, "constant", (0.4, -0.2, 0.0)),
    Scenario("train_landscape", "landscape", 13, (-1.5, 1.0), -30),
)

VALIDATION_SCENARIOS = (
    Scenario("holdout_flat_gust", "flat", 101, (-1.0, -2.0), 90, "gust", (-0.3, 0.4, 0.0)),
    Scenario("holdout_hill", "hill", 102, (2.0, -1.0), -60),
    Scenario("holdout_landscape_wind", "landscape", 103, (1.5, -1.5), 45, "constant", (-0.3, 0.2, 0.0)),
)


def terrain_for(scenario):

    if scenario.terrain == "landscape":
        return LandscapeTerrain(seed = scenario.seed)
    if scenario.terrain == "hill":
        rng = np.random.default_rng(scenario.seed)
        return HillTerrain(center = tuple(rng.uniform(-1, 1, 2)), height = float(rng.uniform(0.4, 0.9)), width = 2.0)
    return FlatTerrain()


def wind_for(scenario):

    return BasicEnvironment(scenario.wind_mode, scenario.wind_velocity, parameters.WIND_FORCE_COEFFICIENT,
                            gust_start = 3.0, gust_duration = 2.0, gust_period = 6.0)


def settling_time(time, within_tolerance):
    """First sample after the last violation; None means it never settled."""

    bad = np.flatnonzero(~np.asarray(within_tolerance, dtype = bool))
    first = int(bad[-1] + 1) if len(bad) else 0
    return float(time[first]) if first < len(time) else None


def control_metrics(motors, time):

    motors = np.asarray(motors, dtype = float)
    saturated = np.any((motors <= 1e-9) | (motors >= parameters.MAX_MOTOR_SPEED - 1e-9), axis = 1)
    slew = np.diff(motors / parameters.MAX_MOTOR_SPEED, axis = 0) / np.diff(time)[:, None]
    return float(np.mean(saturated)), float(np.sqrt(np.mean(slew**2))) if len(slew) else 0.0


def mission_metrics(state, data):

    position = np.asarray(data["position"])
    velocity = np.asarray(data["velocity_vector"])
    time = np.asarray(data["time"])
    targets = np.asarray(data["commanded_position_history"])
    goal = np.asarray(data["waypoints"][-1])
    error = np.linalg.norm(position - goal, axis = 1)
    settled = settling_time(time, (error <= parameters.ARRIVAL_TOLERANCE) & (np.linalg.norm(velocity, axis = 1) <= parameters.SPEED_TOLERANCE))
    active = np.asarray(data["physics_advanced"], dtype = bool)
    saturation, slew = control_metrics(np.asarray(data["motor_speed"])[active], time[active])
    rmse = float(np.sqrt(np.mean(np.sum((targets[active] - position[active])**2, axis = 1))))
    direction = goal[:2] - position[0, :2]
    norm = np.linalg.norm(direction)
    overshoot = max(0.0, float(np.max((position[:, :2] - goal[:2]) @ (direction / norm)))) if norm > 1e-9 else 0.0
    impact = float(data["touchdown_speed"] or 0.0)
    complete = bool(data["mission_complete"] and data["landed"] and not data["landing_failed"])
    duration = float(time[-1])
    score = (15 * rmse**2 + 120 * error[-1]**2 + 20 * overshoot**2
             + 0.1 * (settled if settled is not None else duration * 2)
             + 50 * saturation + 0.05 * slew**2 + 200 * max(impact - 0.15, 0)**2
             + (0 if complete else 10000))
    return {"score": float(score), "mission_complete": complete, "landed": bool(data["landed"]),
            "landing_failed": bool(data["landing_failed"]), "tracking_rmse_m": rmse,
            "final_error_m": float(error[-1]), "overshoot_m": overshoot, "settling_time_s": settled,
            "saturation_fraction": saturation, "motor_slew_rms_per_s": slew,
            "touchdown_speed_m_s": impact, "duration_s": duration, "stop_reason": data["stop_reason"]}


def evaluate_suite(gains, scenarios, backends):
    """Evaluate fixed cases; this function never searches or changes gains."""

    cases = []
    for scenario in scenarios:
        terrain = terrain_for(scenario)
        x, y = scenario.destination
        goal = [x, y, float(terrain.height(x, y)) + parameters.MINIMUM_CLEARANCE]
        for backend in backends:
            try:
                state, data = mission.run_simulation(
                    gains = gains, physics_backend = backend, duration = scenario.duration,
                    include_landing = True, verbose = False, terrain = terrain,
                    goal_position = goal, target_yaw = np.radians(scenario.yaw_degrees),
                    environment = wind_for(scenario))
                metrics = mission_metrics(state, data)
            except (ValueError, FloatingPointError) as error:
                metrics = {"score": 1e9, "mission_complete": False, "error": str(error)}
            cases.append({"scenario": asdict(scenario), "backend": backend, **metrics})
    scores = [case["score"] for case in cases]
    return {"score": float(np.mean(scores) + 0.25 * np.max(scores)),
            "success_rate": float(np.mean([case["mission_complete"] for case in cases])), "cases": cases}


def validate_candidate(baseline, candidate, backends):
    """Holdouts run once after training selection; no holdout-based candidate search."""

    before = evaluate_suite(baseline, VALIDATION_SCENARIOS, backends)
    changed = any(not np.array_equal(candidate[key], baseline[key]) for key in baseline)
    after = evaluate_suite(candidate, VALIDATION_SCENARIOS, backends) if changed else before
    accepted = (after["success_rate"] == 1.0 and after["score"] <= before["score"]
                and all(new["score"] <= old["score"] * 1.05 + 1e-9 for old, new in zip(before["cases"], after["cases"])))
    return {"candidate_accepted": bool(accepted), "baseline": before, "candidate": after,
            "scope": "Unseen seeds, routes, yaw and wind; validation is a gate, never a search objective."}
