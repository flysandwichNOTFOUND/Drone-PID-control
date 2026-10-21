# Drone Dynamics and Control Simulator

Python quadrotor simulation with PID control, terrain-aware A* routing, smooth path tracking, wind disturbances, and automatic landing.

## Physics introduction

Each motor produces thrust proportional to its squared speed (`T = kω²`), capped at the motor's maximum thrust. Total thrust acts along body Z and is rotated into world coordinates; gravity and wind contribute to the net force (`a = F / m`). World Z points upward.

Differences in motor thrust produce roll and pitch torque, while rotor reaction torques control yaw. Angular acceleration includes inertia and gyroscopic effects; orientation is integrated using quaternions. PID controllers adjust motor speeds to track position, altitude, and heading.

## Run

Use Python 3.10+ from the project folder:

```bash
python -m pip install -r requirements.txt
python -m drone_sim.main
```

The simulation prints its final state and saves graphs to `outputs/` without opening plot windows.

## Example

Set a destination and flight conditions in `drone_sim/main.py`, then run the simulation:

```python
TARGET_X = 10          # metres
TARGET_Y = 6
TARGET_HEIGHT = 2.0
TARGET_YAW = 0         # degrees
ROUTE_PLANNING_MODE = "astar"
TERRAIN_MODE = "hill"
WIND_MODE = "none"
LANDING_MODE = "safe_nearby"
```

The planner adjusts altitude for terrain clearance, and `safe_nearby` may choose a landing site near the requested destination. This graph shows the last saved run:

![Drone trajectory over terrain](outputs/drone_3d_trajectory.png)

## Essential settings

- `drone_sim/main.py`: mission targets, routing (`astar` / `straight`), terrain (`flat` / `hill` / `slope` / `landscape`), wind (`none` / `constant` / `gust`), and landing mode (`strict` / `safe_nearby` / `exact`).
- `drone_sim/configuration/parameters.py`: PID gains, drone properties, terrain and wind settings, clearance, flight limits, landing settings, timestep, and duration. Increase `SIMULATION_TIME` for longer routes.
- `drone_sim/visualization/plotting.py`: graph appearance.

Terrain and ground contact are simplified: landing gear, tipping, and surface friction are not modeled. Reference speed, acceleration, and jerk limits do not guarantee the same limits on actual motion.

Run regression tests:

```bash
python -m unittest discover -s tests -v
```

## Project layout

`PHYSICS_BACKEND` in `drone_sim/main.py` selects the physics backend. Leave it
as `"simulation"` to run the existing simulator. `"gym_pybullet"` currently raises
a clear `NotImplementedError`: `drone_sim/physics/gym_backend.py` contains annotated
adapter method scaffolds, and `drone_sim/physics/comparison.py` contains the planned
comparison entry point. The Gym integration and matching drone model are not
implemented yet; no new dependencies are required for the simulation backend.

- `drone_sim/`: simulator code and mission settings.
- `tools/`: PID tuning utilities (`python -m tools.auto_tune_pid`).
- `tests/`: regression tests.
- `outputs/`: generated graphs and tuning reports.

### Simulator categories

```text
drone_sim/
  main.py                 # Entry point, mission settings, backend switch
  validation.py           # Shared input checks
  planning/               # Route planning and A* search
  visualization/          # Graphs and trajectory plots
  control/                # PID, path following, landing; empty tuning placeholder
  world/                  # Terrain and wind
  configuration/          # Parameters and project/output paths
  physics/                # dynamics.py, attitude, Gym adapter and comparison scaffolds
```

The implemented PID tuning tools remain in the top-level `tools/` folder.
Run commands are unchanged: `python -m drone_sim.main` and
`python -m unittest discover -s tests -v` from the project root.
