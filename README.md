# Drone Dynamics and Control Simulator

Python quadrotor simulation with PID control, terrain-aware A* routing, smooth path tracking, wind disturbances, and automatic landing.

## Setup and run

Create and activate a Conda environment:

```bash
conda create --name drone-sim python=3.10 numpy matplotlib
conda activate drone-sim
```

Alternatively, install dependencies in an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

In VS Code, select that environment with **Python: Select Interpreter**. Run from the project folder:

```bash
python main.py
```

The simulation prints its final state and saves graphs without opening plot windows:

- `outputs/altitude_control_results.png`: flight stages, position, velocity, orientation, motor speeds, and waypoint tracking.
- `outputs/drone_3d_trajectory.png`: planned route and actual flight over terrain.

## Configuration

Edit mission targets and modes in `main.py`:

| Setting | Purpose or options |
| --- | --- |
| `TARGET_X`, `TARGET_Y`, `TARGET_HEIGHT` | Destination coordinates in metres; route altitude is adjusted for terrain clearance. |
| `TARGET_YAW` | Heading in degrees. |
| `ROUTE_PLANNING_MODE` | `"astar"` or `"straight"`. |
| `TERRAIN_MODE` | `"flat"`, `"hill"`, `"slope"`, or `"landscape"`. |
| `WIND_MODE` | `"none"`, `"constant"`, or `"gust"`. |

Edit **all PID gains and shared simulation settings** in `parameters.py`: drone properties, controller limits, terrain, wind, A* grid settings, clearance, flight speed/acceleration/jerk, landing, timestep, and duration. Increase `SIMULATION_TIME` if the route needs more time to finish. Graph appearance is configured in `plotting.py`.

## How it works

1. `routeplanner.py` builds takeoff, cruise, and optional landing waypoints. A* searches a terrain-clearance grid; straight mode climbs above the terrain along the direct crossing.
2. `navigation.py` follows smooth position and velocity references with speed, acceleration, and jerk limits. Collinear waypoints remain continuous; corners finish at rest.
3. `controllerV1.py` converts altitude and horizontal-position errors into motor commands through attitude control, tilt compensation, and command limits.
4. `testing.py` advances drone physics; `attitude.py` integrates orientation using quaternions. World Z points upward, thrust follows body Z, and internal angles use radians.
5. `landing.py` checks the landing-site slope, aligns the drone, descends, slows near the ground, and disarms after a successful touchdown.

Terrain is synthetic. Ground contact does not model landing gear, tipping, or surface friction. Flight-reference limits do not guarantee identical limits on actual drone motion, and look-ahead distance is a slowing threshold rather than a hard bound.

## Files and folders

| Location | Purpose |
| --- | --- |
| `main.py`, `parameters.py` | Simulation entry point and configuration. |
| `testing.py`, `attitude.py`, `controllerV1.py` | Drone dynamics, orientation, and PID control. |
| `routeplanner.py`, `pathfinding.py`, `navigation.py`, `landing.py` | Route generation, A*, path following, and landing. |
| `terrain.py`, `environment.py` | Ground elevation and wind forces. |
| `plotting.py`, `project_paths.py`, `validation.py` | Graphs, output paths, and input checks. |
| `tests/` | Regression tests. |
| `tools/` | Optional PID tuning utilities. |
| `outputs/` | Generated graphs and tuning reports. |

## Tests and tuning

Run regression tests:

```bash
python -m unittest discover -s tests -v
```

Optional tuning commands:

```bash
python -m tools.auto_tune_pid
python -m tools.retune_pid
python -m tools.retune_pid --expanded
```

`auto_tune_pid` saves recommendations for manual application. `retune_pid` validates candidates and writes selected gains directly to `parameters.py`; `--expanded` searches wider ranges. Reports and plots go to `outputs/`. Selected gains are the best tested candidates, not a proven global optimum.
