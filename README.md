# Drone Dynamics and Control Simulator

A Python quadrotor simulator for studying flight dynamics, PID control, and wind disturbances.

## Overview

The simulator models a drone’s motion and uses feedback control to track a target position and heading. It combines altitude control with cascaded horizontal-position and orientation control, then mixes their outputs into four motor commands.

The current simulation uses `controllerV1.py`. An experimental backstepping controller in `controllerV2.py` is available for future integration.

## Features

- Three-dimensional position, velocity, orientation, and angular-state simulation.
- Motor thrust, gravity, and roll, pitch, and yaw torque calculations.
- PID control of altitude, horizontal position, and orientation.
- Tilt compensation, motor-speed limits, and commanded tilt limits.
- Wind forces based on relative air velocity.
- Plots of altitude, vertical velocity, motor speeds, orientation, and horizontal position.

## Physics Model

### Coordinates and Assumptions

- Position: $\mathbf{p} = [x,y,z]^T$, with world $z$ pointing upward.
- Orientation: $\boldsymbol{\eta} = [\phi,\theta,\psi]^T$ for roll, pitch, and yaw.
- Thrust acts along the positive body $z$ axis.
- All quantities use SI units; internal angles are in radians and motor speeds are in rad/s.
- The rotational model includes gyroscopic coupling between axes. Angular velocities are integrated directly into Euler angles as a simplifying approximation.

### Core Equations

| Calculation | Formula |
| --- | --- |
| Motor thrust and limit | $T_i = \min(k_T\omega_i^2, T_{\max})$ |
| Level hover speed | $\omega_{\mathrm{hover}} = \sqrt{mg/(4k_T)}$ |
| Maximum motor speed | $\omega_{\max} = \sqrt{T_{\max}/k_T}$ |
| Body-to-world rotation | $R = R_z(\psi) R_y(\theta) R_x(\phi)$ |
| Wind force | $`\mathbf{F}_{\mathrm{wind}} = c_w(\mathbf{v}_{\mathrm{wind}} - \mathbf{v})`$ |
| Translational dynamics | $m\ddot{\mathbf{p}} = R[0,0,\sum_i T_i]^T + [0,0,-mg]^T + \mathbf{F}_{\mathrm{wind}}$ |
| Roll and pitch torque | $\tau_\phi = L(T_2 - T_4),\quad \tau_\theta = L(T_3 - T_1)$ |
| Yaw torque with thrust limiting | $\tau_\psi = (k_\tau/k_T)(T_1 - T_2 + T_3 - T_4)$ |
| Angular acceleration with gyroscopic coupling | $`\dot{\boldsymbol{\omega}} = I^{-1}\left(\boldsymbol{\tau} - \boldsymbol{\omega}\times(I\boldsymbol{\omega})\right)`$ |
| Velocity-first integration | $`\mathbf{v}_{k+1} = \mathbf{v}_{k} + \mathbf{a}_{k}\Delta t,\quad \mathbf{p}_{k+1} = \mathbf{p}_{k} + \mathbf{v}_{k+1}\Delta t`$ |

Here, $m$, $g$, $L$, and $I_j$ represent mass, gravitational acceleration magnitude, arm length, and axis inertia. The coefficients $k_T$, $k_\tau$, and $c_w$ describe motor thrust, yaw torque, and wind force.

For rotational dynamics, $I = \mathrm{diag}(I_\phi,I_\theta,I_\psi)$ is the inertia matrix, $\boldsymbol{\omega}$ is angular velocity, and $\boldsymbol{\tau}$ is net torque. The cross-product term accounts for gyroscopic coupling. Angular motion uses the same velocity-first integration pattern.

## Control System

### Controller Loops

The horizontal-position and orientation controllers form a cascade:

- **Position loop:** converts x and y position errors into desired pitch and roll angles.
- **Orientation loop:** tracks the desired roll, pitch, and yaw using motor-speed corrections.
- **Altitude loop:** independently adjusts the common motor speed to track the height target.

Tilt compensation increases the base motor speed during tilted flight. Motor mixing combines this base speed with the orientation corrections. All controllers update at each simulation step.

### Control Equations

For each controlled state, $e_j = r_j - y_j$ is the tracking error and $S_j = \sum e_j\Delta t$ is the accumulated error.

| Controller component | Formula |
| --- | --- |
| Altitude PID | $u_{z,k} = K_{P,z}e_{z,k} + K_{I,z}S_{z,k} + K_{D,z}(e_{z,k} - e_{z,k-1})/\Delta t$ |
| Altitude motor command | $\omega_b = \mathrm{clip}(\omega_{\mathrm{hover}} + u_z, 0, \omega_{\max})$ |
| Horizontal position to pitch | $\theta_d = \mathrm{clip}(K_{P,x}e_x + K_{I,x}S_x - K_{D,x}v_x, -\theta_{\max}, \theta_{\max})$ |
| Horizontal position to roll | $\phi_d = \mathrm{clip}(-(K_{P,y}e_y + K_{I,y}S_y - K_{D,y}v_y), -\phi_{\max}, \phi_{\max})$ |
| Orientation correction | $`c_{j} = \mathrm{clip}(K_{P,j}e_{j} + K_{I,j}S_{j} - K_{D,j}\dot{\eta}_{j}, -c_{\max}, c_{\max})`$ |
| Tilt compensation | $\omega_c = \mathrm{clip}(\omega_b/\sqrt{\max(\cos\phi\cos\theta, 0.5)}, 0, \omega_{\max})$ |

Clipping limits a value between the specified bounds. The altitude loop starts with a zero derivative term; the position and orientation loops use measured velocity and angular rate for derivative damping.

### Motor Mixing

The compensated base speed and orientation corrections produce four motor commands:

$$
\begin{aligned}
  \omega_1 &= \omega_c - c_\theta + c_\psi \\
  \omega_2 &= \omega_c + c_\phi - c_\psi \\
  \omega_3 &= \omega_c + c_\theta + c_\psi \\
  \omega_4 &= \omega_c - c_\phi - c_\psi
\end{aligned}
$$

Each command is then clipped to $[0,\omega_{\max}]$.

## Project Structure

| File | Purpose |
| --- | --- |
| `main.py` | Simulation entry point, parameters, controller updates, data recording, and plotting. |
| `testing.py` | Core `Drone` dynamics and `DroneState` classes used by the simulator. |
| `controllerV1.py` | Altitude, horizontal-position, and orientation PID controllers. |
| `environment.py` | Wind configuration and external-force calculations. |
| `controllerV2.py` | Experimental backstepping controller awaiting integration. |
| `auto_tune_pid.py` | Optional utility for tuning PID gains through separate simulation trials. |
| `README.md` | Project documentation. |
| `results/` | Plots used in the results section. |

**`auto_tune_PID.py` is only a PID tuning tool. It is not part of the normal simulation execution path.** The supplied script is named `auto_tune_pid.py`; use the capitalization present in your repository. Recommended gains must be applied manually in `main.py`.

## Installation

### Requirements

- [Anaconda Distribution](https://www.anaconda.com/download).
- Python **3.10.21**.
- NumPy and Matplotlib.

NumPy and Matplotlib are the only external Python dependencies. The tuning utility also uses `json` and `pathlib`, which are included with Python.

### Create the Environment

Open **Anaconda Prompt** on Windows, or a terminal with Conda available on macOS or Linux:

```bash
conda create --name drone-sim python=3.10.21 numpy matplotlib
conda activate drone-sim
```

If the environment already exists:

```bash
conda activate drone-sim
conda install python=3.10.21 numpy matplotlib
```

### Verify the Setup

Check the active environment, Python version, interpreter path, and dependencies:

```bash
conda info --envs
python --version
python -c "import sys; print(sys.executable)"
python -c "import numpy, matplotlib; print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__)"
```

The active environment should be `drone-sim`, the Python version should be `3.10.21`, and the interpreter path should point inside that environment.

From the project directory, verify the local imports:

```bash
python -c "from testing import Drone, DroneState; from controllerV1 import HeightPIDController, OrientationPIDController, PositionPIDController; from environment import BasicEnvironment; print('Project imports OK')"
```

Keep `main.py`, `testing.py`, `controllerV1.py`, and `environment.py` in the same directory with these exact filenames. A successful check prints `Project imports OK`.

### VS Code Setup

1. Open the project folder with the Python extension installed.
2. Open the Command Palette and choose **Python: Select Interpreter**.
3. Select the `drone-sim` interpreter running Python **3.10.21**.
4. Open a new terminal and activate the environment if needed.

## Usage

### Run the Simulation

With `drone-sim` active, run this command from the project directory:

```bash
python main.py
```

The simulation prints the final drone state and opens plots of altitude, vertical velocity, motor speeds, orientation, and horizontal position.

### Configure a Run

Edit the settings near the top of `main.py`:

| Setting | Parameters |
| --- | --- |
| Drone properties | `TOTAL_MASS`, `ARM_LENGTH`, `MOMENT_OF_INERTIA`, `MAX_THRUST_PER_MOTOR`, `THRUST_COEFFICIENT`, `TORQUE_COEFFICIENT` |
| Target position and heading | `TARGET_X`, `TARGET_Y`, `TARGET_HEIGHT`, `TARGET_YAW` |
| Altitude gains | `KP`, `KI`, `KD` |
| Orientation gains | `ROLL_GAINS`, `PITCH_GAINS`, `YAW_GAINS` |
| Position gains | `X_POSITION_GAINS`, `Y_POSITION_GAINS` |
| Controller limits | `MAX_TILT_DEGREES`, `MAX_ORIENTATION_CORRECTION` |
| Wind | `WIND_VELOCITY`, `WIND_FORCE_COEFFICIENT` |
| Simulation timing | `DT`, `SIMULATION_TIME` |

### Tune PID Gains (Optional)

Run the tuning tool separately:

```bash
python auto_tune_pid.py
```

Use `auto_tune_PID.py` if that is your repository’s filename. The tool writes `optimized_pid_gains.json` and `auto_tune_results.png`. Copy the selected gains into `main.py`, then rerun the simulation.

## Results

### Test Conditions

The plots compare no wind, constant wind, and gusting wind over approximately **40 seconds**.

| Setting | Value |
| --- | --- |
| Target position `[x, y, z]` | `[1.5, -0.5, 2.0]` m |
| Target yaw | Approximately `2.5°` |
| No-wind velocity | `[0.0, 0.0, 0.0]` m/s |
| Reported wind setting | `[0.5, 0.5, 0.5]` m/s |
| Gust start time | `10` s |
| Gust duration | `2` s |
| Gust period | `6` s |
| Plotted motor-speed limit | `1000` rad/s |

The wind setting uses decimal values of `0.5` on each axis. Gusts switch the configured wind velocity on for 2 seconds every 6 seconds, starting at 10 seconds. Wind velocity is zero before the first gust and between gusts.

### No Wind

Altitude and horizontal position approach their targets after small initial overshoots. Roll and pitch return near zero, yaw approaches 2.5°, and motor speeds settle near hover speed.

*Figure 1. Position, orientation, and motor response without wind.*

### Constant Wind

Altitude and x position show larger initial overshoots. Horizontal position recovers near the target, while a small altitude offset remains. Sustained roll and pitch adjustments oppose the wind.

*Figure 2. Response under constant wind of `[0.5, 0.5, 0.5]` m/s.*

### Gusting Wind

Repeated disturbances cause persistent position and altitude fluctuations. Orientation follows changing commands, and the response remains bounded over the displayed run.

*Figure 3. Response under gusting wind.*

No motor-speed saturation is visible in any of the three plots. These observations are based on plot inspection; exact tracking errors and settling times require recorded simulation data. Reproducible comparisons should also record the gains, time step, initial state, and gust waveform.

## Future Development

- Integrate the experimental backstepping controller.
- Extend the simulation to point-to-point navigation.
- Evaluate gust tracking using recorded error metrics.
