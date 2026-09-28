# Drone PID Simulation

This project is a simple Python drone simulation that combines drone dynamics, wind effects, and cascaded PID control.

## Current features

- Altitude PID control
- Roll, pitch, and yaw PID control
- Horizontal X/Y position control
- Basic wind-force model
- Motor-speed limits
- Matplotlib plots for altitude, vertical velocity, motor speeds, orientation, and horizontal position

## Project files

- `main.py` — creates the drone and controllers, runs the simulation, and plots the results
- `testing.py` — contains the drone state and dynamics model
- `controllerV1.py` — contains the altitude, orientation, and position PID controllers
- `environment.py` — contains the wind model
- `auto_tune_pid.py` — tests and recommends PID gains

## Run the simulation

Place all Python files in the same folder, activate the project environment, and run:

```bash
python main.py
```

Required packages:

```bash
pip install numpy matplotlib
```

## Control structure

The position controller converts X/Y position error into target pitch and roll angles. The orientation controller converts those target angles into individual motor-speed corrections. The altitude controller supplies the base motor speed.

```text
Target position -> Position PID -> Target roll/pitch
Target roll/pitch -> Orientation PID -> Motor corrections
Target altitude -> Height PID -> Base motor speed
```

The orientation PID uses measured angular velocity for derivative damping:

```python
derivative_error = -state.angular_velocity
```

This keeps derivative damping active while the position controller continuously changes the target orientation.

## Current testing setup

Test one horizontal axis at a time before combining X and Y:

```python
TARGET_X = 1.0
TARGET_Y = 0.0

X_POSITION_GAINS = np.array([0.01, 0.0, 0.08])
Y_POSITION_GAINS = np.array([0.01, 0.0, 0.08])

MAX_TILT_DEGREES = 5.0
SIMULATION_TIME = 12.0
```

Allow the drone to finish taking off before enabling horizontal position control. After the X-axis response is stable, test Y by setting `TARGET_X = 0.0` and `TARGET_Y = 1.0`. Then test both axes together.

## Next steps

1. Tune the X-axis position PID.
2. Tune the Y-axis position PID.
3. Test combined X/Y position holding.
4. Compare no wind, constant wind, and gust conditions.
5. Record position error, settling time, and motor saturation.
