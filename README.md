Drone Dynamics and Control Simulator

A Python simulation project for studying quadrotor dynamics, feedback control, and flight under wind disturbances.

Overview

The project develops two connected components:

1. Drone model and control system: model the drone's physical response and regulate altitude, orientation, and horizontal position using a cascaded position and orientation PID control, with a separate altitude PID loop.
2. Navigation simulation and environment: provide wind disturbances and support the development of point-to-point flight.
The current prototype uses PID control. Note that point to point flight component is still in prototyping stage.

Features
- Three dimensional position, velocity, orientation, and angular-state simulation.
- Motor thrust, gravity, and roll, pitch, and yaw torque calculations.
- PID control of altitude, orientation, and horizontal position.
- Tilt compensation, motor-speed limits, and commanded tilt limits.
- Configurable wind velocity with a linear relative-air-velocity force model.
- Plots of altitude, vertical velocity, motor speeds, orientation, and horizontal position.

Physics Model
Coordinates and Assumptions
- Position: $\mathbf{p}=[x,y,z]^T$, with world $z$ upward.
- Orientation: $\boldsymbol{\eta}=[\phi,\theta,\psi]^T$ for roll, pitch, and yaw.
- Thrust acts along the positive body $z$ axis.
- Use SI units; internal angles are in radians and motor speeds are in rad/s.
- The baseline uses independent angular acceleration for each axis and integrates angular-state rates directly into Euler angles.
  
## Core Equations

| Calculation | Formula |
| --- | --- |
| Motor thrust and limit | $T_i=\min(k_T\omega_i^2,T_{\max})$ |
| Level hover speed | $\omega_{\mathrm{hover}}=\sqrt{mg/(4k_T)}$ |
| Maximum motor speed | $\omega_{\max}=\sqrt{T_{\max}/k_T}$ |
| Body-to-world rotation | $R=R_z(\psi)R_y(\theta)R_x(\phi)$ |
| Wind force | $\mathbf{F}_{\mathrm{wind}}=c_w(\mathbf{v}_{\mathrm{wind}}-\mathbf{v})$ |
| Translational dynamics | $m\ddot{\mathbf{p}}=R[0,0,\sum_i T_i]^T+[0,0,-mg]^T+\mathbf{F}_{\mathrm{wind}}$ |
| Roll and pitch torque | $\tau_\phi=L(T_2-T_4),\quad \tau_\theta=L(T_3-T_1)$ |
| Yaw torque with thrust limiting | $\tau_\psi=(k_\tau/k_T)(T_1-T_2+T_3-T_4)$ |
| Simplified angular acceleration | $\alpha_j=\tau_j/I_j$ |
| Velocity-first integration | $\mathbf{v}_{k+1}=\mathbf{v}_k+\mathbf{a}_k\Delta t,\quad \mathbf{p}_{k+1}=\mathbf{p}_k+\mathbf{v}_{k+1}\Delta t$ |

Here, $m$ is mass, $g$ is gravitational acceleration magnitude,
$L$ is arm length, $I_j$ is axis inertia, $k_T$ is the thrust
coefficient, $k_\tau$ is the yaw torque coefficient, and $c_w$
is the wind-force coefficient.

The angular state follows the same velocity-first integration
pattern. The baseline rotational model treats each axis independently.

## Control Equations

Let $e_j=r_j-y_j$ denote target minus actual state, and let
$S_j=\sum e_j\Delta t$ denote accumulated error.

| Controller component | Formula |
| --- | --- |
| Altitude PID | $u_{z,k}=K_{P,z}e_{z,k}+K_{I,z}S_{z,k}+K_{D,z}(e_{z,k}-e_{z,k-1})/\Delta t$ |
| Altitude motor command | $\omega_b=\mathrm{clip}(\omega_{\mathrm{hover}}+u_z,0,\omega_{\max})$ |
| Horizontal position to pitch | $\theta_d=\mathrm{clip}(K_{P,x}e_x+K_{I,x}S_x-K_{D,x}v_x,-\theta_{\max},\theta_{\max})$ |
| Horizontal position to roll | $\phi_d=\mathrm{clip}(-(K_{P,y}e_y+K_{I,y}S_y-K_{D,y}v_y),-\phi_{\max},\phi_{\max})$ |
| Orientation correction | $c_j=\mathrm{clip}(K_{P,j}e_j+K_{I,j}S_j-K_{D,j}\dot{\eta}_j,-c_{\max},c_{\max})$ |
| Tilt compensation | $\omega_c=\mathrm{clip}(\omega_b/\sqrt{\max(\cos\phi\cos\theta,0.5)},0,\omega_{\max})$ |

The clipping function limits a value between its lower and upper bounds.

The altitude controller uses zero derivative correction on its first
step. Position and orientation controllers use measured velocity
or angular rate for derivative damping.

### Motor Mixing

The compensated base speed and orientation corrections are combined
into four motor commands:

$$
\begin{aligned}
\omega_1 &= \omega_c-c_\theta+c_\psi \\
\omega_2 &= \omega_c+c_\phi-c_\psi \\
\omega_3 &= \omega_c+c_\theta+c_\psi \\
\omega_4 &= \omega_c-c_\phi-c_\psi
\end{aligned}
$$

Each final motor command is clipped to $[0,\omega_{\max}]$.

Controller loops

Desired roll and pitchBase motor speedFour motor commandsUpdated state




1. Altitude loop — controls height
   It compares the target altitude with state.position[2]:
   \[
   e_z=z_{\text{target}}-z
   \]
   The PID correction adjusts a common base motor speed around the hover speed. A positive height error generally increases this command. Tilt compensation then increases the base speed to compensate for the reduced vertical thrust when the drone tilts.
   
2. Position loop — decides the required tilt
   It compares the target x/y coordinates with the current horizontal position:
   \[
   e_x=x_{\text{target}}-x,\qquad e_y=y_{\text{target}}-y
   \]
   It converts these errors into desired pitch and roll. Tilting redirects some thrust horizontally, allowing the drone to move toward its target. Velocity feedback damps that movement as the drone approaches the target, and tilt limits constrain the commands.
   
4. Orientation loop — achieves the requested tilt and heading
   It compares the desired roll, pitch, and yaw with the actual angles. Angle-error feedback produces corrections, while angular-rate feedback damps rotation.
   These corrections are mixed with the altitude loop’s base motor speed to produce four motor commands. For example, a positive pitch correction increases motor 3’s speed and decreases motor 1’s speed, creating pitch torque.
After applying those commands, Drone.update_state() calculates the drone’s motion under thrust, gravity, and wind. That updated state feeds the next control cycle, completing the feedback loops.
Across the controllers, \(K_P\) responds to current error, \(K_I\) accumulates persistent error, and \(K_D\) provides damping. Your altitude loop uses the change in error for its derivative term; the position and orientation loops use measured velocities.

With DT = 0.01, all three controllers update 100 times per second of simulated time.

Project Code Structure

File Role

main.py	Simulation entry point; parameters, controller calls, state updates, data recording, and plotting.
testing.py	Core Drone dynamics and DroneState classes; this is part of the simulator despite its filename.
controllerV1.py	Altitude, orientation, and horizontal-position PID controllers.
environment.py	Wind configuration and external-force calculation.
controllerV2.py	Experimental backstepping controller core; currently separate from the simulation execution path.
auto_tune_pid.py	Optional development utility for tuning PID gains through separate simulation trials.
README.md	Project overview and documentation.


PID Tuning Utility

auto_tune_PID.py is only a development tool for finding recommended $K_P$, $K_I$, and $K_D$ values. It is not part of the simulator's normal execution path and does not control the drone during a normal simulation run.
Selected gains are configured in the simulator before running main.py. The available script is named auto_tune_pid.py;

Requirements
- Anaconda Distribution, which provides Conda for environment management.
- Python 3.10.21 
- NumPy and Matplotlib.
- 
Testing conditions
- Controller: V1 PID controller, with cascaded position and orientation loops and a separate altitude loop.
- Simulation duration: approximately 40 seconds.
- Target position: x = 1.5 m, y = −0.5 m, z = 2.0 m.
- Target yaw: approximately 2.5°.
- Initial position: approximately [0, 0, 0] m, as shown in the plots.
- Wind cases: no wind, constant wind, and gusting wind.
- Reported wind setting: [0.5, 0.5, 0.5] m/s, interpreting your decimal commas.
- Maximum motor speed: 1,000 rad/s.
- The screenshots do not specify the PID gains, time step, or gust waveform. Numerical observations below are approximate.
  
Validation and Results

No wind: Altitude and horizontal position approach their targets after small initial overshoots. Roll and pitch return near zero, yaw approaches 2.5°, and motor speeds settle near hover speed.

Constant wind: Altitude and x position show larger initial overshoots. Horizontal position recovers near the target, while a small altitude offset remains. Sustained roll and pitch adjustments oppose the wind.

Gusting wind: Repeated disturbances cause persistent position and altitude fluctuations. Orientation follows changing commands, and the response remains bounded over the displayed run.

Future Development
-reduce horizontal position osculation due gust wind
-develop a fully functional point-to-point navigation system
