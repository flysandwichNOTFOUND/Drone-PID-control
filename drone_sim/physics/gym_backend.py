"""Planned gym-pybullet-drones adapter; physics integration is not implemented.

Reuse physics.dynamics.DroneState for both backends. No third-party imports are needed
until the adapter is implemented. Match our 1 kg model before enabling flights;
the library's default aircraft has different physical properties.
"""

from drone_sim.physics.dynamics import DroneState


class GymPyBulletDrone:
    """Alternative to physics.dynamics.Drone, exposing the same stepping interface."""

    def __init__(self, config, terrain, gui = False):
        """Input: parameter module, terrain object, GUI flag. Output: backend.

        Source: main.create_drone() supplies parameters.py and create_terrain().
        Caller: main.create_drone(); future comparison runner.
        Planned: configure CtrlAviary with matching model, timestep and terrain.
        """
        raise NotImplementedError(
            "gym_pybullet is scaffolded only. Keep PHYSICS_BACKEND = 'simulation' "
            "until the matching drone model and physics adapter are implemented.")

    def reset(self, state: DroneState):
        """Input: initial DroneState. Output: None; initialize the simulator.

        Source: mission/comparison initial position, orientation and velocities.
        Caller: future mission setup and comparison runner, before first step.
        """
        raise NotImplementedError("Initialize the PyBullet body from DroneState.")

    def get_ground_height(self, x, y):
        """Input: world x/y in metres. Output: terrain elevation in metres.

        Source: terrain passed to __init__, also used for collision geometry.
        Caller: main.run_simulation() during initialization and logging.
        """
        raise NotImplementedError("Query the shared terrain height.")

    def update_state(self, state: DroneState, dt, external_force = None):
        """Input: mutable state, seconds, optional world force in newtons.

        Output: None; update the supplied DroneState in place after stepping.
        Source: state.motor_speeds (rad/s) from PID; wind force from environment.
        Caller: main.control_step(); future same-command comparison loop.
        Planned: map motor order, convert rad/s to RPM, apply wind on each
        physics substep, advance CtrlAviary, then call read_state().
        """
        raise NotImplementedError("Translate commands and advance CtrlAviary.")

    def read_state(self, state: DroneState):
        """Input: state to fill. Output: None; replace its measured fields.

        Source: CtrlAviary observation and PyBullet contact information.
        Caller: update_state() and reset() inside this adapter.
        Planned: position/velocity in world frame, Euler angles in radians,
        angular velocity in body frame, motor speeds in rad/s, contact fields.
        """
        raise NotImplementedError("Map observations and contacts to DroneState.")

    def close(self):
        """Input: none. Output: None; release the physics connection and GUI.

        Source: CtrlAviary instance owned by this backend.
        Caller: future mission/comparison runner in a finally block.
        """
        raise NotImplementedError("Close the owned CtrlAviary environment.")
