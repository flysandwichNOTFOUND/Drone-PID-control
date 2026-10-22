"""Drone physics, backend comparison and ML exports.

dynamics.py: Drone integrates forces/contact; DroneState stores flight state.
attitude.py: angle wrapping, Euler rates and quaternion integration.
gym_banckend_dynamics.py: GymPyBulletDrone setup, reset, step, read and cleanup.
comparison.py: run_comparison() runs both backends from fresh initial states.
paired_training.py: PairedAccelerationCollector creates matched-state acceleration X/y pairs.
"""
