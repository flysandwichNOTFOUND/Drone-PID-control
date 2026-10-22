"""Standalone PID tuning utilities; run from the project root.

auto_tune_pid.py: staged gain search, scoring, validation and result export.
pid_scenarios.py: training/holdout cases, performance metrics and acceptance checks.
retune_pid.py: broader search, mission checks and applying gains to parameters.py.
Run: python -m tools.auto_tune_pid or python -m tools.retune_pid.
"""
