"""Per-episode PyBullet drone parameters.

The custom model always uses the nominal values in parameters.py. The PyBullet
reference can draw a different "true" drone each episode, so the learned
correction sees parameter error as well as missing physics and does not fit
one guessed set of values.
"""

from types import SimpleNamespace

import numpy as np

from drone_sim.configuration import parameters


def config_with(overrides = None, base = None):
    """Settings from base (default parameters.py) plus overrides, as a namespace."""

    base = parameters if base is None else base
    values = {name: getattr(base, name) for name in dir(base) if name.isupper()}
    values.update(overrides or {})
    return SimpleNamespace(**values)


def sample_pybullet_parameters(seed = None, ranges = None, base = None):
    """Draw one drone around base (default parameters.py). Returns (overrides, seed).

    Each range is a (low, high) multiplier on the nominal value. Arrays share
    one multiplier, except the moment of inertia, whose three axes vary
    independently while still forming a physically valid body.
    """

    base = parameters if base is None else base
    ranges = base.PYBULLET_PARAMETER_RANGES if ranges is None else ranges
    seed = int(np.random.SeedSequence().entropy % (2**63)) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    overrides = {}

    for name, (low, high) in ranges.items():
        if not 0 < low <= high:
            raise ValueError(f"Randomization range for {name} must satisfy 0 < low <= high.")

        nominal = np.asarray(getattr(base, name), dtype = float)

        if name == "MOMENT_OF_INERTIA":
            for _ in range(1000):
                value = nominal * rng.uniform(low, high, size = 3)
                if 2 * np.max(value) <= np.sum(value):
                    break
            else:
                raise ValueError("Could not draw a valid moment of inertia; narrow its range.")
        else:
            value = nominal * rng.uniform(low, high)

        overrides[name] = value if value.ndim else float(value)

    if "THRUST_COEFFICIENT" in overrides:
        # A motor's top speed is fixed by the motor, so maximum thrust scales
        # with the thrust coefficient. Both models then clip commands alike.
        factor = overrides["THRUST_COEFFICIENT"] / base.THRUST_COEFFICIENT
        overrides["MAX_THRUST_PER_MOTOR"] = float(base.MAX_THRUST_PER_MOTOR * factor)

    return overrides, seed


def flatten(overrides):
    """Names and values as flat lists, for example for model inputs."""

    names, values = [], []

    for name in sorted(overrides):
        value = np.asarray(overrides[name], dtype = float)

        if value.ndim:
            names.extend(f"{name}_{index}" for index in range(value.size))
            values.extend(value.ravel().tolist())
        else:
            names.append(name)
            values.append(float(value))

    return names, values
