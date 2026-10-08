"""Input checks shared by the simulator and controllers."""
import numpy as np


def vector(value, length, name):

    value = np.asarray(value, dtype = float)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite {length}-element vector.")
    return value.copy()


def positive(value, name, allow_zero = False):

    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return value


def gains(value, name):

    value = vector(value, 3, name)
    if np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return value
