"""Body-rate kinematics for the ZYX roll/pitch/yaw convention."""
import numpy as np


def wrap_angle(angle):

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def euler_rates(orientation, body_rates):

    roll, pitch, _ = orientation
    p, q, r = body_rates
    cosine = np.cos(pitch)
    if abs(cosine) < 1e-8:
        raise ValueError("Euler angle rates are undefined at vertical pitch.")
    coupled = q * np.sin(roll) + r * np.cos(roll)
    return np.array([p + np.tan(pitch) * coupled,
                     q * np.cos(roll) - r * np.sin(roll),
                     coupled / cosine])


def integrate_orientation(orientation, body_rates, dt):
    """Apply a body-frame rotation using a normalized quaternion.

    Euler angles remain the public state representation. Converting each step
    also allows callers to set the initial orientation directly.
    """

    roll, pitch, yaw = np.asarray(orientation) / 2.0
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    w = cr * cp * cy + sr * sp * sy
    v = np.array([sr * cp * cy - cr * sp * sy,
                  cr * sp * cy + sr * cp * sy,
                  cr * cp * sy - sr * sp * cy])
    speed = np.linalg.norm(body_rates)
    half_angle = speed * dt / 2.0
    dw = np.cos(half_angle)
    dv = np.asarray(body_rates) * (dt / 2.0) * np.sinc(half_angle / np.pi)
    quaternion = np.r_[w * dw - np.dot(v, dv),
                       w * dv + dw * v + np.cross(v, dv)]
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])
