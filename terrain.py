"""Shared terrain elevations for route planning, contact, and rendering."""
import numpy as np
from validation import positive, vector


class FlatTerrain:
    def __init__(self, base_height = 0.0):

        self.base_height = float(base_height)
        if not np.isfinite(self.base_height):
            raise ValueError("Terrain base height must be finite.")

    def height(self, x, y):

        x, y = np.broadcast_arrays(x, y)
        return np.full(x.shape, self.base_height)

    def gradient(self, x, y):

        return np.zeros(2)

    def normal(self, x, y):

        gradient = self.gradient(x, y)
        normal = np.r_[-gradient, 1.0]
        return normal / np.linalg.norm(normal)

    def maximum_height_along(self, start, goal):

        return self.base_height


class HillTerrain(FlatTerrain):
    def __init__(self, base_height = 0.0, center = (5.0, 4.0), height = 3.0, width = 1.6):

        super().__init__(base_height)
        self.center = vector(center, 2, "Hill center")
        self.hill_height = positive(height, "Hill height", allow_zero = True)
        self.width = positive(width, "Hill width")

    def height(self, x, y):

        return self.base_height + self.hill_height * np.exp(
            -((np.asarray(x) - self.center[0])**2 + (np.asarray(y) - self.center[1])**2)
 / (2 * self.width**2))

    def gradient(self, x, y):

        return -(float(self.height(x, y)) - self.base_height) * (
            np.array([x, y]) - self.center) / self.width**2

    def maximum_height_along(self, start, goal):

        start, goal = np.asarray(start), np.asarray(goal)
        direction = goal - start
        length_squared = np.dot(direction, direction)
        fraction = (np.clip(np.dot(self.center - start, direction) / length_squared, 0, 1)
                    if length_squared else 0.0)
        closest = start + fraction * direction
        return float(self.height(*closest))


class SlopedTerrain(FlatTerrain):
    def __init__(self, base_height = 0.0, gradient = (0.1, 0.0)):

        super().__init__(base_height)
        self.slope = vector(gradient, 2, "Terrain gradient")

    def height(self, x, y):

        return self.base_height + self.slope[0] * np.asarray(x) + self.slope[1] * np.asarray(y)

    def gradient(self, x, y):

        return self.slope.copy()

    def maximum_height_along(self, start, goal):

        return max(float(self.height(*start)), float(self.height(*goal)))


class LandscapeTerrain(FlatTerrain):
    """Deterministic rolling landscape with ridges, valleys, and small undulations.

    This is a synthetic height field, not surveyed elevation data. The same
    continuous surface and its derivatives are used by flight and rendering.
    """
    plot_bounds = (-10.0, 18.0, -10.0, 18.0)

    def __init__(self, base_height = 0.0, seed = 17):

        super().__init__(base_height)
        # amplitude, center x/y, major/minor widths, rotation in radians
        self.features = [
            (4.2, 8.0, 7.5, 5.5, 2.1, 0.65),
            (2.7, -4.5, 8.0, 3.8, 4.5, -0.3),
            (2.3, 11.0, -4.0, 4.8, 2.7, -0.6),
            (1.6, -6.0, -5.5, 3.2, 4.0, 0.4),
            (1.8, 15.0, 14.0, 3.5, 3.0, 0.2),
            (-1.2, 2.5, 5.0, 2.5, 5.5, 0.35),
            (-0.7, 5.0, -5.0, 2.2, 4.0, -0.2),
        ]
        rng = np.random.default_rng(seed)
        self.waves = []
        for wavelength, amplitude in [(9.0, 0.12), (4.5, 0.055), (2.2, 0.018)]:
            for _ in range(3):
                angle = rng.uniform(0, 2 * np.pi)
                wave_vector = (2 * np.pi / wavelength) * np.array([np.cos(angle), np.sin(angle)])
                self.waves.append((amplitude / 3, wave_vector, rng.uniform(0, 2 * np.pi)))
        self.gradient_bound = sum(abs(a) * np.exp(-0.5) / min(sx, sy)
                                  for a, _, _, sx, sy, _ in self.features)
        self.gradient_bound += sum(abs(a) * np.linalg.norm(k) for a, k, _ in self.waves)

    def height(self, x, y):

        x, y = np.broadcast_arrays(np.asarray(x, dtype = float), np.asarray(y, dtype = float))
        result = np.full(x.shape, self.base_height)
        for amplitude, cx, cy, sx, sy, angle in self.features:
            u = np.cos(angle) * (x - cx) + np.sin(angle) * (y - cy)
            v = -np.sin(angle) * (x - cx) + np.cos(angle) * (y - cy)
            result += amplitude * np.exp(-0.5 * ((u / sx)**2 + (v / sy)**2))
        for amplitude, k, phase in self.waves:
            result += amplitude * np.sin(k[0] * x + k[1] * y + phase)
        return result

    def gradient(self, x, y):

        result = np.zeros(2)
        for amplitude, cx, cy, sx, sy, angle in self.features:
            c, s = np.cos(angle), np.sin(angle)
            u, v = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
            height = amplitude * np.exp(-0.5 * ((u / sx)**2 + (v / sy)**2))
            result -= height * np.array([c * u / sx**2 - s * v / sy**2,
                                         s * u / sx**2 + c * v / sy**2])
        for amplitude, k, phase in self.waves:
            result += amplitude * np.cos(k[0] * x + k[1] * y + phase) * k
        return result

    def maximum_height_along(self, start, goal):

        start, goal = np.asarray(start), np.asarray(goal)
        length = np.linalg.norm(goal - start)
        segments = max(1, int(np.ceil(length / 0.15)))
        points = start + np.linspace(0, 1, segments + 1)[:, None] * (goal - start)
        # A derivative bound covers peaks between samples conservatively.
        return float(np.max(self.height(points[:, 0], points[:, 1]))) + self.gradient_bound * length / (2 * segments)
