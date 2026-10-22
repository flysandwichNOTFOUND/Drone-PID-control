from functools import lru_cache
import numpy as np
from drone_sim.configuration.parameters import TERRAIN_BOUNDS
from drone_sim.validation import positive, vector


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

        return self.base_height + self.hill_height * np.exp( -((np.asarray(x) - self.center[0])**2 + (np.asarray(y) - self.center[1])**2) / (2 * self.width**2))

    def gradient(self, x, y):

        return -(float(self.height(x, y)) - self.base_height) * ( np.array([x, y]) - self.center) / self.width**2

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
    """Seeded library Perlin layers sampled onto a cached 200 m height map.

    Height and slope queries use the same bilinear surface. Gym approximates
    this surface with triangles at its configured collision-mesh resolution.
    """
    plot_bounds = TERRAIN_BOUNDS

    def __init__(self, base_height = 0.0, seed = 17):

        from drone_sim.configuration import parameters

        super().__init__(base_height)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("Terrain seed must be an integer.")

        amplitude = positive(parameters.LANDSCAPE_HEIGHT, "Landscape height", allow_zero = True)
        feature_size = positive(parameters.LANDSCAPE_FEATURE_SIZE, "Landscape feature size")
        resolution = positive(parameters.LANDSCAPE_RESOLUTION, "Landscape resolution")
        persistence = positive(parameters.LANDSCAPE_PERSISTENCE, "Landscape persistence")
        lacunarity = positive(parameters.LANDSCAPE_LACUNARITY, "Landscape lacunarity")
        octaves = parameters.LANDSCAPE_OCTAVES

        if isinstance(octaves, bool) or not isinstance(octaves, (int, np.integer)) or not 1 <= octaves <= 8:
            raise ValueError("Landscape octaves must be an integer from 1 to 8.")
        if persistence > 1 or lacunarity <= 1:
            raise ValueError("Persistence must be at most 1 and lacunarity must exceed 1.")

        self._heights = _landscape_grid(int(seed), tuple(self.plot_bounds), amplitude, feature_size, int(octaves), persistence, lacunarity, resolution)
        xmin, xmax, ymin, ymax = self.plot_bounds
        ny, nx = self._heights.shape
        self._dx = (xmax - xmin) / (nx - 1)
        self._dy = (ymax - ymin) / (ny - 1)
        max_x_slope = np.max(np.abs(np.diff(self._heights, axis = 1))) / self._dx
        max_y_slope = np.max(np.abs(np.diff(self._heights, axis = 0))) / self._dy
        self.gradient_bound = float(np.hypot(max_x_slope, max_y_slope))

    def height(self, x, y):

        x, y = np.broadcast_arrays(np.asarray(x, dtype = float), np.asarray(y, dtype = float))
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Terrain coordinates must be finite.")

        xmin, xmax, ymin, ymax = self.plot_bounds
        ny, nx = self._heights.shape
        gx = (np.clip(x, xmin, xmax) - xmin) / self._dx
        gy = (np.clip(y, ymin, ymax) - ymin) / self._dy
        ix = np.minimum(gx.astype(int), nx - 2)
        iy = np.minimum(gy.astype(int), ny - 2)
        tx, ty = gx - ix, gy - iy
        lower = (1 - tx) * self._heights[iy, ix] + tx * self._heights[iy, ix + 1]
        upper = (1 - tx) * self._heights[iy + 1, ix] + tx * self._heights[iy + 1, ix + 1]
        return self.base_height + (1 - ty) * lower + ty * upper

    def gradient(self, x, y):

        # Central differences average the two slopes at a grid boundary.
        epsilon = 1e-5
        dx = (self.height(x + epsilon, y) - self.height(x - epsilon, y)) / (2 * epsilon)
        dy = (self.height(x, y + epsilon) - self.height(x, y - epsilon)) / (2 * epsilon)
        return np.array([float(dx), float(dy)])

    def maximum_height_along(self, start, goal):

        start = vector(start, 2, "Segment start")
        goal = vector(goal, 2, "Segment goal")
        length = np.linalg.norm(goal - start)
        segments = max(1, int(np.ceil(length / 0.15)))
        points = start + np.linspace(0, 1, segments + 1)[:, None] * (goal - start)
        # Bilinear slopes are bounded by adjacent sample differences, so this
        # Lipschitz margin covers every peak between the segment samples.
        return float(np.max(self.height(points[:, 0], points[:, 1]))) + self.gradient_bound * length / (2 * segments)


@lru_cache(maxsize = 4)
def _landscape_grid(seed, bounds, amplitude, feature_size, octaves, persistence, lacunarity, resolution):
    """Build once per configuration; instances share a read-only elevation grid."""

    from perlin_noise import PerlinNoise

    xmin, xmax, ymin, ymax = bounds
    nx = int(np.ceil((xmax - xmin) / resolution)) + 1
    ny = int(np.ceil((ymax - ymin) / resolution)) + 1
    if nx * ny > 250000:
        raise ValueError("Landscape grid exceeds 250000 samples; increase resolution spacing.")

    # The library treats seed=0 as random; map every project seed to a positive seed.
    rng = np.random.default_rng(seed % (2**64))
    offset = rng.uniform(10, 100, size = 2)
    points = [(float(x / feature_size + offset[0]), float(y / feature_size + offset[1]))
              for y in np.linspace(ymin, ymax, ny) for x in np.linspace(xmin, xmax, nx)]
    heights = np.zeros(nx * ny)
    weights = persistence**np.arange(octaves)

    for layer, weight in enumerate(weights):
        noise = PerlinNoise(octaves = lacunarity**layer, seed = int(rng.integers(1, 2**31)))
        heights += weight * np.fromiter((noise(point) for point in points), dtype = float, count = len(points))

    heights = (amplitude * heights / np.sum(weights)).reshape(ny, nx)
    heights -= heights.min()  # The lowest sampled terrain is base_height.
    heights.setflags(write = False)
    return heights
