"""Ground and body shapes shared by both physics backends.

Both backends collide the same cylinder against the same surface:
- CollisionSurface: flat ground is an exact plane; every other terrain is the
  triangle mesh that PyBullet loads, so the custom model sees identical heights.
- CollisionBody: a cylinder centred on the centre of mass. Its bottom disk
  touches the ground, so the resting centre sits half a body height up.
"""

import numpy as np

from drone_sim.physics.attitude import rotation_matrix
from drone_sim.validation import positive, vector
from drone_sim.world.terrain import FlatTerrain


class CollisionSurface:
    """Height, slope and mesh of the ground that physics collides with."""

    def __init__(self, terrain, bounds, resolution):

        if not callable(getattr(terrain, "height", None)):
            raise ValueError("Terrain must provide height(x, y).")

        self.terrain = terrain
        self.base_height = float(getattr(terrain, "base_height", 0.0))
        self.plot_bounds = getattr(terrain, "plot_bounds", None)
        self.is_plane = type(terrain) is FlatTerrain

        if self.is_plane:
            return

        self.bounds = vector(bounds, 4, "Collision mesh bounds")
        xmin, xmax, ymin, ymax = self.bounds
        resolution = positive(resolution, "Collision mesh resolution")

        if not (xmin < xmax and ymin < ymax):
            raise ValueError("Collision mesh bounds must be increasing.")

        nx = int(np.ceil((xmax - xmin) / resolution)) + 1
        ny = int(np.ceil((ymax - ymin) / resolution)) + 1

        if nx * ny > 250000:
            raise ValueError("Collision mesh exceeds 250000 vertices; increase resolution spacing.")

        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.dx = (xmax - xmin) / (nx - 1)
        self.dy = (ymax - ymin) / (ny - 1)
        grid_x, grid_y = np.meshgrid(self.x, self.y)
        heights = np.asarray(terrain.height(grid_x, grid_y), dtype = float)

        if heights.shape != grid_x.shape or not np.all(np.isfinite(heights)):
            raise ValueError("Terrain must return finite heights for the collision mesh.")

        self.heights = heights  # shape (ny, nx), row = y index
        self.heights.setflags(write = False)


    def _cell(self, x, y):

        x, y = np.broadcast_arrays(np.asarray(x, dtype = float), np.asarray(y, dtype = float))

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Ground coordinates must be finite.")

        xmin, xmax, ymin, ymax = self.bounds
        ny, nx = self.heights.shape
        gx = (np.clip(x, xmin, xmax) - xmin) / self.dx
        gy = (np.clip(y, ymin, ymax) - ymin) / self.dy
        ix = np.minimum(gx.astype(int), nx - 2)
        iy = np.minimum(gy.astype(int), ny - 2)
        h = self.heights
        return (gx - ix, gy - iy, h[iy, ix], h[iy, ix + 1], h[iy + 1, ix], h[iy + 1, ix + 1])


    def height(self, x, y):
        """Piecewise-planar height on the exact triangles PyBullet uses."""

        if self.is_plane:
            x, y = np.broadcast_arrays(x, y)
            return np.full(np.shape(x), self.base_height)

        tx, ty, z00, z10, z01, z11 = self._cell(x, y)
        lower = z00 + tx * (z10 - z00) + ty * (z01 - z00)
        upper = z11 + (1 - tx) * (z01 - z11) + (1 - ty) * (z10 - z11)
        return np.where(tx + ty <= 1.0, lower, upper)


    def gradient(self, x, y):

        if self.is_plane:
            return np.zeros(2)

        tx, ty, z00, z10, z01, z11 = (float(value) for value in self._cell(x, y))

        if tx + ty <= 1.0:
            return np.array([(z10 - z00) / self.dx, (z01 - z00) / self.dy])

        return np.array([(z11 - z01) / self.dx, (z11 - z10) / self.dy])


    def normal(self, x, y):

        normal = np.r_[-self.gradient(x, y), 1.0]
        return normal / np.linalg.norm(normal)


    def mesh(self):
        """Return vertices (N, 3) and counter-clockwise triangle indices (M, 3)."""

        if self.is_plane:
            raise ValueError("A plane has no mesh.")

        ny, nx = self.heights.shape
        grid_x, grid_y = np.meshgrid(self.x, self.y)
        vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), self.heights.ravel()))
        corner = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
        # Split each cell along the (x+1, y)-(x, y+1) diagonal, matching height().
        lower = np.column_stack((corner, corner + 1, corner + nx))
        upper = np.column_stack((corner + 1, corner + nx + 1, corner + nx))
        faces = np.empty((2 * len(corner), 3), dtype = np.int64)
        faces[0::2], faces[1::2] = lower, upper
        return vertices, faces


class CollisionBody:
    """Cylinder centred on the centre of mass; the bottom disk meets the ground."""

    def __init__(self, radius, height, rim_points = 12):

        self.radius = positive(radius, "Body collision radius")
        self.height = positive(height, "Body collision height")
        self.half_height = self.height / 2.0
        angles = np.linspace(0.0, 2.0 * np.pi, int(rim_points), endpoint = False)
        rim = np.column_stack((self.radius * np.cos(angles), self.radius * np.sin(angles), np.full(len(angles), -self.half_height)))
        self.bottom_body = np.vstack(([0.0, 0.0, -self.half_height], rim))


    def bottom_points(self, position, orientation):
        """World positions of the bottom-disk sample points."""

        return np.asarray(position, dtype = float) + self.bottom_body @ rotation_matrix(orientation).T


    def clearance(self, surface, position, orientation):
        """Smallest bottom-to-ground gap in metres (negative means overlap)."""

        points = self.bottom_points(position, orientation)
        return float(np.min(points[:, 2] - surface.height(points[:, 0], points[:, 1])))


    def resting_height(self, surface, x, y):
        """Centre-of-mass height at which a level body just touches the ground."""

        points = self.bottom_body[:, :2] + np.array([x, y], dtype = float)
        return float(np.max(surface.height(points[:, 0], points[:, 1]))) + self.half_height


def collision_surface(terrain, config = None):
    """Build the shared ground from parameters.py collision settings."""

    if config is None:
        from drone_sim.configuration import parameters as config

    return CollisionSurface(terrain, config.COLLISION_MESH_BOUNDS, config.COLLISION_MESH_RESOLUTION)


def collision_body(config = None):

    if config is None:
        from drone_sim.configuration import parameters as config

    return CollisionBody(config.COLLISION_BODY_RADIUS, config.COLLISION_BODY_HEIGHT)
