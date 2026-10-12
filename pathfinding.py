"""Terrain-aware A* search producing verified cruise waypoints.

This first design uses a 2D grid with one safe flight altitude per grid node.
It can compare routes around hills with routes over them, but does not search
every possible altitude as a full 3D planner would.

Planning flow:
    validate inputs -> build elevation grid -> search safe edges with A*
    -> reconstruct route -> connect exact endpoints -> verify every segment

RoutePlanner will add takeoff and landing to the returned cruise waypoints.
Navigation and the PID controllers will continue to execute that route.

Terrain must remain static while using a grid and provide height(x, y) plus
a conservative maximum_height_along(start_xy, end_xy). Search bounds restrict
the area explored; an altitude ceiling can exclude high terrain nodes.
"""
import heapq
from itertools import count

import numpy as np

from terrain import FlatTerrain, SlopedTerrain
from validation import positive, vector


class NoPathError(RuntimeError):
    """A required endpoint or segment has no verified feasible connection."""


class AStarPathfinder:
    def __init__(self, terrain, minimum_clearance, grid_resolution = 1.0,
                 minimum_flight_height = 2.0, climb_weight = 2.0,
                 search_bounds = None, search_margin = 5.0, maximum_altitude = None,
                 clearance_margin = 0.05, max_grid_nodes = 200000):
        """Store and validate planner settings.

        Heights are absolute world Z coordinates, in metres.
        minimum_clearance is the vertical distance above the terrain.
        grid_resolution is the spacing between grid nodes, in metres.
        climb_weight adds a cost per metre climbed; 2.0 is a starting choice.
        search_bounds is (xmin, xmax, ymin, ymax), or None for automatic bounds.
        search_margin expands automatic bounds around the start and goal.
        maximum_altitude optionally excludes nodes above a world Z ceiling.
        clearance_margin adds room for interpolation between terrain samples.
        max_grid_nodes limits memory use before allocating the grid.
        """

        self.terrain = terrain
        self.minimum_clearance = minimum_clearance
        self.grid_resolution = grid_resolution
        self.minimum_flight_height = minimum_flight_height
        self.climb_weight = climb_weight
        self.search_bounds = search_bounds
        self.search_margin = search_margin
        self.maximum_altitude = maximum_altitude
        self.clearance_margin = clearance_margin
        self.max_grid_nodes = max_grid_nodes
        self._validate_settings()

    def _validate_settings(self):

        self.minimum_clearance = positive(self.minimum_clearance, "Minimum clearance", allow_zero = True)
        self.grid_resolution = positive(self.grid_resolution, "Grid resolution")
        self.climb_weight = positive(self.climb_weight, "Climb weight", allow_zero = True)
        self.search_margin = positive(self.search_margin, "Search margin", allow_zero = True)
        self.clearance_margin = positive(self.clearance_margin, "Clearance margin", allow_zero = True)
        self.minimum_flight_height = float(self.minimum_flight_height)

        if not np.isfinite(self.minimum_flight_height):
            raise ValueError("Minimum flight height must be finite.")

        if self.maximum_altitude is not None:
            self.maximum_altitude = float(self.maximum_altitude)

            if not np.isfinite(self.maximum_altitude):
                raise ValueError("Maximum altitude must be finite.")

        if self.search_bounds is not None:
            self.search_bounds = vector(self.search_bounds, 4, "Search bounds")
            xmin, xmax, ymin, ymax = self.search_bounds

            if xmin > xmax or ymin > ymax:
                raise ValueError("Search bounds must have ordered minimum and maximum coordinates.")

        if (isinstance(self.max_grid_nodes, bool)
                or not isinstance(self.max_grid_nodes, (int, np.integer))
                or self.max_grid_nodes < 1):
            raise ValueError("Maximum grid nodes must be a positive integer.")

        self.max_grid_nodes = int(self.max_grid_nodes)

        for name in ["height", "maximum_height_along"]:
            if not callable(getattr(self.terrain, name, None)):
                raise ValueError(f"Terrain must provide {name}().")

    def _flight_position(self, xy):

        height = float(self.terrain.height(*xy))
        altitude = max(self.minimum_flight_height, height + self.minimum_clearance + self.clearance_margin)

        if not np.isfinite(height) or not np.isfinite(altitude):
            raise ValueError("Terrain and flight heights must be finite.")

        return np.array([xy[0], xy[1], altitude], dtype = float)

    def validate_inputs(self, start_xy, goal_xy):
        """Check finite XY pairs, settings, bounds, and terrain capabilities.

        Reject invalid resolutions, negative clearance or climb weight, and
        endpoints outside explicit bounds. Report invalid inputs separately
        from a valid search that finds no route.
        """

        self._validate_settings()
        start = vector(start_xy, 2, "Start XY")
        goal = vector(goal_xy, 2, "Goal XY")

        if self.search_bounds is not None:
            xmin, xmax, ymin, ymax = self.search_bounds

            for point in [start, goal]:
                if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
                    raise ValueError("Start and goal must lie inside the search bounds.")

        for point in [start, goal]:
            self._flight_position(point)

        return start, goal

    def build_grid(self, start_xy, goal_xy):
        """Sample terrain and store a world position for each grid index.

        Altitude is max(minimum_flight_height, terrain height + clearance
        + clearance_margin). Nodes above maximum_altitude are omitted.
        Return a dictionary with x/y axes, positions, an edge-clearance cache,
        and verified start/goal connector nodes. Integer (ix, iy) tuples name
        grid nodes; 'start' and 'goal' name the exact endpoints. Axis spacing
        is at most grid_resolution, and both search boundaries are included.
        """

        start, goal = self.validate_inputs(start_xy, goal_xy)
        start_position, goal_position = self._flight_position(start), self._flight_position(goal)

        if self.maximum_altitude is not None:
            if max(start_position[2], goal_position[2]) > self.maximum_altitude:
                raise NoPathError("An endpoint's flight altitude exceeds the altitude ceiling.")

        if self.search_bounds is None:
            low = np.minimum(start, goal) - self.search_margin
            high = np.maximum(start, goal) + self.search_margin
        else:
            low = self.search_bounds[[0, 2]]
            high = self.search_bounds[[1, 3]]

        with np.errstate(over = "ignore", invalid = "ignore"):
            steps = np.ceil((high - low) / self.grid_resolution)

        if not np.all(np.isfinite(steps)) or np.any(steps >= self.max_grid_nodes):
            raise ValueError("Grid is too large; increase spacing or reduce search bounds.")

        nx, ny = int(steps[0]) + 1, int(steps[1]) + 1

        if nx * ny > self.max_grid_nodes:
            raise ValueError("Grid exceeds max_grid_nodes; increase spacing or reduce search bounds.")

        x = np.linspace(low[0], high[0], nx)
        y = np.linspace(low[1], high[1], ny)
        positions = {"start": start_position, "goal": goal_position}

        for ix, px in enumerate(x):
            for iy, py in enumerate(y):
                position = self._flight_position((px, py))

                if self.maximum_altitude is None or position[2] <= self.maximum_altitude:
                    positions[(ix, iy)] = position

        grid = {"x": x, "y": y, "positions": positions, "edge_clearance": {}}
        grid["start_neighbors"] = self._endpoint_neighbors(start_position, grid)
        grid["goal_neighbors"] = set(self._endpoint_neighbors(goal_position, grid))

        return grid

    def _endpoint_neighbors(self, position, grid):
        """Find safe connector nodes in the containing cell and nearby ring."""

        indices = []

        for axis, coordinate in zip([grid["x"], grid["y"]], position[:2]):
            index = int(np.searchsorted(axis, coordinate))
            indices.append(range(max(0, index - 2), min(len(axis), index + 2)))

        return [(ix, iy) for ix in indices[0] for iy in indices[1]
                if (ix, iy) in grid["positions"]
                and self.segment_is_clear(position, grid["positions"][(ix, iy)])]

    def get_neighbors(self, node, grid):
        """Return up to eight adjacent grid nodes, including diagonals.

        Exclude unavailable nodes and edges that fail segment clearance.
        Safe endpoints alone do not establish a safe connecting segment.
        The virtual start node links to verified nearby grid nodes and to the
        goal when directly reachable. Grid nodes also link to a nearby goal.
        The future search must include movement_cost() for these connections.
        """

        positions = grid["positions"]

        if node not in positions:
            raise ValueError("Node is not available in this grid.")

        if node == "goal":
            return []

        if node == "start":
            candidates = list(grid["start_neighbors"]) + ["goal"]
        else:
            ix, iy = node
            candidates = [(ix + dx, iy + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                          if dx != 0 or dy != 0]
            if node in grid["goal_neighbors"]:
                candidates.append("goal")

        neighbors = []

        for candidate in candidates:
            if candidate not in positions:
                continue

            key = frozenset((node, candidate))

            if key not in grid["edge_clearance"]:
                grid["edge_clearance"][key] = self.segment_is_clear(positions[node], positions[candidate])

            if grid["edge_clearance"][key]:
                neighbors.append(candidate)

        return neighbors

    def segment_is_clear(self, start_position, end_position):
        """Check the entire 3D edge against terrain and the altitude ceiling.

        Check interpolated flight height minus terrain height along the edge.
        Terrain upper bounds certify subsegments. Uncertain intervals are
        bisected, up to depth 18 or 8192 interval checks. An interval that
        cannot be certified is rejected. Tolerance is 1e-8 metre. Point samples
        detect collisions but never establish safety on their own.
        """

        start = vector(start_position, 3, "Segment start")
        end = vector(end_position, 3, "Segment end")
        tolerance = 1e-8

        if min(start[2], end[2]) < self.minimum_flight_height - tolerance:
            return False

        if self.maximum_altitude is not None and max(start[2], end[2]) > self.maximum_altitude + tolerance:
            return False

        def clearance(point):

            height = float(self.terrain.height(*point[:2]))

            if not np.isfinite(height):
                raise ValueError("Terrain returned a nonfinite height.")

            return point[2] - height - self.minimum_clearance

        if min(clearance(start), clearance(end)) < -tolerance:
            return False

        # Clearance is linear for these exact flat and sloped terrain models.

        if type(self.terrain) in (FlatTerrain, SlopedTerrain):
            return True

        stack = [(start, end, 0)]
        checks = 0

        while stack:
            left, right, depth = stack.pop()
            checks += 1

            if checks > 8192:
                return False

            upper = float(self.terrain.maximum_height_along(left[:2], right[:2]))

            if not np.isfinite(upper):
                raise ValueError("Terrain returned a nonfinite segment height bound.")

            if min(left[2], right[2]) >= upper + self.minimum_clearance - tolerance:
                continue

            middle = (left + right) / 2.0

            if clearance(middle) < -tolerance or depth >= 18:
                return False

            stack.append((middle, right, depth + 1))
            stack.append((left, middle, depth + 1))

        return True

    def movement_cost(self, start_position, end_position):
        """Return 3D distance plus climb_weight times positive height gain.

        Descending earns no negative cost. This is a geometric preference,
        not a physical estimate of battery energy or flight time.
        """

        start = vector(start_position, 3, "Move start")
        end = vector(end_position, 3, "Move end")

        return float(np.linalg.norm(end - start) + self.climb_weight * max(0.0, end[2] - start[2]))

    def heuristic(self, position, goal_position):
        """Estimate remaining cost using straight-line 3D distance.

        With nonnegative climbing penalties, this is a lower bound on the
        proposed route cost and preserves A*'s grid optimality conditions.
        """

        position = vector(position, 3, "Position")
        goal = vector(goal_position, 3, "Goal position")

        return float(np.linalg.norm(goal - position))

    def reconstruct_path(self, came_from, goal_node, grid):
        """Return ordered XYZ points from parent links ending at a None root.

        The future search should initialize came_from['start'] = None.
        Missing parents and cycles are rejected instead of returning a partial
        path. Reconstruction does not replace final segment verification.
        """

        path, visited = [], set()
        node = goal_node

        if node is None:
            raise ValueError("A goal node is required to reconstruct a path.")

        while node is not None:
            if node in visited or node not in came_from or node not in grid["positions"]:
                raise ValueError("Path predecessor links are missing or cyclic.")

            visited.add(node)
            path.append(grid["positions"][node])
            node = came_from[node]

        return np.asarray(path[::-1], dtype = float).copy()

    def connect_endpoints(self, route, start_xy, goal_xy):
        """Include exact start/goal XY at safe cruise heights, not snapped XY.

        build_grid() prepares verified connector nodes for the future search.
        This function adds endpoints to an existing route, removes consecutive
        duplicate points, and checks every segment. It raises NoPathError for
        a blocked connection. The final goal is above the surface;
        RoutePlanner will add actual landing.
        """

        start, goal = self.validate_inputs(start_xy, goal_xy)
        route = np.asarray(route, dtype = float)

        if route.shape == (0,):
            route = np.empty((0, 3))

        if route.ndim != 2 or route.shape[1] != 3 or not np.all(np.isfinite(route)):
            raise ValueError("Route must be a finite (N, 3) array.")

        points = [self._flight_position(start)]

        for point in [*route, self._flight_position(goal)]:
            if not np.array_equal(point, points[-1]):
                points.append(point.copy())

        if self.search_bounds is not None:
            xmin, xmax, ymin, ymax = self.search_bounds

            if any(not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax) for p in points):
                raise NoPathError("Route leaves the search bounds.")

        edges = zip(points, points[1:]) if len(points) > 1 else [(points[0], points[0])]

        if any(not self.segment_is_clear(a, b) for a, b in edges):
            raise NoPathError("A route segment or endpoint connection lacks verified clearance.")

        return np.asarray(points, dtype = float)

    def plan(self, start_xy, goal_xy):
        """Return an (N, 3) array of verified cruise waypoints using A*.

        Use heapq for the frontier, g_score for best known costs, and
        came_from for parent links. Queue priority is g_score + heuristic.
        Update a neighbor when a cheaper valid route reaches it, and ignore
        stale queue entries. Reconstruct when the goal is reached; explicitly
        report no route if the frontier is exhausted. Do not silently replace
        a failed search with an unchecked straight route.

        The lowest-cost result applies to this finite graph and its endpoint
        connections, not every possible continuous 3D route. NoPathError may
        mean the grid, search bounds, or altitude ceiling are too restrictive.
        """

        grid = self.build_grid(start_xy, goal_xy)
        positions = grid["positions"]
        g_score = {"start": 0.0}
        came_from = {"start": None}
        # A serial number breaks equal-priority ties without comparing node
        # names: grid tuples and the string endpoints have different types.
        serial = count()
        frontier = [(self.heuristic(positions["start"], positions["goal"]),
                     next(serial), 0.0, "start")]

        while frontier:
            _, _, cost, node = heapq.heappop(frontier)

            if cost > g_score[node]:
                continue  # A cheaper entry for this node replaced this one.

            if node == "goal":
                route = self.reconstruct_path(came_from, node, grid)
                return self.connect_endpoints(route, start_xy, goal_xy)

            for neighbor in self.get_neighbors(node, grid):
                candidate_cost = cost + self.movement_cost(positions[node], positions[neighbor])

                if candidate_cost < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = candidate_cost
                    came_from[neighbor] = node
                    priority = candidate_cost + self.heuristic(positions[neighbor], positions["goal"])
                    heapq.heappush(frontier, (priority, next(serial), candidate_cost, neighbor))

        raise NoPathError("No verified route in this grid; try wider bounds, finer spacing, or a higher ceiling.")
