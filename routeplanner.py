import numpy as np
from terrain import FlatTerrain
from pathfinding import AStarPathfinder

class RoutePlanner:
    def __init__(self, start_position, goal_position, ground_height, minimum_clearance, waypoint_spacing,
                 terrain = None, use_astar = True, grid_resolution = 1.0,
                 search_bounds = None, search_margin = 5.0, climb_weight = 2.0,
                 maximum_altitude = None, clearance_margin = 0.25):

        self.start_position = np.array(start_position, dtype = float)
        self.goal_position = np.array(goal_position, dtype = float)
        self.ground_height = np.float64(ground_height)

        self.minimum_clearance = float(minimum_clearance)
        self.waypoint_spacing = float(waypoint_spacing)
        self.waypoints = []
        self.terrain = terrain if terrain is not None else FlatTerrain(ground_height)
        self.landing_start_index = None
        self.cruise_altitude = None
        self.takeoff_waypoint_count = 0
        self.use_astar = use_astar
        self.grid_resolution = grid_resolution
        self.search_bounds = search_bounds
        self.search_margin = search_margin
        self.climb_weight = climb_weight
        self.maximum_altitude = maximum_altitude
        self.clearance_margin = clearance_margin

    def validate_inputs(self):

        if self.start_position.shape != (3,) or self.goal_position.shape != (3,):
            raise ValueError("Start and goal positions must each contain three coordinates.")

        if not np.all(np.isfinite(self.start_position)) or not np.all(np.isfinite(self.goal_position)):
            raise ValueError("Position coordinates must be finite.")

        if not np.isfinite(self.ground_height):
            raise ValueError("Ground height must be finite.")

        if not np.isfinite(self.waypoint_spacing) or self.waypoint_spacing <= 0:
            raise ValueError("Waypoint spacing must be finite and greater than zero.")

        if not np.isfinite(self.minimum_clearance) or self.minimum_clearance < 0:
            raise ValueError("Minimum clearance must be finite and nonnegative.")

        if self.start_position[2] < self.get_ground_height(self.start_position[0], self.start_position[1]):
            raise ValueError("Start position cannot be below ground.")


    def get_ground_height(self, x, y):

        return float(self.terrain.height(x, y))


    def get_minimum_altitude(self, x, y):

        return self.get_ground_height(x, y) + self.minimum_clearance


    def calculate_route_distance(self, start, goal):

        return float(np.linalg.norm(np.asarray(goal, dtype = float) - np.asarray(start, dtype = float)))

    def take_off(self, target_altitude = None):

        start = self.start_position.copy()
        takeoff_goal = start.copy()
        takeoff_goal[2] = max(start[2], self.get_minimum_altitude(start[0], start[1]))
        target_altitude = self.cruise_altitude if target_altitude is None else target_altitude
        if target_altitude is not None:
            if not np.isfinite(target_altitude) or target_altitude < takeoff_goal[2]:
                raise ValueError("Takeoff altitude must be finite and above the start and minimum clearance.")
            takeoff_goal[2] = target_altitude

        distance = self.calculate_route_distance(start, takeoff_goal)

        if distance == 0:
            return []

        segments = max(1, int(np.ceil(distance / self.waypoint_spacing)))

        return [start + (i / segments) * (takeoff_goal - start) for i in range(1, segments + 1)]


    def land(self, landing_start):

        start = np.asarray(landing_start, dtype = float).copy()
        landing_goal = start.copy()
        landing_goal[2] = self.get_ground_height(start[0], start[1])

        distance = self.calculate_route_distance(start, landing_goal)

        if distance == 0:
            return []

        segments = max(1, int(np.ceil(distance / self.waypoint_spacing)))
        waypoints = [start + (i / segments) * (landing_goal - start) for i in range(1, segments + 1)]
        waypoints[-1] = landing_goal.copy()

        return waypoints


    def generate_route(self, include_landing = False):

        self.validate_inputs()
        self.waypoints = []
        self.landing_start_index = None
        self.takeoff_waypoint_count = 0
        self.cruise_altitude = None
        if self.use_astar:
            pathfinder = AStarPathfinder(
                self.terrain, self.minimum_clearance, grid_resolution = self.grid_resolution,
                minimum_flight_height = max(self.start_position[2], self.goal_position[2]),
                climb_weight = self.climb_weight, search_bounds = self.search_bounds,
                search_margin = self.search_margin, maximum_altitude = self.maximum_altitude,
                clearance_margin = self.clearance_margin)
            cruise_route = pathfinder.plan(self.start_position[:2], self.goal_position[:2])
            # Shortcuts remove unnecessary grid turns only when every point
            # along the new segment is clear and its cost does not increase.
            simplified = [cruise_route[0]]
            index = 0
            while index < len(cruise_route) - 1:
                next_index = index + 1
                accumulated_cost = 0.0
                for candidate in range(index + 1, len(cruise_route)):
                    accumulated_cost += pathfinder.movement_cost(cruise_route[candidate - 1], cruise_route[candidate])
                    if (pathfinder.movement_cost(cruise_route[index], cruise_route[candidate]) <= accumulated_cost + 1e-8
                            and pathfinder.segment_is_clear(cruise_route[index], cruise_route[candidate])):
                        next_index = candidate
                simplified.append(cruise_route[next_index])
                index = next_index
            cruise_route = np.asarray(simplified)
        else:
            altitude = max(self.start_position[2], self.goal_position[2],
                           self.terrain.maximum_height_along(self.start_position[:2], self.goal_position[:2])
                           + self.minimum_clearance)
            cruise_route = np.array([np.r_[self.start_position[:2], altitude],
                                     np.r_[self.goal_position[:2], altitude]])

        self.cruise_altitude = float(cruise_route[0, 2])
        self.waypoints.extend(self.take_off(target_altitude = self.cruise_altitude))
        self.takeoff_waypoint_count = len(self.waypoints)
        start = self.waypoints[-1].copy() if self.waypoints else self.start_position.copy()

        for goal in cruise_route[1:]:
            distance = self.calculate_route_distance(start, goal)
            if distance > 0:
                segments = max(1, int(np.ceil(distance / self.waypoint_spacing)))
                self.waypoints.extend(start + (i / segments) * (goal - start) for i in range(1, segments + 1))
                self.waypoints[-1] = goal.copy()
            start = goal.copy()
        if not self.waypoints:
            self.waypoints.append(cruise_route[-1].copy())

        if include_landing:
            self.landing_start_index = len(self.waypoints)
            self.waypoints.extend(self.land(self.waypoints[-1]))
            self.landing_start_index = min(self.landing_start_index, len(self.waypoints) - 1)

        return self.waypoints


    def get_waypoints(self):

        return self.waypoints

