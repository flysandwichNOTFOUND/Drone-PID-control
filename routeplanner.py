import numpy as np

class RoutePlanner:
    def __init__(self, start_position, goal_position, ground_height, minimum_clearance, waypoint_spacing):
        self.start_position = np.array(start_position, dtype=float)
        self.goal_position = np.array(goal_position, dtype=float)
        self.ground_height = np.float64(ground_height)

        self.minimum_clearance = float(minimum_clearance)
        self.waypoint_spacing = float(waypoint_spacing)
        self.waypoints = []

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
        return self.ground_height


    def get_minimum_altitude(self, x, y):
        return self.get_ground_height(x, y) + self.minimum_clearance


    def calculate_route_distance(self, start, goal):
        return float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)))

    def take_off(self):
        start = self.start_position.copy()
        takeoff_goal = start.copy()
        takeoff_goal[2] = max(start[2], self.get_minimum_altitude(start[0], start[1]))

        distance = self.calculate_route_distance(start, takeoff_goal)

        if distance == 0:
            return []

        segments = max(1, int(np.ceil(distance / self.waypoint_spacing)))

        return [start + (i / segments) * (takeoff_goal - start) for i in range(1, segments + 1)]


    def land(self, landing_start):
        start = np.asarray(landing_start, dtype=float).copy()
        landing_goal = start.copy()
        landing_goal[2] = self.get_ground_height(start[0], start[1])

        distance = self.calculate_route_distance(start, landing_goal)

        if distance == 0:
            return []

        segments = max(1, int(np.ceil(distance / self.waypoint_spacing)))
        waypoints = [start + (i / segments) * (landing_goal - start) for i in range(1, segments + 1)]
        waypoints[-1] = landing_goal.copy()

        return waypoints


    def generate_route(self, include_landing=False):
        self.validate_inputs()
        self.waypoints = []

        self.waypoints.extend(self.take_off())
        start = self.waypoints[-1].copy() if self.waypoints else self.start_position.copy()

        goal = self.goal_position.copy()
        goal[2] = max(goal[2], self.get_minimum_altitude(goal[0], goal[1]))

        distance = self.calculate_route_distance(start, goal)

        if distance > 0:
            segments = max(1, int(np.ceil(distance / self.waypoint_spacing)))

            for i in range(1, segments + 1):
                waypoint = start + (i / segments) * (goal - start)
                self.waypoints.append(waypoint)

            self.waypoints[-1] = goal.copy()

        elif not self.waypoints:
            self.waypoints.append(goal.copy())

        if include_landing:
            self.waypoints.extend(self.land(self.waypoints[-1]))

        return self.waypoints


    def get_waypoints(self):
        return self.waypoints

