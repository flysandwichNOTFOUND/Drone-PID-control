import numpy as np
import math


class Navigation:
    def __init__(self, waypoints, current_waypoint_index, arrival_tolerance, speed_tolerance):
        self.waypoints = waypoints
        self.current_waypoint_index = current_waypoint_index

        self.arrival_tolerance = arrival_tolerance
        self.speed_tolerance = speed_tolerance

        self.mission_complete = False

    def get_current_target(self):
        return self.waypoints[self.current_waypoint_index]

    def distance_to_target(self, state):
        x, y, z = self.get_current_target()

        distance = math.sqrt(
            (x - state.position[0]) ** 2
            + (y - state.position[1]) ** 2
            + (z - state.position[2]) ** 2
        )

        return distance

    def has_arrived(self, state):
        distance = self.distance_to_target(state)
        speed = np.linalg.norm(state.velocity)

        return (
            distance <= self.arrival_tolerance
            and speed <= self.speed_tolerance
        )

    def advance_waypoint(self):
        if self.mission_complete:
            return

        if self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
        else:
            self.mission_complete = True

    def update(self, state):
        if not self.mission_complete and self.has_arrived(state):
            self.advance_waypoint()

        return self.get_current_target()

    def reset(self):
        self.current_waypoint_index = 0
        self.mission_complete = False

