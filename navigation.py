import numpy as np
import math
from validation import positive, vector


class Navigation:
    def __init__(self, waypoints, current_waypoint_index, arrival_tolerance, speed_tolerance):

        self.waypoints = np.asarray(waypoints, dtype = float).copy()
        if (self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3
                or len(self.waypoints) == 0 or not np.all(np.isfinite(self.waypoints))):
            raise ValueError("Waypoints must be a nonempty array of finite 3D positions.")
        if (not isinstance(current_waypoint_index, (int, np.integer))
                or not 0 <= current_waypoint_index < len(self.waypoints)):
            raise ValueError("Current waypoint index is out of range.")
        self.current_waypoint_index = current_waypoint_index

        self.arrival_tolerance = positive(arrival_tolerance, "Arrival tolerance", allow_zero = True)
        self.speed_tolerance = positive(speed_tolerance, "Speed tolerance", allow_zero = True)

        self.mission_complete = False


    def get_current_target(self):

        return self.waypoints[self.current_waypoint_index]

    def distance_to_target(self, state):

        x, y, z = self.get_current_target()

        distance = math.sqrt(
            (x - state.position[0])**2
 + (y - state.position[1])**2
 + (z - state.position[2])**2
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


class PathFollower(Navigation):
    """Follow collinear waypoint sections without stopping at each waypoint.

    A fifth-degree S-curve moves the reference ahead along the path.
    Sharp corners finish at rest so the existing climb-then-cross route stays
    intact. Look-ahead throttling prevents a stalled drone losing its reference.
    """
    def __init__(self, waypoints, start_position, arrival_tolerance, speed_tolerance,
                 max_speed = 1.2, max_acceleration = 0.8, max_climb_rate = 0.8,
                 max_descent_rate = 0.4, lookahead_distance = 1.0, stop_index = None,
                 max_jerk = 1.0):

        super().__init__(waypoints, 0, arrival_tolerance, speed_tolerance)
        self.start_position = vector(start_position, 3, "Path start")
        self.max_speed = positive(max_speed, "Maximum path speed")
        self.max_acceleration = positive(max_acceleration, "Maximum path acceleration")
        self.max_jerk = positive(max_jerk, "Maximum path jerk")
        self.max_climb_rate = positive(max_climb_rate, "Maximum climb rate")
        self.max_descent_rate = positive(max_descent_rate, "Maximum descent rate")
        self.lookahead_distance = positive(lookahead_distance, "Look-ahead distance")
        self.stop_index = len(self.waypoints) - 1 if stop_index is None else stop_index
        if not isinstance(self.stop_index, (int, np.integer)) or not 0 <= self.stop_index < len(self.waypoints):
            raise ValueError("Path stop index is out of range.")

        points = [self.start_position]
        anchors = [0]
        for index, point in enumerate(self.waypoints[:self.stop_index + 1]):
            if np.linalg.norm(point - points[-1]) > 1e-10:
                points.append(point)
                anchors.append(index + 1)
            else:
                anchors[-1] = index + 1
        self.path = np.asarray(points, dtype = float)
        self.anchors = np.asarray(anchors)
        differences = np.diff(self.path, axis = 0)
        self.lengths = np.linalg.norm(differences, axis = 1)
        self.distances = np.r_[0.0, np.cumsum(self.lengths)]
        self.directions = differences / self.lengths[:, None] if len(self.lengths) else differences
        corners = [0]
        for index in range(1, len(self.directions)):
            if np.dot(self.directions[index - 1], self.directions[index]) < 1.0 - 1e-8:
                corners.append(index)
        if len(self.path) > 1:
            corners.append(len(self.path) - 1)
        self.corners = corners
        self.reset()

    def reset(self):

        super().reset()
        self.section = 0
        self.elapsed = 0.0
        self.clock_rate = 1.0
        self.clock_input = 1.0
        self.progress = 0.0
        self.target_position = self.start_position.copy()
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.profile_finished = len(self.path) == 1

    def check_arrival(self, state):

        return (np.linalg.norm(state.position - self.path[-1]) <= self.arrival_tolerance
                and np.linalg.norm(state.velocity) <= self.speed_tolerance)

    def update(self, state, dt):

        dt = positive(dt, "Time step")
        if self.mission_complete:
            return self.target_position.copy()
        if len(self.path) == 1:
            self.current_waypoint_index = self.stop_index
            self.mission_complete = self.check_arrival(state)
            return self.target_position.copy()

        first, last = self.corners[self.section:self.section + 2]
        start, end = self.path[first], self.path[last]
        direction = (end - start) / np.linalg.norm(end - start)
        length = self.distances[last] - self.distances[first]
        speed_limit = self.max_speed
        if abs(direction[2]) > 1e-10:
            vertical_limit = self.max_climb_rate if direction[2] > 0 else self.max_descent_rate
            speed_limit = min(speed_limit, vertical_limit / abs(direction[2]))

        # Reserve acceleration and jerk for smooth look-ahead clock changes.
        acceleration = self.max_acceleration / 2.0
        jerk = self.max_jerk / 2.0
        # Exact maxima of the quintic's normalized first three derivatives.
        total_time = max(1.875 * length / speed_limit,
                         np.sqrt((10.0 / np.sqrt(3.0)) * length / acceleration),
                         np.cbrt(60.0 * length / jerk))
        measured_progress = np.clip(np.dot(state.position - start, direction), 0.0, length)
        lead = self.progress - self.distances[first] - measured_progress
        # Two cascaded filters keep the clock rate and its derivative continuous.
        # Their derivative bounds keep the time-warped reference within limits.
        profile_peak_speed = 1.875 * length / total_time
        profile_peak_acceleration = (10.0 / np.sqrt(3.0)) * length / total_time**2
        profile_peak_jerk = 60.0 * length / total_time**3
        remaining_jerk = self.max_jerk - profile_peak_jerk
        clock_gain = min((self.max_acceleration - profile_peak_acceleration) / profile_peak_speed,
                         remaining_jerk / (6.0 * profile_peak_acceleration),
                         np.sqrt(remaining_jerk / (4.0 * profile_peak_speed)))
        desired_clock_rate = float(lead < self.lookahead_distance)
        decay = np.exp(-clock_gain * dt)
        input_error = self.clock_input - desired_clock_rate
        rate_error = self.clock_rate - desired_clock_rate
        increment = (desired_clock_rate * dt
                     + (rate_error + input_error) * (-np.expm1(-clock_gain * dt)) / clock_gain
                     - input_error * dt * decay)
        self.clock_input = desired_clock_rate + input_error * decay
        self.clock_rate = desired_clock_rate + (rate_error + clock_gain * dt * input_error) * decay
        clock_derivative = clock_gain * (self.clock_input - self.clock_rate)
        self.elapsed = min(total_time, self.elapsed + increment)

        fraction = self.elapsed / total_time
        distance = length * fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
        speed = length / total_time * 30.0 * fraction**2 * (1.0 - fraction)**2
        profile_acceleration = length / total_time**2 * 60.0 * fraction * (1.0 - fraction) * (1.0 - 2.0 * fraction)

        self.target_position = start + direction * distance
        self.target_velocity = direction * speed * self.clock_rate
        self.target_acceleration = direction * (profile_acceleration * self.clock_rate**2
                                                + speed * clock_derivative)
        self.progress = self.distances[first] + distance
        segment = min(len(self.path) - 2, np.searchsorted(self.distances, self.progress, side = "right") - 1)
        self.current_waypoint_index = min(self.stop_index, int(self.anchors[segment + 1]) - 1)
        self.profile_finished = self.elapsed >= total_time
        if self.profile_finished:
            self.target_velocity.fill(0.0)
            self.target_acceleration.fill(0.0)
            arrived = (np.linalg.norm(state.position - end) <= self.arrival_tolerance
                       and np.linalg.norm(state.velocity) <= self.speed_tolerance)
            if arrived:
                if self.section == len(self.corners) - 2:
                    self.mission_complete = True
                else:
                    self.section += 1
                    self.elapsed = 0.0
                    self.clock_rate = 1.0
                    self.clock_input = 1.0
        return self.target_position.copy()

