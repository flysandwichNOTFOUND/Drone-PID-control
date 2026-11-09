import numpy as np
from drone_sim.validation import positive, vector


class LandingController:
    def __init__(self, terrain, target_xy, descent_speed = 0.4, flare_speed = 0.12,
                 flare_height = 0.5, position_tolerance = 0.08, speed_tolerance = 0.15,
                 max_slope_degrees = 20.0, footprint_radius = 0.2,
                 mode = "strict", search_radius = None, search_resolution = None,
                 max_acceleration = 0.3, max_jerk = 0.5, approach_gain = 0.5, touchdown_speed = 0.02,
                 collision_body = None, touchdown_margin = 0.0, max_impact_speed = 0.35, spool_down_time = 0.8,
                 preferred_slope_degrees = None):
        """terrain should be the ground physics collides with (a CollisionSurface).

        collision_body: when given, clearance is the gap under the lowest point
        of the body (so a slope's uphill edge counts), not under the centre.
        touchdown_margin: clearance at which the flare reaches touchdown speed,
        leaving room for the smoothed command to settle before contact.
        After first contact the motors ease from flight thrust to zero over
        spool_down_time, so weight moves onto the ground gradually.
        preferred_slope_degrees: safe_nearby mode first picks the nearest site
        this flat, then falls back to max_slope_degrees.
        """

        self.terrain = terrain
        self.target_xy = vector(target_xy, 2, "Landing target")
        self.requested_xy = self.target_xy.copy()
        self.descent_speed = positive(descent_speed, "Descent speed")
        self.flare_speed = positive(flare_speed, "Flare speed")
        self.flare_height = positive(flare_height, "Flare height")
        self.approach_gain = positive(approach_gain, "Landing approach gain")
        self.minimum_touchdown_speed = positive(touchdown_speed, "Touchdown speed")

        if self.minimum_touchdown_speed > self.flare_speed:
            raise ValueError("Touchdown speed must not exceed flare speed.")
        self.position_tolerance = positive(position_tolerance, "Landing position tolerance")
        self.speed_tolerance = positive(speed_tolerance, "Landing speed tolerance")
        self.max_slope = np.radians(positive(max_slope_degrees, "Maximum landing slope"))

        if self.max_slope >= np.pi / 2:
            raise ValueError("Maximum landing slope must be less than 90 degrees.")

        self.footprint_radius = positive(footprint_radius, "Landing footprint")
        self.preferred_slope = (self.max_slope if preferred_slope_degrees is None else
                                min(np.radians(positive(preferred_slope_degrees, "Preferred landing slope", allow_zero = True)), self.max_slope))
        self.commanded_height = None
        self.phase = "align"
        self.landed = False
        self.failed = False
        self.touchdown_speed = None
        self.peak_contact_force = 0.0
        self.collision_body = collision_body
        self.touchdown_margin = positive(touchdown_margin, "Touchdown margin", allow_zero = True)
        self.max_impact_speed = positive(max_impact_speed, "Maximum impact speed")
        self.spool_down_time = positive(spool_down_time, "Spool-down time")
        self.spool_elapsed = 0.0
        self.touchdown_xy = None
        self.touchdown_height = None
        self.ground_attitude = None
        self.landing_position_error = None

        self.commanded_velocity = 0.0
        self.commanded_acceleration = 0.0

        self.max_acceleration = positive(max_acceleration, "Landing acceleration limit")
        self.max_jerk = positive(max_jerk, "Landing jerk limit")

        self.select_site(mode, search_radius, search_resolution)

    def site_slope(self, target_xy):
        """Return the maximum footprint slope in radians."""

        # Check the center and the landing-gear footprint, not just the center.
        target_xy = vector(target_xy, 2, "Landing target")
        slopes = []

        for offset in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            xy = target_xy + self.footprint_radius * np.asarray(offset)
            slopes.append(np.arctan(np.linalg.norm(self.terrain.gradient(*xy))))

        return float(max(slopes))

    def validate_site(self, target_xy = None):
        """Reject a site whose footprint exceeds the configured slope limit."""

        target_xy = self.target_xy if target_xy is None else target_xy

        if self.site_slope(target_xy) > self.max_slope:
            raise ValueError("Landing site is too steep; choose a gentler destination.")

    def find_safe_site(self, search_radius, search_resolution, max_slope = None):

        search_radius = positive(search_radius, "Landing search radius")
        search_resolution = positive(search_resolution, "Landing search resolution")

        offsets = [0.0]
        step = 1

        while step * search_resolution <= search_radius:
            distance = step * search_resolution
            offsets.insert(0, -distance)
            offsets.append(distance)
            step += 1

        # Keep only points inside the circular search area.
        candidates = []

        for dx in offsets:
            for dy in offsets:
                distance_squared = dx**2 + dy**2

                if distance_squared <= search_radius**2:
                    candidate_xy = self.requested_xy + np.array([dx, dy], dtype = float)
                    candidates.append((distance_squared, candidate_xy))

        max_slope = self.max_slope if max_slope is None else max_slope
        candidates.sort(key = lambda candidate: candidate[0])
        for distance_squared, candidate_xy in candidates:
            if self.site_slope(candidate_xy) <= max_slope:
                return candidate_xy.copy()

        raise ValueError("No suitable landing site found within the search radius.")

    def select_site(self, mode = "strict", search_radius = None, search_resolution = None):

        if mode not in ("strict", "safe_nearby", "exact"):
            raise ValueError("Landing mode must be strict, safe_nearby, or exact.")

        selected_xy = self.requested_xy.copy()

        if mode == "strict":
            self.validate_site(selected_xy)

        elif mode == "safe_nearby":
            slope = self.site_slope(selected_xy)

            if slope > self.preferred_slope:
                try:
                    selected_xy = self.find_safe_site(search_radius, search_resolution, self.preferred_slope)
                except ValueError:
                    if slope > self.max_slope:
                        selected_xy = self.find_safe_site(search_radius, search_resolution)
                self.validate_site(selected_xy)

        # Exact mode keeps the requested coordinates without rejecting the slope.
        self.target_xy = selected_xy.copy()
        return self.target_xy.copy()

    def clearance(self, state):
        """Gap between the drone's lowest point and the ground, in metres."""

        if self.collision_body is None:
            return float(state.position[2] - self.terrain.height(*state.position[:2]))

        return self.collision_body.clearance(self.terrain, state.position, state.orientation)

    def is_aligned(self, state):
        """Check position, horizontal speed, and tilt before descending."""

        return (np.linalg.norm(state.position[:2] - self.target_xy) <= self.position_tolerance
                and np.linalg.norm(state.velocity[:2]) <= self.speed_tolerance
                and np.linalg.norm(state.orientation[:2]) <= np.radians(10))

    def desired_descent_velocity(self, clearance):
        """Reduce descent speed near the surface while retaining gentle contact."""

        clearance = float(clearance)

        if not np.isfinite(clearance):
            raise ValueError("Landing clearance must be finite.")

        if clearance > self.flare_height:
            return -self.descent_speed

        speed = min(self.flare_speed, max(self.minimum_touchdown_speed, self.approach_gain * (clearance - self.touchdown_margin)))
        return -speed

    def update_descent_velocity(self, desired_velocity, dt):
        """Smooth signed vertical velocity with bounded acceleration and jerk."""

        dt = positive(dt, "Time step")
        desired_velocity = float(desired_velocity)

        if not np.isfinite(desired_velocity):
            raise ValueError("Desired velocity must be finite.")

        max_acceleration = positive(self.max_acceleration, "Landing acceleration limit")
        max_jerk = positive(self.max_jerk, "Landing jerk limit")

        # Damped velocity tracking reduces acceleration before reaching the target.
        response_rate = max_jerk / max_acceleration
        number_of_steps = max(1, int(np.ceil(dt * response_rate / 0.05)))
        step_dt = dt / number_of_steps

        for step in range(number_of_steps):
            velocity_error = desired_velocity - self.commanded_velocity
            desired_jerk = response_rate**2 * velocity_error - 2.0 * response_rate * self.commanded_acceleration
            jerk = np.clip(desired_jerk, -max_jerk, max_jerk)
            next_acceleration = np.clip(self.commanded_acceleration + jerk * step_dt, -max_acceleration, max_acceleration)

            # Integrate the ramp in acceleration without snapping velocity to target.
            self.commanded_velocity += 0.5 * (self.commanded_acceleration + next_acceleration) * step_dt
            self.commanded_acceleration = float(next_acceleration)

        return float(self.commanded_velocity)

    def target(self, state, dt):

        dt = positive(dt, "Time step")

        if self.landed or self.failed:
            return np.r_[state.position[:2], float(state.position[2])], 0.0

        if self.phase == "touchdown":
            # Hold where the drone touched down and press gently into the
            # ground while throttle_scale lowers the thrust.
            return np.r_[self.touchdown_xy, self.touchdown_height - 0.03], 0.0

        if self.commanded_height is None:
            self.commanded_height = float(state.position[2])

        clearance = self.clearance(state)
        ground = float(state.position[2]) - clearance  # centre height when just touching
        aligned = self.is_aligned(state)
        velocity = 0.0

        if aligned:
            self.phase = "flare" if clearance <= self.flare_height else "descend"
            desired_velocity = self.desired_descent_velocity(clearance)
        else:
            self.phase = "align"
            desired_velocity = 0.0

        velocity = self.update_descent_velocity(desired_velocity, dt)

        floor = ground - 0.03  # slightly below resting height so contact is reached
        previous_height = self.commanded_height
        self.commanded_height = max(floor, previous_height + velocity * dt)

        velocity = (self.commanded_height - previous_height) / dt

        return np.r_[self.target_xy, self.commanded_height], velocity

    @property
    def throttle_scale(self):
        """Fraction of controller thrust to apply: 1 in flight, eases to 0 on the ground."""

        if self.landed or self.failed:
            return 0.0

        if self.phase != "touchdown":
            return 1.0

        progress = min(self.spool_elapsed / self.spool_down_time, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def touchdown_orientation(self, yaw):
        """Attitude target while spooling down, or None to use normal control.

        The body starts level (friction holds it while thrust is high) and is
        steered onto the ground slope as thrust fades, so it rotates onto the
        surface under control instead of dropping onto its downhill edge.
        """

        if self.phase != "touchdown" or self.ground_attitude is None:
            return None

        blend = 1.0 - self.throttle_scale
        return np.array([blend * self.ground_attitude[0], blend * self.ground_attitude[1], yaw])

    def _ground_attitude(self, state):
        """Roll and pitch that put the body's bottom flat on the ground plane under it."""

        if self.collision_body is None:
            return None

        points = self.collision_body.bottom_body[:, :2] + state.position[:2]
        heights = np.asarray(self.terrain.height(points[:, 0], points[:, 1]), dtype = float)
        (slope_x, slope_y, _), *_ = np.linalg.lstsq(np.column_stack((points, np.ones(len(points)))), heights, rcond = None)
        normal = np.array([-slope_x, -slope_y, 1.0]) / np.sqrt(slope_x**2 + slope_y**2 + 1.0)
        yaw = float(state.orientation[2])
        # Express the normal in the yaw frame: body z = Ry(pitch) Rx(roll) e_z.
        nx = np.cos(yaw) * normal[0] + np.sin(yaw) * normal[1]
        ny = -np.sin(yaw) * normal[0] + np.cos(yaw) * normal[1]
        return np.array([np.arcsin(np.clip(-ny, -1.0, 1.0)), np.arctan2(nx, normal[2])])

    def check_touchdown(self, state, dt):
        """Call after each physics step. Returns True once the drone is landed and disarmed."""

        if self.landed or self.failed:
            return self.landed

        if self.phase == "touchdown":
            self.peak_contact_force = max(self.peak_contact_force, state.contact_force)
            self.spool_elapsed += positive(dt, "Time step")

            if self.spool_elapsed >= self.spool_down_time - 1e-12:
                self.landed = True
                self.phase = "landed"
                self.disarm(state)

            return self.landed

        if not state.ground_contact:
            return False

        # First contact: judge the impact, then hand weight to the ground slowly.
        self.touchdown_speed = state.contact_speed
        self.peak_contact_force = max(self.peak_contact_force, state.contact_force)

        if state.contact_speed > self.max_impact_speed:
            self.failed = True
            self.phase = "hard_landing"
            self.disarm(state)
            return False

        self.phase = "touchdown"
        self.spool_elapsed = 0.0
        self.touchdown_xy = state.position[:2].copy()
        self.touchdown_height = float(state.position[2])
        self.ground_attitude = self._ground_attitude(state)
        self.landing_position_error = float(np.linalg.norm(self.touchdown_xy - self.target_xy))
        return False

    @staticmethod
    def disarm(state):

        state.motor_speeds.fill(0.0)
        state.velocity.fill(0.0)
        state.angular_velocity.fill(0.0)
