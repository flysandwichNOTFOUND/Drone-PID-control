"""Terrain-aware alignment, descent, flare, and touchdown detection."""
import numpy as np
from drone_sim.validation import positive, vector


class LandingController:
    def __init__(self, terrain, target_xy, descent_speed = 0.4, flare_speed = 0.12,
                 flare_height = 0.5, position_tolerance = 0.08, speed_tolerance = 0.15,
                 max_slope_degrees = 20.0, footprint_radius = 0.2,
                 mode = "strict", search_radius = None, search_resolution = None,
                 max_acceleration = 0.3, max_jerk = 0.5, approach_gain = 0.5, touchdown_speed = 0.02):

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
        self.commanded_height = None
        self.phase = "align"
        self.landed = False
        self.failed = False
        self.touchdown_speed = None

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

    def find_safe_site(self, search_radius, search_resolution):

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

        candidates.sort(key = lambda candidate: candidate[0])
        for distance_squared, candidate_xy in candidates:
            if self.site_slope(candidate_xy) <= self.max_slope:
                return candidate_xy.copy()

        raise ValueError("No suitable landing site found within the search radius.")

    def select_site(self, mode = "strict", search_radius = None, search_resolution = None):

        if mode not in ("strict", "safe_nearby", "exact"):
            raise ValueError("Landing mode must be strict, safe_nearby, or exact.")

        selected_xy = self.requested_xy.copy()

        if mode == "strict":
            self.validate_site(selected_xy)

        elif mode == "safe_nearby":
            if self.site_slope(selected_xy) > self.max_slope:


                selected_xy = self.find_safe_site(search_radius, search_resolution)
                self.validate_site(selected_xy)

        # Exact mode keeps the requested coordinates without rejecting the slope.
        self.target_xy = selected_xy.copy()
        return self.target_xy.copy()

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

        speed = min(self.flare_speed, max(self.minimum_touchdown_speed, self.approach_gain * max(clearance, 0.0)))
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
            return np.r_[self.target_xy, float(self.terrain.height(*state.position[:2]))], 0.0

        if self.commanded_height is None:
            self.commanded_height = float(state.position[2])

        ground = float(self.terrain.height(*state.position[:2]))
        clearance = state.position[2] - ground
        aligned = self.is_aligned(state)
        velocity = 0.0

        if aligned:
            self.phase = "flare" if clearance <= self.flare_height else "descend"
            desired_velocity = self.desired_descent_velocity(clearance)
        else:
            self.phase = "align"
            desired_velocity = 0.0

        velocity = self.update_descent_velocity(desired_velocity, dt)

        floor = ground - 0.03
        previous_height = self.commanded_height
        self.commanded_height = max(floor, previous_height + velocity * dt)

        velocity = (self.commanded_height - previous_height) / dt

        return np.r_[self.target_xy, self.commanded_height], velocity

    def check_touchdown(self, state):

        if not state.ground_contact:
            return False

        self.touchdown_speed = state.contact_speed

        if state.contact_speed > 0.35:
            self.failed = True
            self.phase = "hard_landing"
            self.disarm(state)
            return False

        if (np.linalg.norm(state.position[:2] - self.target_xy) <= self.position_tolerance
                and np.linalg.norm(state.velocity) <= self.speed_tolerance):
            self.landed = True
            self.phase = "landed"
            self.disarm(state)

        return self.landed

    @staticmethod
    def disarm(state):

        state.motor_speeds.fill(0.0)
        state.velocity.fill(0.0)
        state.angular_velocity.fill(0.0)
