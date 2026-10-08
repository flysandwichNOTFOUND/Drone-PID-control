"""Terrain-aware alignment, descent, flare, and touchdown detection."""
import numpy as np
from validation import positive, vector


class LandingController:
    def __init__(self, terrain, target_xy, descent_speed = 0.4, flare_speed = 0.12,
                 flare_height = 0.5, position_tolerance = 0.08, speed_tolerance = 0.15,
                 max_slope_degrees = 20.0, footprint_radius = 0.2):

        self.terrain = terrain
        self.target_xy = vector(target_xy, 2, "Landing target")
        self.descent_speed = positive(descent_speed, "Descent speed")
        self.flare_speed = positive(flare_speed, "Flare speed")
        self.flare_height = positive(flare_height, "Flare height")
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
        self.validate_site()

    def validate_site(self):

        # Check the center and the landing-gear footprint, not just the center.
        for offset in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            xy = self.target_xy + self.footprint_radius * np.asarray(offset)
            slope = np.arctan(np.linalg.norm(self.terrain.gradient(*xy)))
            if slope > self.max_slope:
                raise ValueError("Landing site is too steep; choose a gentler destination.")

    def target(self, state, dt):

        dt = positive(dt, "Time step")
        if self.landed or self.failed:
            return np.r_[self.target_xy, float(self.terrain.height(*state.position[:2]))], 0.0
        if self.commanded_height is None:
            self.commanded_height = float(state.position[2])
        ground = float(self.terrain.height(*state.position[:2]))
        clearance = state.position[2] - ground
        aligned = (np.linalg.norm(state.position[:2] - self.target_xy) <= self.position_tolerance
                   and np.linalg.norm(state.velocity[:2]) <= self.speed_tolerance
                   and np.linalg.norm(state.orientation[:2]) <= np.radians(10))
        velocity = 0.0
        if aligned:
            self.phase = "flare" if clearance <= self.flare_height else "descend"
            speed = self.flare_speed if self.phase == "flare" else self.descent_speed
            floor = ground - 0.03  # A small contact bias prevents hovering above the surface.
            self.commanded_height = max(floor, self.commanded_height - speed * dt)
            velocity = -speed if self.commanded_height > floor else 0.0
        else:
            self.phase = "align"
            self.commanded_height = max(self.commanded_height, float(state.position[2]), ground + 0.1)
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
