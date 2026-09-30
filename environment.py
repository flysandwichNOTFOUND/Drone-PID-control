import numpy as np


class BasicEnvironment:
    def __init__(self, wind_mode, wind_velocity, wind_force_coefficient, gust_start=10.0, gust_duration=2.0, gust_period=6.0):

        valid_modes = ["none", "constant", "gust"]

        if wind_mode not in valid_modes:
            raise ValueError(f"wind_mode must be one of {valid_modes}.")

        self.wind_mode = wind_mode

        self.wind_velocity = np.asarray(wind_velocity, dtype = float)

        self.wind_force_coefficient = (wind_force_coefficient)

        self.gust_start = gust_start
        self.gust_duration = gust_duration
        self.gust_period = gust_period

    def wind_velocity_at_time(self, current_time):

        if self.wind_mode == "none":
            return np.zeros(3, dtype=float)

        if self.wind_mode == "constant":
            return self.wind_velocity

        # No gust before gust_start
        if current_time < self.gust_start:
            return np.zeros(3, dtype=float)

        gust_time = (current_time - self.gust_start) % self.gust_period

        # Wind is active during the gust duration
        if gust_time < self.gust_duration:
            return self.wind_velocity

        return np.zeros(3, dtype=float)

    def calculate_wind_force(self, state, current_time):

        current_wind_velocity = (self.wind_velocity_at_time(current_time))
        relative_air_velocity = (current_wind_velocity - state.velocity)
        wind_force = (self.wind_force_coefficient * relative_air_velocity)

        return wind_force
