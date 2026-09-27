import numpy as np


class BasicEnvironment:
    def __init__(self, wind_velocity, wind_force_coefficient):
        self.wind_velocity = np.array(
            wind_velocity,
            dtype=float
        )

        self.wind_force_coefficient = wind_force_coefficient

    def calculate_wind_force(self, state):
        relative_air_velocity = (
            self.wind_velocity - state.velocity
        )

        wind_force = (
            self.wind_force_coefficient
            * relative_air_velocity
        )

        return wind_force