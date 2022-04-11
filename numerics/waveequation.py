import numpy as np
from numerics.grid import Grid


class WaveEquation:
    def __init__(
        self,
        wave_propagation_velocity: float,
        timestep_delta: float,
    ) -> None:
        self._wave_propagation_velocity = wave_propagation_velocity
        self._timestep_delta = timestep_delta

    def __call__(self, grid: Grid) -> np.ndarray:
        distribution = grid.distribution
        previous_distribution = grid.next_distribution
        central = distribution[1:-1, 1:-1]
        previous_central = previous_distribution[1:-1, 1:-1]
        next_y, previous_y = distribution[2:, 1:-1], distribution[:-2, 1:-1]
        next_x, previous_x = distribution[1:-1, 2:], distribution[1:-1, :-2]
        dy, dx = grid.node_distances
        return (
            2.0 * central
            - previous_central
            + (self._wave_propagation_velocity**2)
            * (self._timestep_delta**2)
            * (
                (next_y - 2.0 * central + previous_y) / (dx**2)
                + (next_x - 2.0 * central + previous_x) / (dy**2)
            )
        )
