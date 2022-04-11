import numpy as np

from numerics.grid import Grid


class HeatEquation:
    def __init__(self, thermal_diffusivity: float, timestep_delta: float) -> None:
        self._thermal_diffusivity = thermal_diffusivity
        self._timestep_delta = timestep_delta

    def __call__(self, grid: Grid) -> np.ndarray:
        distribution = grid.distribution
        dy, dx = grid.node_distances
        central = distribution[1:-1, 1:-1]
        next_y, previous_y = distribution[2:, 1:-1], distribution[:-2, 1:-1]
        next_x, previous_x = distribution[1:-1, 2:], distribution[1:-1, :-2]
        return central + self._thermal_diffusivity * self._timestep_delta * (
            (next_y - 2 * central + previous_y) / (dx**2)
            + (next_x - 2 * central + previous_x) / (dy**2)
        )
