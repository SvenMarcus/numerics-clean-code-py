import numpy as np
from numerics.grid import Grid, Index2D


class HeatEquation:
    def __init__(self, thermal_diffusivity: float, timestep_delta: float) -> None:
        self._thermal_diffusivity = thermal_diffusivity
        self._timestep_delta = timestep_delta

    def __call__(
        self,
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        y, x = position
        node_distance_in_y, node_distance_in_x = grid.node_distances
        return (
            grid.distribution[y, x]
            + (
                (
                    grid.distribution[y + 1, x]
                    - 2 * grid.distribution[y, x]
                    + grid.distribution[y - 1, x]
                )
                / (node_distance_in_x**2)
                + (
                    grid.distribution[y, x + 1]
                    - 2 * grid.distribution[y, x]
                    + grid.distribution[y, x - 1]
                )
                / (node_distance_in_y**2)
            )
            * self._timestep_delta
            * self._thermal_diffusivity
        )
