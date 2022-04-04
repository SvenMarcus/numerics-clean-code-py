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
            grid.get(position)
            + (
                self._central_space_value(
                    grid, (y - 1, x), (y, x), (y + 1, x), node_distance_in_x
                )
                + self._central_space_value(
                    grid, (y, x - 1), (y, x), (y, x + 1), node_distance_in_y
                )
            )
            * self._timestep_delta
            * self._thermal_diffusivity
        )

    def _central_space_value(
        self,
        grid: Grid,
        prev: Index2D,
        central: Index2D,
        next: Index2D,
        node_distance: float,
    ) -> np.float64:
        return (grid.get(next) - 2 * grid.get(central) + grid.get(prev)) / (
            node_distance**2
        )
