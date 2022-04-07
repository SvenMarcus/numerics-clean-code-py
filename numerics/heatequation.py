from numba import njit
import numpy as np
from numerics.grid import Grid, Index2D
from numerics.numerical import NumericalFunction


def HeatEquation(
    thermal_diffusivity: float, timestep_delta: float
) -> NumericalFunction:
    @njit  # type: ignore
    def _heat_equation(
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        def _central_space_value(
            grid: Grid,
            prev: Index2D,
            central: Index2D,
            next: Index2D,
            node_distance: float,
        ) -> np.float64:
            return (grid.get(next) - 2 * grid.get(central) + grid.get(prev)) / (
                node_distance**2
            )

        y, x = position
        node_distance_in_y, node_distance_in_x = grid.node_distances
        return (
            grid.get(position)
            + (
                _central_space_value(
                    grid, (y - 1, x), (y, x), (y + 1, x), node_distance_in_x
                )
                + _central_space_value(
                    grid, (y, x - 1), (y, x), (y, x + 1), node_distance_in_y
                )
            )
            * timestep_delta
            * thermal_diffusivity
        )

    return _heat_equation
