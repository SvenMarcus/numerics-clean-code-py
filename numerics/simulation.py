from typing import Dict, Protocol

import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from numba import literal_unroll, njit

from numerics.grid import Grid, Index2D
from numerics.numerical import NumericalFunction


BoundaryConditionMap = Dict[Index2D, NumericalFunction]


@njit  # type: ignore
def run(
    numerical_scheme: NumericalFunction,
    boundary_conditions: BoundaryConditionMap,
    grid: Grid,
    number_of_timesteps: int,
) -> npt.NDArray[np.float64]:
    for t in range(number_of_timesteps):
        for current_position in grid.iter_index():
            next_value = _get_next_function(
                grid, numerical_scheme, boundary_conditions, current_position
            )
            grid.set_next(current_position, next_value)

        grid.swap_distributions()

    return grid.distribution


@njit  # type: ignore
def _get_next_function(
    grid: Grid,
    numerical_scheme: NumericalFunction,
    boundary_conditions: BoundaryConditionMap,
    current_position: Index2D,
) -> np.float64:
    if current_position in boundary_conditions:
        return boundary_conditions[current_position](grid, current_position)
    else:
        return numerical_scheme(grid, current_position)
