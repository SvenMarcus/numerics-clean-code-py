from typing import Iterable, Protocol

import numpy as np
from numerics.boundaryconditions import Slice2D

from numerics.grid import Grid


class NumericalFunction(Protocol):
    def __call__(self, grid: Grid) -> np.ndarray:
        pass


class BoundaryCondition(NumericalFunction, Protocol):

    positions: Slice2D


def run_simulation(
    grid: Grid,
    numerical_scheme: NumericalFunction,
    boundary_conditions: Iterable[BoundaryCondition],
    number_of_timesteps: int,
) -> np.ndarray:
    for t in range(number_of_timesteps):
        grid._next_distribution[1:-1, 1:-1] = numerical_scheme(grid)
        for bc in boundary_conditions:
            grid._next_distribution[bc.positions] = bc(grid)

        grid.swap_distributions()

    return grid.distribution
