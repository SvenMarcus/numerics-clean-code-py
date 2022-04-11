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
    numerical_scheme: NumericalFunction,
    grid: Grid,
    nt: int,
    bc: Iterable[BoundaryCondition],
) -> np.ndarray:
    for t in range(nt):
        grid._next_distribution[1:-1, 1:-1] = numerical_scheme(grid)
        for bc_entry in bc:
            grid._next_distribution[bc_entry.positions] = bc_entry(grid)

        grid.swap_distributions()

    return grid.distribution
