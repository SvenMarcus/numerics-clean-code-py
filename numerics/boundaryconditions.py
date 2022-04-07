from enum import Enum
from typing import Literal, Tuple
from numba import float64, njit
from numba.experimental import jitclass

import numpy as np

from numerics.grid import Grid, Index2D
from numerics.numerical import NumericalFunction


def dirichlet_boundary(value: np.float64) -> NumericalFunction:
    @njit  # type: ignore
    def _dirichlet_bc(grid: Grid, position: Index2D) -> np.float64:
        return value

    return _dirichlet_bc


class Direction:
    NORTH: Index2D = (-2, 0)
    SOUTH: Index2D = (2, 0)
    WEST: Index2D = (0, -2)
    EAST: Index2D = (0, 2)


def neumann_boundary(value: float64, direction: Index2D) -> NumericalFunction:
    @njit  # type: ignore
    def _neumann_boundary(
        grid: Grid,
        position: Index2D,
    ) -> float64:
        def _get_relevant_grid_distance(node_distances: Tuple[float, float]) -> float:
            grid_distance: float
            if direction[0] != 0:
                grid_distance = node_distances[0]
            else:
                grid_distance = node_distances[1]
            return grid_distance

        def _get_sign(direction: Index2D) -> Literal[1, -1]:
            if direction[0] >= 0 and direction[1] >= 0:
                return -1

            return 1

        y, x = position
        offset_y, offset_x = direction
        grid_value = grid.get((y + offset_y, x + offset_x))
        sign = _get_sign(direction)
        grid_distance = _get_relevant_grid_distance(grid.node_distances)

        return grid_value + 2 * sign * value * grid_distance

    return _neumann_boundary
