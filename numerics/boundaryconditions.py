from enum import Enum
from typing import Literal, Tuple
import numpy as np

from numerics.grid import Grid, Index2D


class DirichletBoundaryCondition:
    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(
        self,
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        return self._value  # type: ignore


class Direction(Enum):
    NORTH: Index2D = (-2, 0)
    SOUTH: Index2D = (2, 0)
    WEST: Index2D = (0, -2)
    EAST: Index2D = (0, 2)


class NeumannBoundaryCondition:
    def __init__(self, value: float, direction: Direction) -> None:
        self._value = value
        self._direction = direction

    def __call__(
        self,
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        y, x = position
        offset_y, offset_x = self._direction.value
        grid_value = grid.get((y + offset_y, x + offset_x))
        sign = self._get_sign()
        grid_distance = self._get_relevant_grid_distance(grid.node_distances)

        return grid_value + 2 * sign * self._value * grid_distance

    def _get_relevant_grid_distance(self, node_distances: Tuple[float, float]) -> float:
        grid_distance: float
        if self._direction.value[0] != 0:
            grid_distance = node_distances[0]
        else:
            grid_distance = node_distances[1]
        return grid_distance

    def _get_sign(self) -> Literal[1, -1]:
        if self._direction.value[0] >= 0 and self._direction.value[1] >= 0:
            return -1

        return 1