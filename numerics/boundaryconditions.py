from enum import Enum
from typing import Literal, Tuple
import numpy as np

from numerics.grid import Grid, Index2D, Slice2D


class DirichletBoundaryCondition:
    def __init__(self, value: float, positions: Slice2D) -> None:
        self._value = np.array((value,), dtype=np.float64)
        self.positions = positions

    def __call__(
        self,
        grid: "Grid",
    ) -> np.ndarray:
        return self._value


class Direction(Enum):
    NORTH: "Index2D" = (-2, 0)
    SOUTH: "Index2D" = (2, 0)
    WEST: "Index2D" = (0, -2)
    EAST: "Index2D" = (0, 2)


class NeumannBoundaryCondition:
    def __init__(self, value: float, direction: Direction, positions: Slice2D) -> None:
        self._value = value
        self._direction = direction
        self.positions = positions
        offset_y, offset_x = self._direction.value
        self._shifted_positions = positions.shift(offset_y, offset_x)

    def __call__(
        self,
        grid: Grid,
    ) -> np.ndarray:
        grid_values = grid.distribution[self._shifted_positions]
        sign = self._get_sign()
        grid_distance = self._get_relevant_grid_distance(grid.node_distances)

        return grid_values + 2 * sign * self._value * grid_distance

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
