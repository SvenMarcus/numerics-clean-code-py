from enum import Enum
from typing import Literal, Protocol, Tuple
import numpy as np
import numpy.typing as npt


Index2D = tuple[int, int]


class BoundaryCondition(Protocol):
    def __call__(
        self,
        distribution: npt.NDArray[np.float64],
        node_distances: Tuple[float, float],
        position: Index2D,
    ) -> np.float64:
        pass


BoundaryConditionMap = dict[Index2D, BoundaryCondition]


def apply_heat_equation(
    distribution: npt.NDArray[np.float64],
    node_distance_in_y: float,
    node_distance_in_x: float,
    timestep_delta: float,
    thermal_diffusivity: float,
    y: int,
    x: int,
) -> np.float64:
    return (
        distribution[y, x]
        + (
            (distribution[y + 1, x] - 2 * distribution[y, x] + distribution[y - 1, x])
            / (node_distance_in_x**2)
            + (distribution[y, x + 1] - 2 * distribution[y, x] + distribution[y, x - 1])
            / (node_distance_in_y**2)
        )
        * timestep_delta
        * thermal_diffusivity
    )


def ftcs(
    distribution: npt.NDArray[np.float64],
    next_distribution: npt.NDArray[np.float64],
    number_of_timesteps: int,
    timestep_delta: float,
    nodes_in_y: int,
    node_distance_in_y: float,
    nodes_in_x: int,
    node_distance_in_x: float,
    thermal_diffusivity: float,
    boundary_conditions: BoundaryConditionMap,
) -> npt.NDArray[np.float64]:
    for t in range(number_of_timesteps):
        for i in range(1, nodes_in_y - 1):
            for j in range(1, nodes_in_x - 1):
                current_position = (i, j)
                if current_position not in boundary_conditions:
                    next_distribution[i, j] = apply_heat_equation(
                        distribution,
                        node_distance_in_y,
                        node_distance_in_x,
                        timestep_delta,
                        thermal_diffusivity,
                        i,
                        j,
                    )
                else:
                    boundary_condition = boundary_conditions[current_position]
                    next_distribution[i, j] = boundary_condition(
                        distribution,
                        (node_distance_in_y, node_distance_in_x),
                        (i, j),
                    )

        distribution, next_distribution = next_distribution, distribution

    return distribution


class DirichletBoundaryCondition:
    def __init__(self, value: np.float64) -> None:
        self._value = value

    def __call__(
        self,
        distribution: npt.NDArray[np.float64],
        node_distances: Tuple[float, float],
        position: Index2D,
    ) -> np.float64:
        return self._value


class Direction(Enum):
    NORTH: Index2D = (-2, 0)
    SOUTH: Index2D = (2, 0)
    WEST: Index2D = (0, -2)
    EAST: Index2D = (0, 2)


class NeumannBoundaryCondition:
    def __init__(self, value: np.float64, direction: Direction) -> None:
        self._value = value
        self._direction = direction

    def __call__(
        self,
        distribution: npt.NDArray[np.float64],
        node_distances: Tuple[float, float],
        position: Index2D,
    ) -> np.float64:
        y, x = position
        offset_y, offset_x = self._direction.value
        grid_value = distribution[y + offset_y, x + offset_x]
        sign = self._get_sign()
        grid_distance = self._get_relevant_grid_distance(node_distances)

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
