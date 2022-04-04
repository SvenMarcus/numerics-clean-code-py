from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Literal, Protocol, Tuple
import numpy as np
import numpy.typing as npt


Index2D = tuple[int, int]


@dataclass
class Grid:
    dimensions: Tuple[int, int]
    node_distances: Tuple[float, float]

    distribution: npt.NDArray[np.float64] = field(init=False)
    next_distribution: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.distribution = np.zeros(self.dimensions)
        self.next_distribution = np.zeros(self.dimensions)

    def __iter__(self) -> Iterator[Index2D]:
        ny, nx = self.dimensions
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                yield (y, x)

    def set_next(self, position: Index2D, value: np.float64) -> None:
        self.next_distribution[position[0], position[1]] = value

    def swap_distributions(self) -> None:
        self.distribution, self.next_distribution = (
            self.next_distribution,
            self.distribution,
        )


class NumericalFunction(Protocol):
    def __call__(
        self,
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        pass


BoundaryConditionMap = dict[Index2D, NumericalFunction]


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


class Simulation:
    def __init__(
        self,
        numerical_scheme: NumericalFunction,
        boundary_conditions: BoundaryConditionMap,
    ) -> None:
        self._numerical_scheme = numerical_scheme
        self._boundary_conditions = boundary_conditions

    def run(
        self,
        grid: Grid,
        number_of_timesteps: int,
    ) -> npt.NDArray[np.float64]:
        for t in range(number_of_timesteps):
            for current_position in grid:
                calculation_function = self._get_next_function(current_position)
                next_value = calculation_function(grid, current_position)
                grid.set_next(current_position, next_value)

            grid.swap_distributions()

        return grid.distribution

    def _get_next_function(
        self,
        current_position: Index2D,
    ) -> NumericalFunction:
        calculation_function = self._numerical_scheme
        if current_position in self._boundary_conditions:
            calculation_function = self._boundary_conditions[current_position]

        return calculation_function


class DirichletBoundaryCondition:
    def __init__(self, value: np.float64) -> None:
        self._value = value

    def __call__(
        self,
        grid: Grid,
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
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        y, x = position
        offset_y, offset_x = self._direction.value
        grid_value = grid.distribution[y + offset_y, x + offset_x]
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
