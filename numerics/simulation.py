from typing import Protocol
import numpy as np
import numpy.typing as npt

from numerics.grid import Grid, Index2D


class NumericalFunction(Protocol):
    def __call__(
        self,
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        pass


BoundaryConditionMap = dict[Index2D, NumericalFunction]


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
