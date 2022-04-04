from typing import Literal, TypedDict
import numpy as np
import numpy.typing as npt


class BoundaryCondition(TypedDict, total=False):
    t: Literal["d", "n"]
    v: float
    d: Literal["N", "W", "S", "E"]


Index2D = tuple[int, int]
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
                if (i, j) not in boundary_conditions:
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
                    next_distribution[i, j] = apply_boundary_condition(
                        distribution,
                        node_distance_in_y,
                        node_distance_in_x,
                        boundary_conditions,
                        i,
                        j,
                    )

        distribution, next_distribution = next_distribution, distribution

    return distribution


def apply_boundary_condition(
    distribution: npt.NDArray[np.float64],
    node_distance_in_y: float,
    node_distance_in_x: float,
    boundary_conditions: BoundaryConditionMap,
    i: int,
    j: int,
) -> np.float64:
    bc_data = boundary_conditions[(i, j)]
    bc_type = bc_data["t"]
    bc_value = boundary_conditions[(i, j)]["v"]
    next_value: np.float64
    if bc_type == "d":  # dirichlet
        next_value = bc_value
    elif bc_type == "n":  # neumann
        direction = bc_data["d"]
        bc_value = boundary_conditions[(i, j)]["v"]
        grid_value: float
        sign: int
        grid_distance: float
        if direction == "N":
            grid_value = distribution[i - 2, j]
            sign = 1
            grid_distance = node_distance_in_y
        elif direction == "S":
            grid_value = distribution[i + 2, j]
            sign = -1
            grid_distance = node_distance_in_y
        elif direction == "W":
            grid_value = distribution[i, j - 2]
            sign = 1
            grid_distance = node_distance_in_x
        elif direction == "E":
            grid_value = distribution[i, j + 2]
            sign = -1
            grid_distance = node_distance_in_x

        next_value = grid_value + 2 * sign * bc_value * grid_distance

    return next_value
