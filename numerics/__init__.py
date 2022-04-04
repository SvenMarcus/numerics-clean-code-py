from typing import Literal, TypedDict
import numpy as np
import numpy.typing as npt


class BoundaryCondition(TypedDict, total=False):
    t: Literal["d", "n"]
    v: float
    d: Literal["N", "W", "S", "E"]


Index2D = tuple[int, int]
BoundaryConditionMap = dict[Index2D, BoundaryCondition]


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
                    next_distribution[i, j] = (
                        distribution[i, j]
                        + (
                            (
                                distribution[i + 1, j]
                                - 2 * distribution[i, j]
                                + distribution[i - 1, j]
                            )
                            / (node_distance_in_x**2)
                            + (
                                distribution[i, j + 1]
                                - 2 * distribution[i, j]
                                + distribution[i, j - 1]
                            )
                            / (node_distance_in_y**2)
                        )
                        * timestep_delta
                        * thermal_diffusivity
                    )
                else:
                    if boundary_conditions[(i, j)]["t"] == "d":  # dirichlet
                        next_distribution[i, j] = boundary_conditions[(i, j)]["v"]
                    elif boundary_conditions[(i, j)]["t"] == "n":  # neumann
                        gd = boundary_conditions[(i, j)]["d"]  # gradient direction
                        v = boundary_conditions[(i, j)]["v"]
                        grid_val: float
                        sign: int
                        grid_distance: float
                        if gd == "N":
                            grid_val = distribution[i - 2, j]
                            sign = 1
                            grid_distance = node_distance_in_y
                        elif gd == "S":
                            grid_val = distribution[i + 2, j]
                            sign = -1
                            grid_distance = node_distance_in_y
                        elif gd == "W":
                            grid_val = distribution[i, j - 2]
                            sign = 1
                            grid_distance = node_distance_in_x
                        elif gd == "E":
                            grid_val = distribution[i, j + 2]
                            sign = -1
                            grid_distance = node_distance_in_x

                        next_distribution[i, j] = (
                            grid_val + 2 * sign * v * grid_distance
                        )
        distribution, next_distribution = next_distribution, distribution

    return distribution
