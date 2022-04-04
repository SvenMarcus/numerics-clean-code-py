import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Iterator, Tuple


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
