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
    _next_distribution: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.distribution = np.zeros(self.dimensions)
        self._next_distribution = np.zeros(self.dimensions)

    def __iter__(self) -> Iterator[Index2D]:
        ny, nx = self.distribution.shape
        for position in np.ndindex(ny - 1, nx - 1):
            if 0 in position:
                continue

            yield position 

    def get(self, position: Index2D) -> np.float64:
        return self.distribution[position[0], position[1]]  # type: ignore

    def set_next(self, position: Index2D, value: np.float64) -> None:
        self._next_distribution[position[0], position[1]] = value

    def swap_distributions(self) -> None:
        self.distribution, self._next_distribution = (
            self._next_distribution,
            self.distribution,
        )
