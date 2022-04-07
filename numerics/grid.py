from numba import float64
from numba.experimental import jitclass
import numpy as np
import numpy.typing as npt
from typing import Iterator, Tuple


Index2D = Tuple[int, int]


@jitclass([ ("distribution", float64[:,:]), ("_next_distribution", float64[:,:]) ])
class Grid:
    dimensions: Tuple[int, int]
    node_distances: Tuple[float, float]

    distribution: np.ndarray
    _next_distribution: np.ndarray

    def __init__(self, dimensions: Tuple[int, int], node_distances: Tuple[float, float]) -> None:
        self.dimensions = dimensions
        self.node_distances = node_distances
        self.distribution = np.zeros(self.dimensions)
        self._next_distribution = np.zeros(self.dimensions)

    def iter_index(self) -> Iterator[Index2D]:
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
