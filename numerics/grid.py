import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Iterator, Tuple


Index2D = Tuple[int, int]


@dataclass
class Grid:
    dimensions: Tuple[int, int]
    node_distances: Tuple[float, float]

    distribution: np.ndarray = field(init=False)
    _next_distribution: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.distribution = np.zeros(self.dimensions)
        self._next_distribution = np.zeros(self.dimensions)

    def set_next(self, position: Index2D, value: np.float64) -> None:
        self._next_distribution[position[0], position[1]] = value

    def swap_distributions(self) -> None:
        self.distribution, self._next_distribution = (
            self._next_distribution,
            self.distribution,
        )
