from typing import Protocol
import numpy as np

from numerics.grid import Grid, Index2D


class NumericalFunction(Protocol):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        grid: Grid,
        position: Index2D,
    ) -> np.float64:
        pass
