import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Tuple, Type


Index2D = Tuple[int, int]


class Slice2D(tuple[slice, slice]):
    @classmethod
    def point(cls: Type["Slice2D"], y: int, x: int) -> "Slice2D":
        return cls(y_start=y, y_end=y + 1, x_start=x, x_end=x + 1)

    @classmethod
    def horizontal(cls: Type["Slice2D"], y: int, x_start: int = 0, x_end: int = -1) -> "Slice2D":
        return cls(
            y_start=y,
            y_end=y+1,
            x_start=x_start,
            x_end=x_end
        )

    def __new__(
        cls: Type["Slice2D"],
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        y_step: int = 1,
        x_step: int = 1,
    ) -> "Slice2D":
        return super(Slice2D, cls).__new__(
            cls, (slice(y_start, y_end, y_step), slice(x_start, x_end, x_step))  # type: ignore
        )

    def shift(self, y: int, x: int) -> "Slice2D":
        y_slice = self[0]
        x_slice = self[1]
        return Slice2D(
            y_start=y_slice.start + y,
            y_end=y_slice.stop + y,
            y_step=y_slice.step,
            x_start=x_slice.start + x,
            x_end=x_slice.stop + x,
            x_step=x_slice.step,
        )


@dataclass
class Grid:
    dimensions: Tuple[int, int]
    node_distances: Tuple[float, float]

    distribution: np.ndarray = field(init=False)
    next_distribution: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.distribution = np.zeros(self.dimensions)
        self.next_distribution = np.zeros(self.dimensions)

    def swap_distributions(self) -> None:
        self.distribution, self.next_distribution = (
            self.next_distribution,
            self.distribution,
        )
