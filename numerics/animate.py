from typing import Callable, Optional
from matplotlib.image import AxesImage  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import numpy as np
import numpy.typing as npt


def animate(simulation_runner: Callable[[int], np.ndarray]) -> None:
    fig, ax = plt.subplots()
    plot: Optional[AxesImage] = None

    def _animate(_: int) -> None:
        nonlocal plot
        next_values = simulation_runner(10)
        if plot is None:
            plot = ax.imshow(next_values, cmap="rainbow")
        else:
            plot.set(data=next_values, cmap="rainbow")

    _ = FuncAnimation(fig, _animate, interval=16)
    plt.show()
