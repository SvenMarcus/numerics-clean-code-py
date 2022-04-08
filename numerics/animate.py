from typing import Callable, Optional
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import numpy.typing as npt


def animate(simulation_runner: Callable[[int], npt.NDArray[np.float64]]) -> None:
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
