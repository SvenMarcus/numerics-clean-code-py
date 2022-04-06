from typing import Optional
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from numerics import Grid, Simulation


def animate(simulation: Simulation, grid: Grid) -> None:
    fig, ax = plt.subplots()
    plot: Optional[AxesImage] = None
    def _animate(_: int) -> None:
        nonlocal plot
        distribution = simulation.run(grid, 10)
        if plot is None:
            plot = ax.imshow(distribution, cmap="rainbow", interpolation="nearest")
        else:
            plot.set(data=distribution, cmap="rainbow", interpolation="nearest")

    _ = FuncAnimation(fig, _animate, interval=20)
    plt.show()
    