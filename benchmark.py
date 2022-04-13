from timeit import timeit

import numpy as np

from main import K, L, W, bc, dt, dx, dy, grid, heat_equation, nx, ny
from numerics import run_simulation
from numerics.original import ftcs

number_of_timesteps = 100_000
number_of_runs = 5
print(
    "Clean solution took:",
    timeit(
        lambda: run_simulation(grid, heat_equation, bc, number_of_timesteps),
        number=number_of_runs,
    )
    / number_of_runs,
)

T0 = np.zeros((ny, nx))
T1 = np.zeros((ny, nx))

_bc = []
_bc.append({"t": "d", "p": (slice(1, 2), slice(nx)), "v": 1.0})
_bc.append(
    {
        "t": "n",
        "p": (ny // 2, nx // 2),
        "v": 0.5,
        "d": "S",
    }
)

print(
    "Dirty solution took:",
    timeit(
        lambda: ftcs(T0, T1, number_of_timesteps, dt, ny, dy, nx, dx, K, _bc),
        number=number_of_runs,
    )
    / number_of_runs,
)
