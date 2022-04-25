from timeit import timeit
from typing import Any, Callable

import numpy as np

from main import K, L, W, bc, dt, dx, dy, grid, heat_equation, nx, ny
from numerics import run_simulation
from numerics.original import ftcs

NUMBER_OF_RUNS = 5


def benchmark_average_runtime(func: Callable[[], Any]) -> float:
    return timeit(func, number=NUMBER_OF_RUNS) / NUMBER_OF_RUNS


number_of_timesteps = 100_000
clean_runtime = benchmark_average_runtime(
    lambda: run_simulation(grid, heat_equation, bc, number_of_timesteps)
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


dirty_runtime = benchmark_average_runtime(
    lambda: ftcs(T0, T1, number_of_timesteps, dt, ny, dy, nx, dx, K, _bc)
)

print("Clean solution took:", clean_runtime)
print("Dirty solution took:", dirty_runtime)
print("Ratio of clean / dirty is", clean_runtime / dirty_runtime)
