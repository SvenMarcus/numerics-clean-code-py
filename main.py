import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt

import numerics


L = 10.0
W = 10.0
dx = 0.25
dy = 0.25
ny = int(L / dy)
nx = int(W / dx)

T0: npt.NDArray[np.float64] = np.ndarray((ny, nx))
T1: npt.NDArray[np.float64] = np.ndarray((ny, nx))

nt = 1000
dt = 0.1

K = 0.111

bc: dict[tuple[int, int], numerics.BoundaryCondition] = {}
for x in range(nx):
    bc[(1, x)] = {
        "t": "d",
        "v": 1.0,
    }

    bc[(ny - 2, x)] = {
        "t": "d",
        "v": 0.0,
    }


bc[(ny // 2, nx // 2)] = {
    "t": "n",
    "v": -0.5,
    "d": "S",
}


T0 = numerics.ftcs(T0, T1, nt, dt, ny, dy, nx, dy, K, bc)

plt.imshow(T0, cmap="plasma", interpolation="nearest")
plt.savefig("result.png")
