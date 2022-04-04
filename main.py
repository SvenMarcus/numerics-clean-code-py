from typing import cast
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

T0: npt.NDArray[np.float64] = np.zeros((ny, nx))
T1: npt.NDArray[np.float64] = np.zeros((ny, nx))

nt = 1000
dt = 0.1

K = 0.111

assert K * dt / (dx**2) <= 0.25

bc: numerics.BoundaryConditionMap = {}
dirichlet_top = numerics.DirichletBoundaryCondition(cast(np.float64, 1.0))
for x in range(nx):
    bc[(1, x)] = dirichlet_top
    bc[(ny - 2, x)] = dirichlet_top

bc[(ny // 2, nx // 2)] = numerics.NeumannBoundaryCondition(
    cast(np.float64, 0.5), numerics.Direction.SOUTH
)


heat_equation = numerics.HeatEquation(K, dt)
T0 = numerics.ftcs(T0, T1, nt, (ny, nx), (dy, dx), heat_equation, bc)

plt.imshow(T0, cmap="plasma", interpolation="nearest")
plt.savefig("result.png")
