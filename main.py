from typing import cast
from matplotlib import pyplot as plt
from numba.typed import Dict
import numpy as np
# from animate import animate

import numerics


L = 10.0
W = 10.0
dx = 0.25
dy = 0.25
ny = int(L / dy)
nx = int(W / dx)

nt = 1000
dt = 0.1

K = 0.111

assert K * dt / (dx**2) <= 0.25

bc: numerics.BoundaryConditionMap = Dict()
dirichlet_top = numerics.dirichlet_boundary(cast(np.float64, 1.0))
for x in range(nx):
    bc[(1, x)] = dirichlet_top
    bc[(ny - 2, x)] = dirichlet_top

bc[(ny // 2, nx // 2)] = numerics.neumann_boundary(
    cast(np.float64, 0.5), numerics.Direction.SOUTH
)

heat_equation = numerics.HeatEquation(K, dt)
# simulation = numerics.Simulation(heat_equation, bc)
grid = numerics.Grid((ny, nx), (dy, dx))

# animate(simulation, grid)
dist = numerics.run(heat_equation, bc, grid, nt)
plt.imsave("result.png", dist, cmap="rainbow")
