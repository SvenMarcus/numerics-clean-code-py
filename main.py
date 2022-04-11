import numpy as np

import numerics
from numerics.animate import animate
from numerics.boundaryconditions import (
    Direction,
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
)
from numerics.grid import Grid
from numerics.heatequation import HeatEquation


L = 50.0
W = 50.0
dx = 0.25
dy = 0.25
ny = int(L / dy)
nx = int(W / dx)

T0 = np.zeros((ny, nx))
T1 = np.zeros((ny, nx))

nt = 10000
dt = 0.1

K = 0.111

bc = set()
diricht_bc_top = DirichletBoundaryCondition(1.0, (slice(0, 1), slice(0, nx)))
diricht_bc_bot = DirichletBoundaryCondition(1.0, (slice(ny - 1, ny), slice(0, nx)))
for x in range(nx):
    bc.add(diricht_bc_top)
    bc.add(diricht_bc_bot)

neumann_bc = NeumannBoundaryCondition(
    0.5, Direction.SOUTH, (slice(ny // 2, ny // 2 + 1), slice(nx // 2, nx // 2 + 1))
)
bc.add(neumann_bc)

heat_equation = HeatEquation(K, dt)
grid = Grid((ny, nx), (dy, dx))
simulation_runner = lambda steps: numerics.ftcs(heat_equation, grid, steps, dy, dx, bc)
animate(simulation_runner)
