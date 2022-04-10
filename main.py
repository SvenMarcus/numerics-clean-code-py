import numpy as np

import numerics
from numerics.animate import animate
from numerics.boundaryconditions import Direction, DirichletBoundaryCondition, NeumannBoundaryCondition
from numerics.grid import Grid
from numerics.heatequation import HeatEquation


L = 50.0
W = 50.0
dx = .25
dy = .25
ny = int(L / dy)
nx = int(W / dx)

T0 = np.zeros((ny, nx))
T1 = np.zeros((ny, nx))

nt = 10000
dt = .1

K = .111

bc = set()
diricht_bc = DirichletBoundaryCondition(1.0)
for x in range(nx):
    bc.add((0, x, diricht_bc))
    bc.add((ny - 1, x, diricht_bc))

neumann_bc = NeumannBoundaryCondition(.5, Direction.SOUTH)
bc.add((ny // 2, nx // 2, neumann_bc))

heat_equation = HeatEquation(K, dt)
grid = Grid((ny, nx), (dy, dx))
simulation_runner = lambda steps: numerics.ftcs(heat_equation, grid, steps, dy, dx, bc)
animate(simulation_runner)
