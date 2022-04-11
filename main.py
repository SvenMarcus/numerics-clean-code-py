from typing import Set

import numpy as np

import numerics
from numerics.animate import animate
from numerics.boundaryconditions import (
    Direction,
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
    Slice2D,
)
from numerics.grid import Grid
from numerics.heatequation import HeatEquation

L = 100.0
W = 100.0
dx = 0.25
dy = 0.25
ny = int(L / dy)
nx = int(W / dx)

T0 = np.zeros((ny, nx))
T1 = np.zeros((ny, nx))

nt = 10000
dt = 0.1

K = 0.111

diricht_bc_top = DirichletBoundaryCondition(1.0, Slice2D.horizontal(0))
diricht_bc_bot = DirichletBoundaryCondition(1.0, Slice2D.horizontal(ny - 1))

neumann_bc = NeumannBoundaryCondition(
    0.5, Direction.SOUTH, Slice2D.point(ny // 2, nx // 2)
)

bc: Set[numerics.BoundaryCondition] = {diricht_bc_top, diricht_bc_bot, neumann_bc}

heat_equation = HeatEquation(K, dt)
grid = Grid((ny, nx), (dy, dx))
simulation_runner = lambda steps: numerics.run_simulation(
    grid, heat_equation, bc, steps
)
animate(simulation_runner)
