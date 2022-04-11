from numerics import run_simulation
from numerics.animate import animate
from numerics.boundaryconditions import DirichletBoundaryCondition
from numerics.grid import Grid, Slice2D
from numerics.waveequation import WaveEquation


length = 10.0
dy = dx = 0.05
ny = nx = int(length // dy)

time_resolution = 0.005
propagation_velocity = 0.25

wave_equation = WaveEquation(propagation_velocity, time_resolution)
spike = DirichletBoundaryCondition(1.0, Slice2D.point(ny // 2, nx // 2))
boundary_conditions = {spike}

grid = Grid((ny, nx), (dy, dx))
simulation_runner = lambda nt: run_simulation(grid, wave_equation, boundary_conditions, nt)
animate(simulation_runner)