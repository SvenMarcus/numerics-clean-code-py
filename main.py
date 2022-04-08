import matplotlib.pyplot as plt
import numpy as np

import numerics
from numerics.animate import animate


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

bc = {}
for x in range(nx):
    bc[(1, x)] = {
        "t": "d",
        "v": 1.0
    }

    bc[(ny - 2, x)] = {
        "t": "d",
        "v": 1.0
    }


bc[(ny // 2, nx // 2)] = {
    "t": "n",
    "v": .5,
    "d": "S"
}


simulation_runner = lambda steps: numerics.ftcs(T0, T1, steps, dt, dy, dx, K, bc)
animate(simulation_runner)
