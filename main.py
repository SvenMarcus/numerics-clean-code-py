import matplotlib.pyplot as plt
import numpy as np

import numerics
from numerics.animate import animate


L = 10.0
W = 10.0
dx = 0.25
dy = 0.25
ny = int(L / dy)
nx = int(W / dx)

T0 = np.ndarray((ny, nx))
T1 = np.ndarray((ny, nx))

dt = 0.1

K = 0.111

bc = {}
for x in range(nx):
    bc[(1, x)] = {"t": "d", "v": 1.0}

    bc[(ny - 2, x)] = {"t": "d", "v": 0.0}


bc[(ny // 2, nx // 2)] = {"t": "n", "v": -0.5, "d": "S"}

animate(lambda nt: numerics.ftcs(T0, T1, nt, dt, ny, dy, nx, dy, K, bc))
