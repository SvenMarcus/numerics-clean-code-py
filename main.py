import matplotlib.pyplot as plt
import numpy as np

import numerics
from numerics.animate import animate


L = 50.0
W = 50.0
dx = 0.25
dy = 0.25
ny = int(L / dy)
nx = int(W / dx)

T0 = np.zeros((ny, nx))
T1 = np.zeros((ny, nx))

dt = 0.1

K = 0.111

# BC Explanations
# t: Type           d or n
# p: posistions     tuple of slices (y, x)
# v: value          value of bc
# d: direction      direction of bc (only neumann)

bc = []
bc.append({"t": "d", "p": (slice(1, 2), slice(nx)), "v": 1.0})
bc.append(
    {
        "t": "n",
        "p": (ny // 2, nx // 2),
        "v": 0.5,
        "d": "S",
    }
)

animate(lambda nt: numerics.ftcs(T0, T1, nt, dt, ny, dy, nx, dy, K, bc))
