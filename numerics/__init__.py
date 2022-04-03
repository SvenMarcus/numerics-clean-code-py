def ftcs(T0, T1, nt, dt, ny, dy, nx, dx, k, bc):
    for t in range(nt):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if (i, j) not in bc:
                    T1[i, j] = (
                        T0[i, j]
                        + (
                            (T0[i + 1, j] - 2 * T0[i, j] + T0[i - 1, j]) / (dx**2)
                            + (T0[i, j + 1] - 2 * T0[i, j] + T0[i, j - 1]) / (dy**2)
                        )
                        * dt
                        * k
                    )
                else:
                    if bc[(i, j)]["t"] == "d":  # dirichlet
                        T1[i, j] = bc[(i, j)]["v"]
                    elif bc[(i, j)]["t"] == "n":  # neumann
                        gd = bc[(i, j)]["d"]  # gradient direction
                        v = bc[(i, j)]["v"]
                        grid_val: float
                        sign: int
                        grid_distance: float
                        if gd == "N":
                            grid_val = T0[i - 2, j]
                            sign = 1
                            grid_distance = dy
                        elif gd == "S":
                            grid_val = T0[i + 2, j]
                            sign = -1
                            grid_distance = dy
                        elif gd == "W":
                            grid_val = T0[i, j - 2]
                            sign = 1
                            grid_distance = dx
                        elif gd == "E":
                            grid_val = T0[i, j + 2]
                            sign = -1
                            grid_distance = dx

                        T1[i, j] = grid_val + 2 * sign * v * grid_distance
        T0, T1 = T1, T0

    return T0
