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
                        val: float
                        dist: float
                        if gd == "N":
                            val = T0[i - 2, j]
                            s = 1
                            dist = dy
                        elif gd == "S":
                            val = T0[i + 2, j]
                            s = -1
                            dist = dy
                        elif gd == "W":
                            val = T0[i, j - 2]
                            s = 1
                            dist = dx
                        elif gd == "E":
                            val = T0[i, j + 2]
                            s = -1
                            dist = dx

                        T1[i, j] = val + 2 * s * v * dist

        T0, T1 = T1, T0

    return T0
