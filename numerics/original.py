def ftcs(T0, T1, nt, dt, ny, dy, nx, dx, k, bc):
    for t in range(nt):
        T1[1:-1, 1:-1] = T0[1:-1, 1:-1] + k * dt * (
            (T0[2:, 1:-1] - 2 * T0[1:-1, 1:-1] + T0[:-2, 1:-1]) / (dx**2)
            + (T0[1:-1, 2:] - 2 * T0[1:-1, 1:-1] + T0[1:-1, :-2]) / (dy**2)
        )
        for _bc in bc:
            i, j = _bc["p"]
            if _bc["t"] == "d":  # dirichlet
                T1[i, j] = _bc["v"]
            elif _bc["t"] == "n":  # neumann
                gd = _bc["d"]  # gradient direction
                v = _bc["v"]
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