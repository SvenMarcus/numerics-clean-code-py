def ftcs(numerical_scheme, grid, nt, dy, dx, bc):
    for t in range(nt):
        grid.distribution[1:-1, 1:-1] = numerical_scheme(grid)
        for i, j, bc_entry in bc:
            grid.distribution[i, j] = bc_entry(grid, (i, j))

        grid.swap_distributions()

    return grid.distribution
