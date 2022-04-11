def ftcs(numerical_scheme, grid, nt, dy, dx, bc):
    for t in range(nt):
        grid.distribution[1:-1, 1:-1] = numerical_scheme(grid)
        for bc_entry in bc:
            y, x = bc_entry.positions
            grid.distribution[y, x] = bc_entry(grid)

        grid.swap_distributions()

    return grid.distribution
