import numpy as np
from simulator.lattice import get_neighbors, create_lattice, coverage, grain_boundary_density
from simulator.grain_tracking import add_site

NU = 1e12

def compute_rates(grid, params, n=100):
    F     = params['F']
    E_d   = params['E_d']
    E_des = params['E_des']
    T     = params['T']
    kT    = 8.617e-5 * T

    k_diff = NU * np.exp(-E_d  / kT)
    k_des  = NU * np.exp(-E_des / kT)

    events = []
    rates  = []

    empty_sites    = list(zip(*np.where(grid == 0)))
    occupied_sites = list(zip(*np.where(grid == 1)))

    for (r, c) in empty_sites:
        events.append(('ads', r, c, None))
        rates.append(F)

    for (r, c) in occupied_sites:
        events.append(('des', r, c, None))
        rates.append(k_des)
        for (nr, nc) in get_neighbors(r, c, n):
            if grid[nr, nc] == 0:
                events.append(('diff', r, c, (nr, nc)))
                rates.append(k_diff)

    return events, np.array(rates, dtype=np.float64)


def execute_event(event, grid, n=100):
    etype, r, c, target = event
    if etype == 'ads':
        add_site(r, c, grid)
    elif etype == 'des':
        grid[r, c] = 0
    elif etype == 'diff':
        nr, nc = target
        grid[r, c] = 0
        add_site(nr, nc, grid) 


def run_kmc(params, max_time=10.0, max_steps=100000, n=100):
    grid = create_lattice(n)
    time = 0.0

    for step in range(max_steps):
        if time >= max_time:
            break
            
        events, rates = compute_rates(grid, params, n)
        R_total = rates.sum()
        if R_total == 0:
            break

        dt  = -np.log(np.random.rand()) / R_total
        idx = np.searchsorted(np.cumsum(rates), np.random.rand() * R_total)
        idx = min(idx, len(events) - 1)

        execute_event(events[idx], grid, n)
        time += dt

    cov = coverage(grid)
    gbd = grain_boundary_density(grid, n)
    return cov, gbd, time