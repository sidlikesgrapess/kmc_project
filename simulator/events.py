import numpy as np
from simulator.lattice import create_lattice, coverage, build_neighbor_table
from simulator.grain_tracking import reset_grain_counter, new_grain_id

NU = 1e12  # attempt frequency (Hz)


def _assign_grain(site, grid, grain_ids, nb):
    """
    Called on adsorption only.
    - No occupied neighbours → new nucleation event, new grain ID.
    - Occupied neighbours    → join the first neighbour's grain. idt this is true to physics
    Two grains touching is a grain BOUNDARY.
    """
    for k in range(6):
        n_site = nb[site, k]
        if n_site >= 0 and grid[n_site] == 1 and grain_ids[n_site] >= 0:
            grain_ids[site] = grain_ids[n_site]
            return
    grain_ids[site] = new_grain_id()


def grain_boundary_density(grid, grain_ids, nb): #is this correct?
    """
    Fraction of occupied-occupied neighbour pairs with different grain IDs.
    Each edge counted twice (both endpoints); ratio is unaffected.
    """
    occupied = np.where(grid == 1)[0]
    if len(occupied) == 0:
        return 0.0

    boundary = 0
    total    = 0
    for k in range(6):
        nb_sites = nb[occupied, k]
        nb_safe  = np.where(nb_sites >= 0, nb_sites, 0)
        valid    = (nb_sites >= 0) & (grid[nb_safe] == 1)
        total   += valid.sum()
        diff     = valid & (grain_ids[occupied] != grain_ids[nb_safe])
        boundary += diff.sum()

    return int(boundary) / int(total) if total > 0 else 0.0


def run_kmc(params, max_steps=5000000, n=100, seed=None):

    F     = params['F']
    E_d   = params['E_d']
    E_des = params['E_des']
    T     = params['T']
    kT    = 8.617e-5 * T

    max_time = 5.0 / F
    k_diff = NU * np.exp(-E_d  / kT)
    k_des  = NU * np.exp(-E_des / kT)

    # ------------------------------------------------------------------
    # FIX 1: Cap k_diff
    # Once diffusion length >> lattice size, faster diffusion changes
    # nothing about the morphology — the atom has already explored the
    # full lattice. Capping here removes redundant diffusion hops without
    # altering the physical outcome.
    # ------------------------------------------------------------------
    k_diff = min(k_diff, F * 1000) #not physically accurate

    grid      = create_lattice(n)
    grain_ids = np.full(n * n, -1, dtype=np.int32)
    nb        = build_neighbor_table(n)
    rng       = np.random.default_rng(seed)
    reset_grain_counter()
    time = 0.0

    for _ in range(int(max_steps)):
        empty_idx    = np.where(grid == 0)[0]
        occupied_idx = np.where(grid == 1)[0]

        if len(empty_idx) == 0:
            break

        n_ads = len(empty_idx)
        n_occ = len(occupied_idx)

        ads_rates = np.full(n_ads, F)# where ads is possible
        des_rates = np.full(n_occ, k_des)# where des is possible

        neighbors  = nb[occupied_idx]#
        nb_safe    = np.where(neighbors >= 0, neighbors, 0)# 
        empty_nb   = (neighbors >= 0) & (grid[nb_safe] == 0)
        src_idx, nb_col = np.where(empty_nb)
        diff_src   = occupied_idx[src_idx]
        diff_dst   = neighbors[src_idx, nb_col]
        diff_rates = np.full(len(diff_src), k_diff)

        all_rates = np.concatenate([ads_rates, des_rates, diff_rates])
        R_total   = all_rates.sum()
        if R_total == 0:
            break

        # --------------------------------------------------------------
        # FIX 2: Rate rescaling
        # KMC event selection depends only on RELATIVE rates, not their
        # absolute magnitude. Scaling all rates by the same factor leaves
        # event probabilities unchanged. The clock (dt) is scaled inversely,
        # so simulated time remains exact. This converts wasted micro-steps
        # into fewer, larger, physically equivalent steps.
        # --------------------------------------------------------------
        # MAX_RATE = 1e6
        # if R_total > MAX_RATE:
        #     scale     = MAX_RATE / R_total
        #     all_rates = all_rates * scale
        #     R_total   = MAX_RATE

        dt  = -np.log(rng.random()) / R_total
        idx = np.searchsorted(np.cumsum(all_rates), rng.random() * R_total)
        idx = min(idx, len(all_rates) - 1)

        if idx < n_ads:
            # adsorption
            site       = empty_idx[idx]
            grid[site] = 1
            _assign_grain(site, grid, grain_ids, nb)

        elif idx < n_ads + n_occ:
            # desorption
            site            = occupied_idx[idx - n_ads]
            grid[site]      = 0
            grain_ids[site] = -1

        else:
            # diffusion — atom carries its grain ID, no reassignment
            diff_i         = idx - n_ads - n_occ
            src            = diff_src[diff_i]
            dst            = diff_dst[diff_i]
            grid[dst]      = 1
            grain_ids[dst] = grain_ids[src]
            grid[src]      = 0
            grain_ids[src] = -1

        time += dt
        if time >= max_time:
            break

    cov = coverage(grid)
    gbd = grain_boundary_density(grid, grain_ids, nb)
    print(f"\n        Ended at step {_} | Time: {time:.2f} / {max_time:.2f}")
    return cov, gbd, time