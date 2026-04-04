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


def _pick_kth_true(mask_row, k):
    """Return column index of the k-th True in a 1D boolean row."""
    count = 0
    for col, is_true in enumerate(mask_row):
        if is_true:
            if count == k:
                return col
            count += 1
    return -1


def run_kmc(
    params,
    max_steps=5000000,
    n=100,
    seed=None,
    time_factor=5.0,
    target_coverage=None,
    max_diff_to_ads_ratio=120.0,
):

    F     = params['F']
    E_d   = params['E_d']
    E_des = params['E_des']
    T     = params['T']
    kT    = 8.617e-5 * T

    max_time = time_factor / F
    k_diff = NU * np.exp(-E_d  / kT)
    k_des  = NU * np.exp(-E_des / kT)

    # ------------------------------------------------------------------
    # FIX 1: Cap k_diff
    # Once diffusion length >> lattice size, faster diffusion changes
    # nothing about the morphology — the atom has already explored the
    # full lattice. Capping here removes redundant diffusion hops without
    # altering the physical outcome.
    # ------------------------------------------------------------------
    # Optional speed heuristic: cap diffusion so the event stream is not
    # completely dominated by diffusion hops. Set to None for uncapped kinetics.
    if max_diff_to_ads_ratio is not None:
        k_diff = min(k_diff, max_diff_to_ads_ratio * F)

    grid      = create_lattice(n)
    grain_ids = np.full(n * n, -1, dtype=np.int32)
    nb        = build_neighbor_table(n)
    rng       = np.random.default_rng(seed)
    reset_grain_counter()
    time = 0.0
    n_sites = n * n
    target_sites = None
    if target_coverage is not None:
        target_sites = int(np.ceil(target_coverage * n_sites))

    for _ in range(int(max_steps)):
        empty_idx    = np.where(grid == 0)[0]
        occupied_idx = np.where(grid == 1)[0]

        if len(empty_idx) == 0:
            break

        n_ads = len(empty_idx)
        n_occ = len(occupied_idx)

        if target_sites is not None and n_occ >= target_sites:
            break

        R_ads = n_ads * F
        R_des = n_occ * k_des

        neighbors  = nb[occupied_idx]#
        nb_safe    = np.where(neighbors >= 0, neighbors, 0)# 
        empty_nb   = (neighbors >= 0) & (grid[nb_safe] == 0)
        empty_nb_count = empty_nb.sum(axis=1)
        n_diff_edges = int(empty_nb_count.sum())
        R_diff = n_diff_edges * k_diff

        R_total = R_ads + R_des + R_diff
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
        event_pick = rng.random() * R_total

        if event_pick < R_ads:
            # adsorption
            site       = empty_idx[rng.integers(n_ads)]
            grid[site] = 1
            _assign_grain(site, grid, grain_ids, nb)

        elif event_pick < R_ads + R_des:
            # desorption
            site            = occupied_idx[rng.integers(n_occ)]
            grid[site]      = 0
            grain_ids[site] = -1

        else:
            # diffusion — sample one of all possible empty-neighbor hops
            if n_diff_edges == 0:
                time += dt
                if time >= max_time:
                    break
                continue

            hop_pick = rng.integers(n_diff_edges)
            csum = np.cumsum(empty_nb_count)
            src_pos = np.searchsorted(csum, hop_pick, side='right')
            src = occupied_idx[src_pos]

            prev = 0 if src_pos == 0 else csum[src_pos - 1]
            local_k = int(hop_pick - prev)
            nb_col = _pick_kth_true(empty_nb[src_pos], local_k)
            if nb_col < 0:
                time += dt
                if time >= max_time:
                    break
                continue

            dst = neighbors[src_pos, nb_col]
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