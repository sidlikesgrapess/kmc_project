import numpy as np
from simulator.lattice import create_lattice, coverage, build_neighbor_table
from simulator.grain_tracking import reset_grain_counter, new_grain_id

NU = 1e12  # attempt frequency (Hz)

def _assign_grain(site, grid, grain_ids, nb):
    for k in range(6):
        n_site = nb[site, k]
        if n_site >= 0 and grid[n_site] == 1 and grain_ids[n_site] >= 0:
            grain_ids[site] = grain_ids[n_site]
            return
    grain_ids[site] = new_grain_id()

def grain_boundary_density(grid, grain_ids, nb):
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
    immobile_if_neighbors_ge=2,
    snapshot_every_steps=None,
    snapshot_callback=None
):
    F     = params['F']
    E_d   = params['E_d']
    E_des = params['E_des']
    T     = params['T']
    kT    = 8.617e-5 * T

    max_time = time_factor / F
    k_diff = NU * np.exp(-E_d  / kT)
    k_des  = NU * np.exp(-E_des / kT)

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

    for step in range(int(max_steps)):
        if snapshot_callback and snapshot_every_steps:
            if step % snapshot_every_steps == 0:
                snapshot_callback(step, time, grid, grain_ids, n)

        empty_idx    = np.where(grid == 0)[0]
        occupied_idx = np.where(grid == 1)[0]

        if len(empty_idx) == 0:
            break

        n_ads = len(empty_idx)
        R_ads = n_ads * F

        if target_sites is not None and len(occupied_idx) >= target_sites:
            break

        neighbors  = nb[occupied_idx]
        nb_safe    = np.where(neighbors >= 0, neighbors, 0)
        
        occupied_nb_mask = (neighbors >= 0) & (grid[nb_safe] == 1)
        occupied_nb_count = occupied_nb_mask.sum(axis=1)
        
        # Stability rule here
        mobile_mask = occupied_nb_count < immobile_if_neighbors_ge
        
        n_desorb = int(mobile_mask.sum())
        R_des = n_desorb * k_des

        empty_nb   = (neighbors >= 0) & (grid[nb_safe] == 0)
        diff_nb_mask = empty_nb & mobile_mask[:, None]
        diff_nb_count = diff_nb_mask.sum(axis=1)
        n_diff_edges = int(diff_nb_count.sum())
        R_diff = n_diff_edges * k_diff

        R_total = R_ads + R_des + R_diff
        if R_total == 0:
            break

        dt  = -np.log(rng.random()) / R_total
        event_pick = rng.random() * R_total

        if event_pick < R_ads:
            site       = empty_idx[rng.integers(n_ads)]
            grid[site] = 1
            _assign_grain(site, grid, grain_ids, nb)
        elif event_pick < R_ads + R_des:
            mobile_sites = occupied_idx[mobile_mask]
            site            = mobile_sites[rng.integers(n_desorb)]
            grid[site]      = 0
            grain_ids[site] = -1
        else:
            if n_diff_edges == 0:
                time += dt
                if time >= max_time: break
                continue

            hop_pick = rng.integers(n_diff_edges)
            csum = np.cumsum(diff_nb_count)
            src_pos = np.searchsorted(csum, hop_pick, side='right')
            src = occupied_idx[src_pos]

            prev = 0 if src_pos == 0 else csum[src_pos - 1]
            local_k = int(hop_pick - prev)
            nb_col = _pick_kth_true(diff_nb_mask[src_pos], local_k)
            if nb_col < 0:
                time += dt
                if time >= max_time: break
                continue

            dst = neighbors[src_pos, nb_col]
            grid[dst]      = 1
            grain_ids[dst] = grain_ids[src]
            grid[src]      = 0
            grain_ids[src] = -1

        time += dt
        if time >= max_time:
            break

    if snapshot_callback and snapshot_every_steps:
        snapshot_callback(step, time, grid, grain_ids, n)

    cov = coverage(grid)
    gbd = grain_boundary_density(grid, grain_ids, nb)
    if step > 0:
        print(f"\n        Ended at step {step} | Time: {time:.2f} / {max_time:.2f}")
    return cov, gbd, time

