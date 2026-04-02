import numpy as np
from joblib import Parallel, delayed, parallel_backend
from simulator.events import run_kmc

# BOUNDS = {
#     'F'    : (0.1, 1.0),
#     'E_d'  : (0.3,  0.8),
#     'E_des': (1.4,  2.0),  # raised further
#     'T'    : (200,  400),  # tightened further
# }

BOUNDS = {
    'F'    : (0.01, 1.0),   # ML/s, typical MBE range
    'E_d'  : (0.1,  0.8),   # eV, metal-on-metal diffusion barriers
    'E_des': (1.0,  3.0),   # eV, metallic binding energies
    'T'    : (200,  800),   # K, cryogenic to near-melting
}


def latin_hypercube_sample(n_samples, bounds, seed=42):
    rng    = np.random.default_rng(seed)
    n_dims = len(bounds)
    result = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        perm        = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples
    keys   = list(bounds.keys())
    params = []
    for j in range(n_samples):
        p = {}
        for i, k in enumerate(keys):
            lo, hi = bounds[k]
            p[k]   = lo + result[j, i] * (hi - lo)
        params.append(p)
    return params, keys


# Change n to 100 and raise max_steps
def run_one(i, params, max_steps=5000000, n=100): 
    print(f"Sim {i+1:>3}/200 | F={params['F']:.4f} E_d={params['E_d']:.2f} " 
          f"E_des={params['E_des']:.2f} T={params['T']:.0f}")
    
    cov, gbd, t = run_kmc(params, max_steps=max_steps, n=n, seed=i)
    return cov, gbd, t


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)

    param_list, keys = latin_hypercube_sample(10, BOUNDS)

    with parallel_backend('loky'):
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_one)(i, p) for i, p in enumerate(param_list)
        )

    coverages = np.array([r[0] for r in results])
    gbds      = np.array([r[1] for r in results])
    times     = np.array([r[2] for r in results])

    X = np.array([[p[k] for k in keys] for p in param_list])

    np.savez('data/dataset.npz',
             X=X, coverages=coverages, gbds=gbds, times=times,
             param_keys=keys)

    print(f"\n      Done. Coverage range : {coverages.min():.3f} – {coverages.max():.3f}")
    print(f"        GBD range      : {gbds.min():.3f} – {gbds.max():.3f}")