import numpy as np
from joblib import Parallel, delayed
from simulator.events import run_kmc

# Parameter bounds
# F     : adsorption flux (site^-1 time^-1)
# E_d   : diffusion barrier (eV)
# E_des : desorption barrier (eV)
# T     : temperature (K)

BOUNDS = {
    'F'    : (1e-4, 1e-1),
    'E_d'  : (0.1,  0.8),
    'E_des': (0.5,  1.5),
    'T'    : (200,  800),
}

def latin_hypercube_sample(n_samples, bounds, seed=42):
    """LHS over 4D parameter space — better coverage than random grid."""
    rng    = np.random.default_rng(seed)
    n_dims = len(bounds)
    result = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        perm        = rng.permutation(n_samples)
        result[:,i] = (perm + rng.random(n_samples)) / n_samples
    # scale to bounds
    keys   = list(bounds.keys())
    params = []
    for j in range(n_samples):
        p = {}
        for i, k in enumerate(keys):
            lo, hi = bounds[k]
            p[k]   = lo + result[j, i] * (hi - lo)
        params.append(p)
    return params, keys

def run_one(i, params, max_time=10.0, max_steps=100000, n=100):
    print(f"Sim {i+1}/200 | F={params['F']:.4f} E_d={params['E_d']:.2f} "
          f"E_des={params['E_des']:.2f} T={params['T']:.0f}")
    cov, gbd, t = run_kmc(params, max_time=max_time, max_steps=max_steps, n=n)
    return cov, gbd, t

if __name__ == '__main__':
    param_list, keys = latin_hypercube_sample(10, BOUNDS)

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(i, p) for i, p in enumerate(param_list)
    )

    coverages = np.array([r[0] for r in results])
    gbds      = np.array([r[1] for r in results])
    times     = np.array([r[2] for r in results])

    # Build X matrix (200, 4)
    X = np.array([[p[k] for k in keys] for p in param_list])

    np.savez('data/dataset.npz',
             X=X, coverages=coverages, gbds=gbds, times=times,
             param_keys=keys)

    print(f"\nDone. Coverage range: {coverages.min():.3f} – {coverages.max():.3f}")
    print(f"GBD range:      {gbds.min():.3f} – {gbds.max():.3f}")