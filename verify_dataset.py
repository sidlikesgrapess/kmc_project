import numpy as np

data = np.load('data/dataset.npz')
X    = data['X']
cov  = data['coverages']
gbd  = data['gbds']
keys = list(data['param_keys'])

print("=== Correlations ===")
for i, k in enumerate(keys):
    print(f"corr({k:<6}, coverage) = {np.corrcoef(X[:,i], cov)[0,1]:+.3f} | "
          f"corr({k:<6}, gbd)      = {np.corrcoef(X[:,i], gbd)[0,1]:+.3f}")

print("\n=== Time sanity ===")
print(f"All times > 0  : {np.all(data['times'] > 0)}")
print(f"Any inf times  : {np.any(np.isinf(data['times']))}")

print("\n=== Value ranges ===")
print(f"Coverage : {cov.min():.3f} – {cov.max():.3f}")
print(f"GBD      : {gbd.min():.3f} – {gbd.max():.3f}")

print("\n=== GBD peaks at mid coverage? ===")
bins  = np.linspace(0, 1, 6)
idx   = np.digitize(cov, bins) - 1
for b in range(5):
    mask = idx == b
    if mask.sum() > 0:
        print(f"  cov {bins[b]:.1f}–{bins[b+1]:.1f} : mean GBD = {gbd[mask].mean():.3f}  (n={mask.sum()})")