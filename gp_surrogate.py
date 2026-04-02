"""
gp_surrogate.py
Train two GP regressors: one for surface coverage, one for grain boundary density.
Saves trained models + shared scaler for Bayesian optimization.
"""

import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data      = np.load('data/dataset.npz', allow_pickle=True)
X_raw     = data['X'].astype(float)         # (N, 4): F, E_d, E_des, T
coverages = data['coverages'].astype(float)  # (N,)
gbds      = data['gbds'].astype(float)       # (N,)
keys      = list(data['param_keys'])         # ['F', 'E_d', 'E_des', 'T']
N         = X_raw.shape[0]

print(f"Loaded {N} samples | keys: {keys}")
print(f"Coverage : {coverages.min():.3f} – {coverages.max():.3f}")
print(f"GBD      : {gbds.min():.3f} – {gbds.max():.3f}")

if N < 20:
    print(f"WARNING: only {N} samples. Edit run_batch.py: "
          f"change latin_hypercube_sample(10, ...) to latin_hypercube_sample(200, ...).")

# ---------------------------------------------------------------------------
# 2. Preprocessing
#    F spans 0.01-1.0 (two decades) -> log10-transform before standardizing.
#    All four features then standardized to zero mean / unit variance.
#
#    Note: the scaler fitted here on all data is for the FINAL model and BO.
#    CV uses per-fold scalers (see cross_validate) to avoid leakage.
# ---------------------------------------------------------------------------
F_idx = keys.index('F')


def preprocess(X_in, scaler=None):
    """Apply log10(F), then StandardScaler. Fits scaler if not provided."""
    X = X_in.copy().astype(float)
    X[:, F_idx] = np.log10(X[:, F_idx])
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


# Fit on full dataset -- used for final models and BO
X_scaled, scaler = preprocess(X_raw)


# ---------------------------------------------------------------------------
# 3. Kernel + GPR factory
# ---------------------------------------------------------------------------
def make_kernel(n_dims):
    return (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1e-1)
        )
    )


def make_gp(n_dims):
    return GaussianProcessRegressor(
        kernel=make_kernel(n_dims),
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# 4. 5-fold cross-validation
#    Scaler is re-fit on each training fold to avoid leakage.
# ---------------------------------------------------------------------------
def cross_validate(X_in, y, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rmse_list, r2_list = [], []

    for tr, te in kf.split(X_in):
        # Fit scaler on training fold only
        X_tr, fold_scaler = preprocess(X_in[tr])
        X_te, _           = preprocess(X_in[te], scaler=fold_scaler)

        gp = make_gp(X_tr.shape[1])
        gp.fit(X_tr, y[tr])
        pred = gp.predict(X_te)

        rmse_list.append(np.sqrt(np.mean((pred - y[te]) ** 2)))
        r2_list.append(r2_score(y[te], pred))

    rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
    r2_mean,   r2_std   = np.mean(r2_list),   np.std(r2_list)
    print(f"[{label:12s}] CV RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f} | "
          f"R2: {r2_mean:.4f} +/- {r2_std:.4f}")
    return rmse_mean, r2_mean


print("\n-- Cross-validation (per-fold scaler, no leakage) --")
cv_rmse_cov, cv_r2_cov = cross_validate(X_raw, coverages, "Coverage")
cv_rmse_gbd, cv_r2_gbd = cross_validate(X_raw, gbds,      "GBD")

# ---------------------------------------------------------------------------
# 5. Final models on full dataset
# ---------------------------------------------------------------------------
print("\n-- Fitting final models on full dataset --")
n_dims = X_scaled.shape[1]
gp_cov = make_gp(n_dims)
gp_gbd = make_gp(n_dims)

gp_cov.fit(X_scaled, coverages)
gp_gbd.fit(X_scaled, gbds)

print(f"\nCoverage kernel : {gp_cov.kernel_}")
print(f"GBD kernel      : {gp_gbd.kernel_}")


def get_length_scales(gp):
    """Extract ARD length-scales from fitted kernel. Uses isinstance(np.ndarray)
    to distinguish the actual array from the bounds tuple in get_params()."""
    for name, val in gp.kernel_.get_params().items():
        if name.endswith('length_scale') and isinstance(val, np.ndarray):
            return val
    return None


ls_cov = get_length_scales(gp_cov)
ls_gbd = get_length_scales(gp_gbd)

if ls_cov is not None:
    print("\nARD length-scales (smaller = more influential):")
    print(f"  Coverage GP : { {k: f'{v:.3f}' for k, v in zip(keys, ls_cov)} }")
if ls_gbd is not None:
    print(f"  GBD GP      : { {k: f'{v:.3f}' for k, v in zip(keys, ls_gbd)} }")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
os.makedirs('models', exist_ok=True)

meta = {
    'scaler': scaler,   # fitted on full X_raw; used by BO
    'keys'  : keys,
    'F_idx' : F_idx,
    'cv_rmse': {'coverage': cv_rmse_cov, 'gbd': cv_rmse_gbd},
    'cv_r2' : {'coverage': cv_r2_cov,   'gbd': cv_r2_gbd},
}

joblib.dump({'gp': gp_cov, **meta}, 'models/gp_coverage.pkl')
joblib.dump({'gp': gp_gbd, **meta}, 'models/gp_gbd.pkl')

print("\nSaved: models/gp_coverage.pkl")
print("Saved: models/gp_gbd.pkl")