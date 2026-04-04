"""
Constrained Bayesian optimization over trained GP surrogate models.

Goal: minimize predicted grain boundary density (GBD) subject to
predicted coverage >= target threshold.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from run_batch import BOUNDS


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x):
    # Fast smooth approximation of Normal CDF; avoids scipy dependency.
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def sample_uniform(bounds_by_key, keys, n_samples, rng):
    x = np.empty((n_samples, len(keys)), dtype=float)
    for i, key in enumerate(keys):
        lo, hi = bounds_by_key[key]
        x[:, i] = rng.uniform(lo, hi, size=n_samples)
    return x


def to_model_space(x_raw, scaler, f_idx):
    x = x_raw.copy().astype(float)
    x[:, f_idx] = np.log10(np.clip(x[:, f_idx], 1e-12, None))
    return scaler.transform(x)


def predict_stats(x_raw, gp_cov, gp_gbd, scaler, f_idx):
    x_model = to_model_space(x_raw, scaler, f_idx)
    mu_cov, std_cov = gp_cov.predict(x_model, return_std=True)
    mu_gbd, std_gbd = gp_gbd.predict(x_model, return_std=True)
    return mu_cov, std_cov, mu_gbd, std_gbd


def probability_feasible(mu_cov, std_cov, cov_target):
    z = (cov_target - mu_cov) / np.maximum(std_cov, 1e-12)
    return 1.0 - norm_cdf(z)


def expected_improvement_for_min(mu, std, incumbent_best):
    sigma = np.maximum(std, 1e-12)
    improvement = incumbent_best - mu
    z = improvement / sigma
    ei = improvement * norm_cdf(z) + sigma * norm_pdf(z)
    return np.maximum(ei, 0.0)


def novelty_weight(x_candidates, x_history, bounds_matrix):
    if x_history.shape[0] == 0:
        return np.ones(x_candidates.shape[0], dtype=float)

    spans = np.maximum(bounds_matrix[:, 1] - bounds_matrix[:, 0], 1e-12)
    diff = (x_candidates[:, None, :] - x_history[None, :, :]) / spans[None, None, :]
    min_dist = np.sqrt(np.min(np.sum(diff**2, axis=2), axis=1))
    return 0.25 + 0.75 * np.clip(min_dist / 0.12, 0.0, 1.0)


def infer_training_bounds(gp_cov, scaler, f_idx, keys, pad_frac=0.05):
    """Recover raw-space bounds from GP training inputs and lightly pad them."""
    x_scaled = np.asarray(gp_cov.X_train_, dtype=float)
    x_raw = scaler.inverse_transform(x_scaled)
    x_raw[:, f_idx] = 10 ** x_raw[:, f_idx]

    bounds = {}
    for i, key in enumerate(keys):
        lo = float(np.min(x_raw[:, i]))
        hi = float(np.max(x_raw[:, i]))
        span = hi - lo
        if span > 0:
            lo -= pad_frac * span
            hi += pad_frac * span

        # Keep F positive after padding.
        if i == f_idx:
            lo = max(lo, 1e-6)

        if key in BOUNDS:
            phys_lo, phys_hi = BOUNDS[key]
            lo = max(lo, float(phys_lo))
            hi = min(hi, float(phys_hi))

        bounds[key] = (lo, hi)
    return bounds


def run_constrained_bo(
    gp_cov,
    gp_gbd,
    scaler,
    f_idx,
    keys,
    bounds,
    cov_target,
    n_init,
    n_iter,
    n_candidates,
    seed,
):
    rng = np.random.default_rng(seed)

    bounds_matrix = np.array([bounds[k] for k in keys], dtype=float)

    x_hist = sample_uniform(bounds, keys, n_init, rng)
    mu_cov_raw, std_cov, mu_gbd_raw, std_gbd = predict_stats(
        x_hist, gp_cov, gp_gbd, scaler, f_idx
    )
    # Keep optimization target in physically meaningful ranges.
    mu_cov = np.clip(mu_cov_raw, 0.0, 1.0)
    mu_gbd = np.clip(mu_gbd_raw, 0.0, 1.0)
    pf = probability_feasible(mu_cov, std_cov, cov_target)

    for _ in range(n_iter):
        x_cand = sample_uniform(bounds, keys, n_candidates, rng)
        c_mu_cov_raw, c_std_cov, c_mu_gbd_raw, c_std_gbd = predict_stats(
            x_cand, gp_cov, gp_gbd, scaler, f_idx
        )
        c_mu_cov = np.clip(c_mu_cov_raw, 0.0, 1.0)
        c_mu_gbd = np.clip(c_mu_gbd_raw, 0.0, 1.0)
        c_pf = probability_feasible(c_mu_cov, c_std_cov, cov_target)

        feasible_hist = mu_cov >= cov_target
        if np.any(feasible_hist):
            incumbent = float(np.min(mu_gbd[feasible_hist]))
        else:
            penalty = 2.0 * np.maximum(0.0, cov_target - c_mu_cov)
            incumbent = float(np.min(c_mu_gbd + penalty))

        ei = expected_improvement_for_min(c_mu_gbd, c_std_gbd, incumbent)
        acq = ei * c_pf
        acq *= novelty_weight(x_cand, x_hist, bounds_matrix)

        best_idx = int(np.argmax(acq))
        x_new = x_cand[best_idx : best_idx + 1]
        n_mu_cov_raw, n_std_cov, n_mu_gbd_raw, n_std_gbd = predict_stats(
            x_new, gp_cov, gp_gbd, scaler, f_idx
        )
        n_mu_cov = np.clip(n_mu_cov_raw, 0.0, 1.0)
        n_mu_gbd = np.clip(n_mu_gbd_raw, 0.0, 1.0)
        n_pf = probability_feasible(n_mu_cov, n_std_cov, cov_target)

        x_hist = np.vstack([x_hist, x_new])
        mu_cov = np.concatenate([mu_cov, n_mu_cov])
        std_cov = np.concatenate([std_cov, n_std_cov])
        mu_gbd = np.concatenate([mu_gbd, n_mu_gbd])
        std_gbd = np.concatenate([std_gbd, n_std_gbd])
        pf = np.concatenate([pf, n_pf])

    cov_lcb = mu_cov - 1.0 * std_cov
    feasible = cov_lcb >= cov_target

    if np.any(feasible):
        feasible_idx = np.where(feasible)[0]
        best_local = feasible_idx[np.argmin(mu_gbd[feasible_idx])]
    else:
        best_local = int(np.argmax(pf))

    order = np.argsort(mu_gbd)
    top = []
    for idx in order:
        row = {
            keys[i]: float(x_hist[idx, i]) for i in range(len(keys))
        }
        row.update(
            {
                "pred_coverage_mean": float(mu_cov[idx]),
                "pred_coverage_std": float(std_cov[idx]),
                "pred_gbd_mean": float(mu_gbd[idx]),
                "pred_gbd_std": float(std_gbd[idx]),
                "prob_cov_ge_target": float(pf[idx]),
                "coverage_lcb": float(cov_lcb[idx]),
                "meets_lcb_constraint": bool(feasible[idx]),
            }
        )
        top.append(row)
        if len(top) >= 10:
            break

    best = {
        keys[i]: float(x_hist[best_local, i]) for i in range(len(keys))
    }
    best.update(
        {
            "pred_coverage_mean": float(mu_cov[best_local]),
            "pred_coverage_std": float(std_cov[best_local]),
            "pred_gbd_mean": float(mu_gbd[best_local]),
            "pred_gbd_std": float(std_gbd[best_local]),
            "prob_cov_ge_target": float(pf[best_local]),
            "coverage_lcb": float(cov_lcb[best_local]),
            "meets_lcb_constraint": bool(feasible[best_local]),
        }
    )

    return {
        "target_coverage": cov_target,
        "n_init": n_init,
        "n_iter": n_iter,
        "n_candidates": n_candidates,
        "seed": seed,
        "best": best,
        "top_10": top,
    }


def main():
    parser = argparse.ArgumentParser(description="Constrained BO on GP surrogate models")
    parser.add_argument("--coverage-target", type=float, default=0.80)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--n-candidates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bounds-source",
        choices=["training", "run_batch"],
        default="training",
        help="Search box source: recovered GP training bounds (default) or run_batch.BOUNDS.",
    )
    parser.add_argument("--out", type=str, default="models/bo_result.json")
    args = parser.parse_args()

    cov_bundle = joblib.load("models/gp_coverage.pkl")
    gbd_bundle = joblib.load("models/gp_gbd.pkl")

    gp_cov = cov_bundle["gp"]
    gp_gbd = gbd_bundle["gp"]
    scaler = cov_bundle["scaler"]
    keys = [str(k) for k in cov_bundle["keys"]]
    f_idx = int(cov_bundle["F_idx"])

    if args.bounds_source == "training":
        bounds = infer_training_bounds(gp_cov, scaler, f_idx, keys)
    else:
        missing = [k for k in keys if k not in BOUNDS]
        if missing:
            raise ValueError(f"Missing bounds for keys: {missing}")
        bounds = {k: tuple(BOUNDS[k]) for k in keys}

    result = run_constrained_bo(
        gp_cov=gp_cov,
        gp_gbd=gp_gbd,
        scaler=scaler,
        f_idx=f_idx,
        keys=keys,
        bounds=bounds,
        cov_target=args.coverage_target,
        n_init=args.n_init,
        n_iter=args.n_iter,
        n_candidates=args.n_candidates,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    best = result["best"]
    print("Best constrained candidate (using GP surrogate):")
    print(
        "F={F:.6f}, E_d={E_d:.6f}, E_des={E_des:.6f}, T={T:.2f}".format(**best)
    )
    print(
        "pred_cov={pred_coverage_mean:.4f} +/- {pred_coverage_std:.4f}, "
        "pred_gbd={pred_gbd_mean:.4f} +/- {pred_gbd_std:.4f}, "
        "P(cov>=target)={prob_cov_ge_target:.4f}, cov_lcb={coverage_lcb:.4f}".format(**best)
    )
    print(f"Bounds source: {args.bounds_source}")
    print(f"Saved optimization report: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
