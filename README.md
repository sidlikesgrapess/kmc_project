# 2D kMC Crystal Growth with GP Surrogate and Constrained Optimization

## Goal
Implement a 2D kMC growth simulator on a triangular lattice (100x100, three event types: adsorption, desorption, diffusion, with nucleation and grain tracking), generate a dataset of 200 simulations across a 4D parameter space, train a GP surrogate, and use Bayesian optimization to find the growth conditions that minimize grain boundary density while maintaining coverage above 80%.

## What Is Implemented
1. A 2D triangular-lattice kMC simulator in [simulator/events.py](simulator/events.py).
2. 4D Latin Hypercube dataset generation (200 runs) in [run_batch.py](run_batch.py).
3. Two GP surrogate models (coverage and GBD) in [gp_surrogate.py](gp_surrogate.py).
4. Constrained surrogate optimization in [bayesian_optimize.py](bayesian_optimize.py).

## Model and Simulation Setup

### 4D parameter space
- F: 0.01 to 1.0
- E_d: 0.1 to 0.8
- E_des: 1.0 to 3.0
- T: 200 to 800

### kMC details
- Lattice: 100x100 triangular neighborhood
- Events: adsorption, desorption, diffusion
- Grain logic: adsorption assigns grain id (new nucleus or join neighboring grain), diffusion carries grain id
- GBD metric: fraction of occupied-occupied neighbor pairs with different grain ids

### Runtime/accuracy tradeoff currently used
- Aggregated event-rate sampling is used for speed (same event probabilities under current rate model).
- Diffusion cap heuristic is enabled by default with max_diff_to_ads_ratio = 120.0 to avoid diffusion-dominated runtime collapse.
- Coverage is not forced during dataset generation (target_coverage = None).

## Current Artifact Summary (from files in this repo)
The numbers below are from:
- [data/dataset_old.npz](data/dataset_old.npz)
- [models/gp_coverage.pkl](models/gp_coverage.pkl)
- [models/gp_gbd.pkl](models/gp_gbd.pkl)
- [models/bo_result.json](models/bo_result.json)

### Dataset summary (n = 200)
- Coverage range: 0.0 to 1.0
- GBD range: 0.0 to 1.0

Correlations:
- corr(F, coverage) = +0.1044, corr(F, gbd) = +0.1232
- corr(E_d, coverage) = +0.0295, corr(E_d, gbd) = +0.5136
- corr(E_des, coverage) = +0.5848, corr(E_des, gbd) = +0.1017
- corr(T, coverage) = -0.4970, corr(T, gbd) = -0.5603

### GP cross-validation (saved model metadata)
- Coverage GP: RMSE = 0.0604, R2 = 0.9665
- GBD GP: RMSE = 0.0253, R2 = 0.9797

### Current constrained optimization output
From [models/bo_result.json](models/bo_result.json), best candidate is:
- F = 0.9924518084
- E_d = 0.1029979674
- E_des = 2.4647343538
- T = 225.5253259793

Predicted at this point:
- coverage mean = 0.9689, std = 0.0485
- gbd mean = 0.0000, std = 0.0558
- P(coverage >= 0.8) = 0.9998
- coverage LCB = 0.9203

*Important - This result is not physically accurate due to my error in generating the dataset. I am trying to fix the parameters to get a diverse coverage and gbd to train the model on.
## Important Interpretation Notes
1. The optimizer currently clips surrogate means to [0, 1]. This keeps outputs physical but can flatten objective ranking when many candidates hit gbd mean = 0.0.
2. The BO script is surrogate-driven, not closed-loop with simulator updates after each suggestion.
3. BO recommendations should be validated with direct kMC runs before claiming physical optima.

## How To Run

Install requirements:

  pip install -r requirements.txt

Generate a 200-run dataset:

  python run_batch.py

This writes:
- [data/dataset.npz](data/dataset.npz)

Train GP surrogate models:

  python gp_surrogate.py

This writes:
- [models/gp_coverage.pkl](models/gp_coverage.pkl)
- [models/gp_gbd.pkl](models/gp_gbd.pkl)

Run constrained optimization (coverage >= 0.8):

  python bayesian_optimize.py --coverage-target 0.80 --n-init 20 --n-iter 80 --n-candidates 5000 --seed 42 --bounds-source training

This writes:
- [models/bo_result.json](models/bo_result.json)

## Generating Simulation Movies
You can generate visual animations (.mp4 or .gif) of a single kMC simulation run using the `make_simulation_movie.py` script. The script applies a strict physical stability rule: by default, any atom with 2 or more occupied neighbors is permanently locked in place (cannot detach or diffuse) unless it loses neighbors.

Generate a standard movie representing a 50,000 step simulation at 10 FPS:
```bash
python make_simulation_movie.py --max-steps 50000 --fps 10 --snapshot-every 10 --out movies/sim_50k.mp4
```

**Key Parameters**:
- `--F`, `--E_d`, `--E_des`, `--T`: Physical simulation parameters.
- `--max-steps`: Total number of kMC steps to simulate (default: 1,500,000).
- `--snapshot-every`: Capture a frame every N steps (e.g., 10 or 20 for smoother playback on long runs).
- `--fps`: Frames per second of the output video.
- `--immobile-if-neighbors-ge`: Minimum occupied neighbors for an atom to become immobile (default: 2).
- `--out`: Output file path. Use `.mp4` for much smaller, faster-rendering files, or `.gif` if FFMpeg is not available on your system.

## Project Structure
- [simulator/events.py](simulator/events.py): kMC event loop and grain/GBD logic
- [simulator/lattice.py](simulator/lattice.py): lattice and neighbor table
- [run_batch.py](run_batch.py): LHS sampling and parallel dataset generation
- [gp_surrogate.py](gp_surrogate.py): GP training and CV
- [bayesian_optimize.py](bayesian_optimize.py): constrained surrogate optimization
- [verify_dataset.py](verify_dataset.py): quick dataset diagnostics

## Dependencies
- numpy
- scikit-learn
- joblib
