# 2D Kinetic Monte Carlo (kMC) Crystal Growth & ML Optimization

This repository contains a simplified 2-week proof-of-concept for a kinetic Monte Carlo (kMC) simulator modeling 2D crystal growth on a 100x100 triangular lattice. The project attempts to generate a dataset across a 4D parameter space, train a Gaussian Process (GP) surrogate model, and use Bayesian Optimization to find growth conditions that minimize Grain Boundary Density (GBD) while maintaining >80% coverage.

**Status: Functionally Limited / Does Not Work as Intended**

The current pure-Python kMC implementation has severe computational bottlenecks that prevent it from accurately simulating the physical system to the required >80% coverage within a reasonable timeframe.

### Known Issues & Limitations
1. **The Diffusion Bottleneck:** In pure Python, evaluating grid states and allocating arrays at every kMC step is too slow. To prevent single simulations from taking hours, the diffusion rate (`k_diff`) had to be artificially capped. 
2. **Coverage Ceiling:** Because we are restricted to the 4D parameter space (`F`, `E_d`, `E_des`, `T`) without a dedicated `dose` parameter, we used a fixed absolute simulation time. Consequently, the maximum coverage achieved in the dataset is only 50%, falling short of the >80% coverage constraint required for the Bayesian Optimization phase.
3. **Desorption is Negligible:** Given the sampled energy bounds (`E_des` = 1.6–2.0) and temperature bounds (`T` = 200–350), desorption is virtually non-existent, turning this essentially into a diffusion-aggregation model rather than a full adsorption-desorption-diffusion system.

---

### Current Dataset Statistics (n=50)

Despite the limitations, an initial Latin Hypercube Sampling (LHS) run was executed. The statistical correlations and sanity checks from the resulting dataset are below:

```text
=== Correlations ===
corr(F     , coverage) = +0.272 | corr(F     , gbd)      = -0.248
corr(E_d   , coverage) = +0.649 | corr(E_d   , gbd)      = -0.738
corr(E_des , coverage) = -0.117 | corr(E_des , gbd)      = +0.087
corr(T     , coverage) = -0.440 | corr(T     , gbd)      = +0.331

=== Time sanity ===
All times > 0  : True
Any inf times  : False

=== Value ranges ===
Coverage : 0.011 – 0.500
GBD      : 0.333 – 1.000

=== GBD peaks at mid coverage? ===
  cov 0.0–0.2 : mean GBD = 0.878  (n=42)
  cov 0.2–0.4 : mean GBD = 0.418  (n=1)
  cov 0.4–0.6 : mean GBD = 0.347  (n=7)
