# RESUME — current state & next action

One-screen handoff for picking this project back up. Full detail: `README.md` (phase log),
`ARCHITECTURE.md` (system map + §12 cross-era roadmap), `SCHEMA_NOTES.md` (data).

## What this is
A causal driver-vs-car attribution metric for F1 — separate a driver's contribution from the car,
answered as `do(constructor = X)` holding the driver fixed. DoWhy `gcm` SCM + a hierarchical
Bayesian (PyMC) latent skill/pace model, over the f1db dataset.

## Where we are (current best model)
**v2, time-varying skill** (per-driver per-season Gaussian random walk on qualifying pace) →
posterior `skill`/`car_pace` fed into the gcm SCM as continuous nodes. This reproduces the known
**car-dominant** split with believable driver rankings and career arcs.

**Key result — the split is ERA-DEPENDENT** (no single "X% driver / Y% car"; it depends on how much
car variation the window spans):

| era | car (median, 90% CrI) | driver | P(car>driver) |
|---|---|---|---|
| 2018–2025 (4 yr) | 31.9% [23, 42] | 21.4% [13, 29] | 73% (overlapping) |
| 2006–2025 (20 yr) | 43.6% [35, 48] | 12.4% [6, 15] | 100% (separated) |

`v1/` (categorical SCM) is the documented baseline that **fails** identification — do not trust v1 numbers.

## Reproduce (era is a CLI param everywhere; artifacts tagged by era)
```bash
.venv/bin/pip install -r requirements.txt
python scripts/download_data.py && python scripts/build_dataset.py --start 2018 --out-tag _2018_2025
python v2/build_quali.py --start 2018 --out-tag _2018_2025
python v2/fit_skill_rw.py --data data/f1_quali_2018_2025.parquet --tag _2018_2025_rw   # PyMC
python v2/build_scm_data.py --results data/f1_results_2018_2025.parquet \
    --idata models/v2_idata_2018_2025_rw.pkl --out data/f1_scm_v2_2018_2025_rw.parquet
python v2/attribution_v2.py --data data/f1_scm_v2_2018_2025_rw.parquet --tag _2018_2025_rw
python v2/uncertainty_propagation.py --idata models/v2_idata_2018_2025_rw.pkl \
    --results data/f1_results_2018_2025.parquet --tag _2018_2025_rw   # credible intervals
python v2/era_connectivity.py                                          # teammate-graph sweep
```
`.pkl`/`data/*.parquet` are gitignored (regenerable); reports/figures are tracked.

## Next action — toward cross-era ("Senna in a modern Red Bull")
Connectivity to Senna is **confirmed** (`v2/era_connectivity.py`). Plan: 3 PRs (cross-era plan file).
1. ✅ **Step 8a — session-consistent quali normalization** done (`build_quali.py --gap-method session`):
   gap to same-session pole; fixes a ~0.8% backmarker bias, unifies the 2006 format boundary,
   attribution robust. Default stays `pole`.
2. **NEXT — Step 8b:** fit back to Senna's era (1988–2025) with `--gap-method session` + an era-varying
   likelihood noise term in `v2/fit_skill_rw.py` (`sigma` per era bucket). Check convergence + that
   Senna/Prost/Schumacher rank believably; then SCM attribution on the full span.
3. **Step 8c:** new `v2/cross_era_query.py` — `do(car_pace=2024-RedBull, skill=Senna@peak)` → finish
   distribution + WIDE credible interval (reuse uncertainty_propagation's joint-draw pattern) + an
   explicit off-support-extrapolation caveat.

Other open threads: session-matched quali normalization; model race pace as a 2nd signal; a
driver-error-DNF (incident-proneness) term.

## Workflow
Land changes via **PRs, never commit to `main`** (see `CLAUDE.md`). Branch → commit → push →
`gh pr create`. Cross-session memory lives at the Claude project memory dir (auto-loads).
