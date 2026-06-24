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

## Current focus — a "validate → play → write up" week (plan file)
The core model is done & validated; cross-era hit a fundamental identification wall (8b: old-era
legends shrunk, R-hat 1.04 — NOT committed). So the week's spine is VALIDATION (highest learning +
admissions value), then fun demos, then a write-up.
- ✅ **Phase A — out-of-sample backtest** (`v2/backtest.py`): fit skills on 2018–2023, predict held-out
  2024–2025 teammate H2H (car cancels). **67% race / 80% season-long accuracy** (vs 50%), correlation
  0.40, intervals slightly conservative (50%→74% coverage). The model predicts the future.
- **NEXT — Phase B (fun):** `v2/predict.py` (forecast 2026 / a race); fun queries (over/under-rated
  driver); the **illustrative** cross-era demo (re-fit 1988–2025 to convergence first; report a WIDE
  CrI + "off-support / biased-low / illustrative-only" caveat). Era-sigma code is on branch
  `v2-senna-era-fit` (cherry-pick for the re-fit).
- **Phase C (write-up):** `WRITEUP.md` — the narrative (question → naive failure → fix → validation →
  honest limits → fun demos). Lead with validation.

Cross-era detail / staged blockers: ARCHITECTURE §12. Earlier cross-era step 8a (session norm) is done.

Other open threads: session-matched quali normalization; model race pace as a 2nd signal; a
driver-error-DNF (incident-proneness) term.

## Workflow
Land changes via **PRs, never commit to `main`** (see `CLAUDE.md`). Branch → commit → push →
`gh pr create`. Cross-session memory lives at the Claude project memory dir (auto-loads).
