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
| 2018–2026 (+ new regs, half season) | 41.9% [37, 48] | 13.5% [8, 19] | 100% (separated) |
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

# current-regs view: same chain with --start 2018 --end 2026 / --out-tag _2018_2026 / --tag _2018_2026_rw
python v2/score_forecast.py                                            # score the 2026 forecast vs reality
```
`.pkl`/`data/*.parquet` are gitignored (regenerable); reports/figures are tracked.

## Current focus — a "validate → play → write up" week (plan file)
The core model is done & validated; cross-era hit a fundamental identification wall (8b: old-era
legends shrunk, R-hat 1.04 — NOT committed). So the week's spine is VALIDATION (highest learning +
admissions value), then fun demos, then a write-up.
- ✅ **Phase A — out-of-sample backtest** (`v2/backtest.py`): fit skills on 2018–2023, predict held-out
  2024–2025 teammate H2H (car cancels). **67% race / 80% season-long accuracy** (vs 50%), correlation
  0.40, intervals slightly conservative (50%→74% coverage). The model predicts the future.
- **Phase B (fun):**
  - ✅ `v2/predict.py` — forecast a season's teammate H2H from skills projected forward via the RW
    (`sqrt(forward)*sigma_rw` uncertainty widening). Emits a projected skill power-ranking + per-team
    H2H (expected gap + P(out-qualifies), backtest-calibrated noise). Default = last-season line-ups;
    `--lineup pairs.json` for hypotheticals. 2026: Verstappen > Tsunoda P=78%, close pairs ≈ coin-flip.
  - ✅ `v2/insights.py` — over-/under-rated drivers: actual finish vs SCM `do(car_pace=median)`
    expected finish → car effect in positions (Hamilton/Piastri flattered ~+4; Albon/Sargeant held
    back). Plus pace-vs-results table and best/worst car ranking. Reuses attribution_v2's SCM.
  - ✅ `v2/cross_era.py` — "Senna in a modern Red Bull", illustrative only. Era-scale fix via
    z-score: legend's SDs-ahead-of-their-field mapped onto the modern field spread (Senna −2.18 SD
    → modern-equiv −1.59%, ≈ Verstappen). Fed with `red-bull@2024` pace through the modern SCM →
    era-normalized greats land ≈ P3–3.6 (vs Verstappen P3.4). Loud caveats (off-support, era-scale
    assumption, source model R-hat 1.04). Uses existing `v2_idata_1988_2025_sess_rw.pkl` — no re-fit.
  - **Phase B complete.** Optional polish: a proper era-varying-σ re-fit to convergence
    (8b, branch `v2-senna-era-fit`) would replace the z-score heuristic; not required for the demo.
- **Phase C (write-up):** `WRITEUP.md` — the narrative (question → naive failure → fix → validation →
  honest limits → fun demos). Lead with validation.

Cross-era detail / staged blockers: ARCHITECTURE §12. Earlier cross-era step 8a (session norm) is done.

Race pace as a 2nd signal (IN PROGRESS): `v2/build_race_pace.py` + `v2/fit_skill_joint.py` fit two
correlated latents (`quali_skill`, `racecraft`), R-hat 1.01, rho +0.92; Pérez races > qualifies,
backmarker tail is lapped-sensitive. Next: thread race latents into the SCM (`build_scm_data.py
--skill-source race`) + an out-of-sample race-pace backtest (`backtest_race.py`).

Driver-error-DNF / incident-proneness (DONE): `v2/fit_incident.py` — hierarchical logistic
(driver + circuit hazard), partial-pooled. Shrinkage compresses raw rates (Grosjean 15.8%, Norris
2.6%) to a narrow ~5.4-7.0% band (real but modest; ordering robust). R-hat 1.00. Saves
`models/incident_rates_2018_2025.json`, folded into `v2/unified_metric.py` (E_all = finish +
mechanical + incident risk). Incident tax is stakes-dominated (Verstappen loses most/race to a low
crash rate × costly fall); cleanliness dividend (proneness only) tiny — Norris/Hamilton save
~0.06 pos/race. Incident-proneness = real but modest tiebreaker.

2026 data (DONE, f1db pin `v2026.11.0`): first half of 2026 (rounds 1–11, to the summer break)
imported. Two results. (1) `v2/score_forecast.py` scores the pre-season 2026 forecast against
reality — a truly prospective test, nothing retuned: **70% season-long teammate H2H (7/10)**, 58%
race-level, corr 0.39; confident calls held (Verstappen > Hadjar, Alonso > Stroll), clearest miss is
**Antonelli beating Russell** (predicted P=70% the other way). Honest negative: MAE 0.42% loses to a
predict-zero baseline because the reg reset widened the field 2.6× (grid SD 0.46% → 1.19%) — ordering
survives, magnitudes are mis-scaled. (2) The reset re-stratified the grid, so 2018–2026 shifts hard
toward the car (necessity: driver 84% → 57%; OLS pace overtakes skill; ICC P(car>driver) 73% → 100%).
**Caveat: half a season** — refresh when 2026 completes. 2018–2025 artifacts preserved as the
converged-era baseline.

Site moved to 2018–2026 (PR after #56): joint re-fit `_2018_2026_joint` (R-hat 1.020, rho +0.91)
now drives car-swap / arcs / H2H, so Cadillac, Audi, Hadjar-at-RBR and Lindblad are selectable;
2018–2025 kept as an era-slider stop. Masthead necessity hook 82/68 → **91/61**. NOTE the joint-scale
move is MILDER than the quali-RW headline — ICC car share actually dips slightly (35.6→32.4) while
driver drops more (16.3→13.7); it's the RATIO that moves, not a car-share jump. `manifest.partialSeason`
(derived from f1db round counts) drives an inline `PartialSeasonNote` and self-clears when 2026 ends.

Other open threads: (none major outstanding).

## Workflow
Land changes via **PRs, never commit to `main`** (see `CLAUDE.md`). Branch → commit → push →
`gh pr create`. Cross-session memory lives at the Claude project memory dir (auto-loads).
