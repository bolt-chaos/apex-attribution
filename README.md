# apex-attribution

A **causal attribution metric for Formula 1 drivers** — separating a driver's own
contribution from the car they happened to be driving. Instead of a hand-weighted
box-score sum (à la basketball's PER), the "weights" fall out of a fitted **structural
causal model**, so we can pose the eternal F1 bar argument — *how much is the driver vs.
the machinery?* — formally, as a `do(constructor = X)` intervention holding the driver fixed.

Built on [DoWhy's `gcm` module](https://github.com/py-why/dowhy) over the
[f1db](https://github.com/f1db/f1db) open dataset (CC-BY-4.0).

For the full system map (data flow, both modelling lines, artifacts, reproduction) see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Why this can work

Two drivers in the same constructor in a season **share the car**, so their head-to-head
difference largely nets out car quality. Because drivers change teams across seasons, those
pairwise comparisons **chain into a connected graph** spanning the grid (the same idea behind
chess Elo and adjusted plus-minus). The novelty over prior mixed-effects work (Bell et al.,
*"Formula for success"*) is **causal**: `gcm` supports interventional and counterfactual
queries — "put driver A in constructor B's car" — that an associational model cannot.

## Status

MVP, built incrementally:

- [x] **(a)** env + data download + schema introspection → [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md)
- [x] **(b)** clean per-result dataframe with the teammate-connected-component filter
- [x] **(c)** DAG definition (`networkx.DiGraph`) → [`scripts/dag.py`](scripts/dag.py), [`figures/dag.png`](figures/dag.png)
- [x] **(d)** mechanism fitting (`gcm.auto.assign_causal_mechanisms`, `gcm.fit`) → [`scripts/fit_model.py`](scripts/fit_model.py)
- [x] **(e)** validation (`gcm.evaluate_causal_model`, `falsify_graph`) → [`scripts/validate_model.py`](scripts/validate_model.py), [`outputs/validation_report.txt`](outputs/validation_report.txt)
- [x] **(f)** attribution (`intrinsic_causal_influence`) + counterfactual driver-swap demo + unified metric → [`scripts/attribution.py`](scripts/attribution.py), [`outputs/attribution_report.txt`](outputs/attribution_report.txt), [`figures/attribution_diagnostic.png`](figures/attribution_diagnostic.png)

**v1 scope:** 2022–2025 (single-ish regulation era), the 25-driver largest teammate-connected
component. Known limitations (selection confounding, latent skill/pace, DNF censoring, era
effects) are documented in `SCHEMA_NOTES.md` and handled explicitly, not papered over.

## Headline finding (v1): the machinery works, but the attribution is not identified

The full gcm pipeline runs end-to-end — ICC, interventional/counterfactual car-swaps, and the
reliability-combined "expected finish including breakdown risk" metric. **But v1 fails the
project's own sanity check** (under modern regs the *car* should dominate): both ICC and the
interventional spreads put almost all weight on the **driver** (ICC: driver 45% vs constructor
1.3%; swapping Verstappen's car moves him ~2 positions, swapping the driver in a Red Bull moves
it ~8 — see `figures/attribution_diagnostic.png`).

**Why:** each driver is nested in ~one constructor (Cramér's V 0.84), so the raw categorical SCM
cannot separate latent skill from latent car pace and dumps the car's pace onto driver identity.
The teammate structure was used only to pick a connected cohort, **not** to identify relative
skill. The signal *is* in the data — a simple teammate-contrast anchor recovers a believable
ranking (Verstappen, Alonso, Norris on top; Sargeant/Latifi/Stroll at the bottom) — the SCM as
specified just can't extract it.

**Fix (deferred, out of v1 scope):** a hierarchical latent skill/pace model that estimates
constructor-level car pace and driver-level skill as separate parameters (identified via those
teammate contrasts) and feeds them into the SCM. **Do not trust the v1 attribution numbers.**

## v2 (in progress): the hierarchical latent skill/pace model

Building the deferred fix in [`v2/`](v2). Phase 1 — **identification works**:

- [x] **quali-pace data** ([`v2/build_quali.py`](v2/build_quali.py)) — per-entry % gap to pole, the
  cleanest car-equalized skill signal; same connected cohort as v1.
- [x] **hierarchical PyMC fit** ([`v2/fit_skill.py`](v2/fit_skill.py)) — robust crossed random
  effects `pct_gap = skill[driver] + pace[constructor-season] + noise`. Teammates share `pace`,
  drivers switching teams chain the scales, sum-to-zero anchors skill. Converges (R-hat 1.00) and
  recovers a **believable skill ranking** (Verstappen clear #1, then Leclerc/Norris/Russell/Sainz;
  Stroll/Sargeant/Latifi last — see [`figures/v2_driver_skill.png`](figures/v2_driver_skill.png))
  with car pace cleanly separated. This is exactly what v1's raw-categorical SCM could not do.
- [x] **feed latents into the SCM** ([`v2/build_scm_data.py`](v2/build_scm_data.py),
  [`v2/attribution_v2.py`](v2/attribution_v2.py)) — categorical nodes replaced by continuous
  `driver_skill`/`car_pace` (now only corr 0.49, so separable). Re-ran the race-outcome ICC +
  counterfactuals → [`figures/v2_attribution_diagnostic.png`](figures/v2_attribution_diagnostic.png).

**v2 phase 2 result — big improvement, not yet a full fix.** The car effect is restored from v1's
flat line (ICC car 1.3%→6.5%; sweeping Verstappen across cars now spans ~4.5 positions, not ~1;
counterfactual swaps move sensibly — Albon Williams→Red Bull P13→P10). **But car-dominance still
isn't reproduced**: driver skill still leads car pace ~5:1 in ICC. OLS standardized betas confirm
this lives in the *latents*, not gcm (`finish ~ skill 0.56, pace 0.20`). The race attribution
inherits the quali-stage split, where residual skill/pace entanglement (corr 0.49) and
thinly-connected backmarkers (Latifi/Sargeant absorbing Williams) overstate driver skill.
**Phase 3 — propagate posterior uncertainty** ([`v2/uncertainty_propagation.py`](v2/uncertainty_propagation.py)).
Re-ran the ICC over 30 *joint* posterior draws (skill+pace from the same draw, respecting their
anti-correlation) instead of using point means → [`figures/v2_uncertainty.png`](figures/v2_uncertainty.png).
Result: car share 6.5%→median **9.7%** (90% CrI [5.3, 17.0]); driver 31.6%→**34.7%** [25.2, 42.1];
driver:car ratio 4.9x→**3.8x** [1.8, 7.8]. So propagating uncertainty shifts *modestly* toward the
car and shows the split is **not sharply identified** — but it does **not rescue car-dominance**:
the driver leads in every draw. That's informative — it localizes the remaining bias **upstream**
in the quali-stage skill identification (which SCM-stage uncertainty can't undo), not in
overconfident point estimates.

**Next:** better latent identification is the lever — more seasons (more team-switching to chain
scales), session-matched quali normalization, and/or modelling race pace directly as a second signal.

## Setup

```bash
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

python scripts/download_data.py      # caches f1db SQLite under data/ (pinned: v2026.7.0)
python scripts/build_dataset.py      # writes data/f1_results.parquet (+ .csv)
```

The `data/` artifacts are reproducible and untracked; `data/f1db.version` records the pinned
release. Verified env: Python 3.12, `dowhy==0.14` (see `requirements.txt` for the full pin set).

## Data attribution

F1 data from [f1db](https://github.com/f1db/f1db), licensed CC-BY-4.0.
