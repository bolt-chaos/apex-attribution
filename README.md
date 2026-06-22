# apex-attribution

A **causal attribution metric for Formula 1 drivers** — separating a driver's own
contribution from the car they happened to be driving. Instead of a hand-weighted
box-score sum (à la basketball's PER), the "weights" fall out of a fitted **structural
causal model**, so we can pose the eternal F1 bar argument — *how much is the driver vs.
the machinery?* — formally, as a `do(constructor = X)` intervention holding the driver fixed.

Built on [DoWhy's `gcm` module](https://github.com/py-why/dowhy) over the
[f1db](https://github.com/f1db/f1db) open dataset (CC-BY-4.0).

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
