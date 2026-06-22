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
- [ ] **(c)** DAG definition (`networkx.DiGraph`)
- [ ] **(d)** mechanism fitting (`gcm.auto.assign_causal_mechanisms`, `gcm.fit`)
- [ ] **(e)** validation (`gcm.evaluate_causal_model`, `falsify_graph`)
- [ ] **(f)** attribution (`intrinsic_causal_influence`) + counterfactual driver-swap demo

**v1 scope:** 2022–2025 (single-ish regulation era), the 25-driver largest teammate-connected
component. Known limitations (selection confounding, latent skill/pace, DNF censoring, era
effects) are documented in `SCHEMA_NOTES.md` and handled explicitly, not papered over.

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
