# Architecture

System map for **apex-attribution** — a causal driver-vs-car attribution metric for Formula 1.

This documents the state through **v2 phase 2**. v1 (phases a–f) and v2 phase 1 are on `main`;
v2 phase 2 (`v2/build_scm_data.py`, `v2/attribution_v2.py`) lands with **PR #7** — links to those
two files resolve once it merges. See `README.md` for status, `SCHEMA_NOTES.md` for the data
schema, and `CLAUDE.md` for working conventions.

## 1. Purpose

Separate a driver's own contribution to a result from the car they drove, and answer it as an
intervention: *`do(constructor = X)` holding the driver fixed*. The identification lever is the
**teammate structure** — two drivers in the same constructor-season share the car, so their gap
is mostly skill; drivers switching teams across seasons chain these comparisons across the grid.

There are **two modelling lines**:

- **v1** — a DoWhy `gcm` structural causal model with raw **categorical** `driver`/`constructor`
  nodes. Outcome: it runs end-to-end but **does not identify** driver vs car (drivers are nested
  in cars, so the SCM dumps the car's pace onto driver identity).
- **v2** — a hierarchical Bayesian model (PyMC) that estimates **latent driver skill and car pace
  as separate continuous parameters** from qualifying pace, then feeds them into the SCM. Outcome:
  a large improvement (car effect restored, sensible counterfactuals) but car-dominance is still
  not fully reproduced — residual entanglement overstates driver skill.

## 2. System at a glance (data flow)

```
  f1db SQLite (v2026.7.0)
        │  scripts/download_data.py        (cache, version-pinned)
        ▼
  data/f1db.sqlite ──────────────────────────────────────────────┐
        │  scripts/build_dataset.py                                │ v2/build_quali.py
        ▼                                                          ▼
  data/f1_results.parquet                              data/f1_quali.parquet
  (1508 started / 1343 classified)                     (1480 quali entries, pct_gap)
        │                                                          │
        │  ┌─── V1 PIPELINE (gcm, categorical) ───┐                │  v2/fit_skill.py (PyMC)
        │  │ scripts/dag.py        (DAG)          │                ▼
        │  │ scripts/fit_model.py  (assign+fit)   │      models/v2_idata.pkl
        │  │ scripts/validate_model.py (checks)   │      (posterior skill[driver],
        │  │ scripts/attribution.py (ICC+CF)      │       pace[team-year])
        │  └──────────────────────────────────────┘                │
        │            │                                              │ v2/build_scm_data.py
        ▼            ▼                                              ▼  (merge latents)
  models/scm_*.pkl   outputs/*_report.txt              data/f1_scm_v2.parquet
  figures/*.png                                                    │
                                                                   │ v2/attribution_v2.py
                                          ┌── V2 PIPELINE (gcm, continuous skill/pace) ──┐
                                          │ same gcm SCM, but driver_skill & car_pace    │
                                          │ replace the categorical nodes (corr 0.49)    │
                                          └───────────────────────────────────────────────┘
                                                                   ▼
                                          outputs/v2_attribution_report.txt
                                          figures/v2_attribution_diagnostic.png
```

## 3. Repository layout

```
apex-attribution/
├── README.md             status, headline findings, setup
├── ARCHITECTURE.md       this file
├── SCHEMA_NOTES.md       f1db entity→column map, DNF taxonomy, gcm-0.14 API notes
├── CLAUDE.md             agent conventions (PR-based workflow)
├── requirements.txt      pinned deps (verified env)
├── scripts/              v1 pipeline + shared data prep
│   ├── download_data.py    cache f1db SQLite (idempotent, version-pinned)
│   ├── build_dataset.py    -> data/f1_results.parquet (cohort + DNF taxonomy)
│   ├── dag.py              v1 causal DAG (networkx.DiGraph) + figure
│   ├── fit_model.py        assign gcm mechanisms + fit -> models/scm_*.pkl
│   ├── validate_model.py   evaluate_causal_model + falsify_graph + chi-square
│   └── attribution.py      ICC + counterfactual swaps + unified metric
├── v2/                   hierarchical latent skill/pace line
│   ├── build_quali.py      -> data/f1_quali.parquet (pct gap to pole)
│   ├── fit_skill.py        PyMC crossed random-effects -> models/v2_idata.pkl
│   ├── build_scm_data.py   merge posterior latents -> data/f1_scm_v2.parquet
│   ├── attribution_v2.py   race-outcome SCM with continuous skill/pace
│   └── uncertainty_propagation.py  ICC over posterior draws (credible intervals)
├── data/                 (gitignored except f1db.version) source DB + derived frames
├── models/               (gitignored *.pkl) fitted SCMs + idata; reliability_rates.json tracked
├── outputs/              text reports + intervention grids (tracked)
└── figures/              PNG diagrams (tracked)
```

## 4. Data layer

**Source:** [f1db](https://github.com/f1db/f1db) `v2026.7.0`, CC-BY-4.0, cached as
`data/f1db.sqlite` (73 MB). String slug ids for entities; `race_data` is one long table keyed by a
`type` discriminator. Full schema mapping in `SCHEMA_NOTES.md`.

Three derived dataframes (all reproducible, gitignored):

| file | grain | rows | key columns |
|---|---|---|---|
| `data/f1_results.parquet` | race entry (started) | 1508 (1343 classified) | grid, finish_pos, points, `dnf_cause`, reliability_dnf |
| `data/f1_quali.parquet` | quali entry | 1480 | `pct_gap` (% off pole), team_year |
| `data/f1_scm_v2.parquet` | race entry + latents | 1508 | adds `driver_skill`, `car_pace` |

**Cohort (both lines):** 2022–2025 (one broad regulation era), restricted to the **largest
teammate-connected component (25 drivers)** so the comparison scale is identified. Six detached
Haas/Sauber backmarkers are dropped (logged). DNFs split into `mechanical` / `driver_error` /
`other` so a car failure is never charged to the driver (Option A — see §7).

## 5. v1 pipeline — gcm SCM with categorical nodes

**DAG** (`scripts/dag.py`; node names == dataframe columns so `gcm.fit` needs no renaming):

```
circuit_type ─┐
constructor ──┼─→ grid ──────────┐
driver ───────┘                  ├─→ finish_pos        reliability_dnf is a PARALLEL leaf
constructor ─────────────────────┤  (← constructor)    (Option A: not a parent of finish_pos;
driver ──────────────────────────┤                      combined with finish only at reporting)
circuit_type ────────────────────┘
```

**Phases:** (a) env+download+schema → (b) `build_dataset.py` → (c) `dag.py` →
(d) `fit_model.py` (`gcm.auto.assign_causal_mechanisms` + `gcm.fit`; roots→EmpiricalDistribution,
grid/finish_pos→DiscreteAdditiveNoiseModel, reliability→ClassifierFCM; finish SCM is
`InvertibleStructuralCausalModel` on classified rows) → (e) `validate_model.py` → (f)
`attribution.py` (`intrinsic_causal_influence` + `counterfactual_samples` + unified metric).

**Validation result (e):** mechanisms reproduce the data and invertibility holds, but the
`driver ⊥ constructor` independence is strongly **violated** (chi-square p≈0, Cramér's V 0.84) —
selection confounding.

**Attribution result (f):** **non-identified.** ICC gives driver 45% vs constructor 1.3%;
swapping Verstappen's car moves him ~2 positions while swapping the driver in a Red Bull moves it
~8. Each driver is nested in ~one car, so the SCM loads the car's pace onto driver identity.
`figures/attribution_diagnostic.png` shows the flat car panel. **Do not trust v1 numbers.**

## 6. v2 pipeline — hierarchical latent skill/pace + SCM integration

**Step 1 — identify latents** (`v2/fit_skill.py`, PyMC). Robust crossed random effects on
qualifying pace:

```
pct_gap[driver, race] ~ StudentT(ν, μ, σ)
μ = skill[driver] + pace[constructor-season]
skill ~ ZeroSumNormal(σ_skill)   # sum-to-zero anchors the scale; relative driver pace
pace  ~ Normal(μ_pace, σ_pace)   # car pace per team-year; carries the absolute level
```

Teammates share `pace`; team-switchers chain the scales. Converges (R-hat 1.00) and recovers a
**believable** ranking (Verstappen clear #1; Stroll/Sargeant/Latifi last) with car pace cleanly
separated — the v1 separation failure is fixed at the quali stage.
(`figures/v2_driver_skill.png`)

**Step 2 — feed latents into the SCM** (`v2/build_scm_data.py` + `v2/attribution_v2.py`, PR #7).
Posterior means replace the categorical nodes:

```
circuit_type ─┐
driver_skill ─┼─→ grid ─────────┐        driver_skill & car_pace are now CONTINUOUS and only
car_pace ─────┘                 ├─→ finish_pos   correlated 0.49 (vs v1's near-perfect nesting),
driver_skill ───────────────────┤        so the SCM can separate them.
car_pace ───────────────────────┤
circuit_type ───────────────────┘
```

**Result:** big improvement, not a full fix. ICC car 1.3%→**6.5%** vs driver 32%; interventional
car effect ~2.3→**4.3** positions; counterfactual swaps now **move** (Albon Williams→Red Bull
P13→P10). But driver still leads ~5:1, vs the literature's car-dominant split. OLS standardized
betas (`finish ~ skill 0.56, pace 0.20`) show this lives in the **latents**, not gcm: residual
skill/pace entanglement and thinly-connected backmarkers (Latifi/Sargeant absorbing Williams)
overstate driver skill, which the race attribution inherits.

**Step 3 — propagate posterior uncertainty** (`v2/uncertainty_propagation.py`). Re-runs ICC over
30 *joint* posterior draws (not point means). Car share 6.5%→median **9.7%** (90% CrI [5.3, 17.0]),
driver 31.6%→**34.7%** [25.2, 42.1], ratio 4.9x→**3.8x** [1.8, 7.8]. Shifts modestly toward the
car and shows the split is poorly identified, but does **not** rescue car-dominance — driver leads
in every draw. Conclusion: the driver-heaviness is robust to SCM-stage uncertainty, so the bias is
**upstream** in the quali-stage skill identification, not in point estimates.
(`figures/v2_uncertainty.png`)

## 7. Key design decisions

| decision | choice | why |
|---|---|---|
| Era | 2022–2025 | one broad regulation era; minimizes era/reg confounds |
| Cohort | largest teammate-connected component (25 drivers) | one identified comparison scale |
| DNF handling | **Option A** — finish_pos on classified rows only; reliability a separate leaf | never charge a mechanical failure to the driver; unified metric combined only at reporting |
| v2 identifying signal | **qualifying pace** (% gap to pole) | cleanest car-equalized skill measure; avoids DNF/strategy/lap-1 noise |
| v2 tool | **PyMC** (Bayesian) | partial pooling shrinks thin drivers; uncertainty intervals |
| v2 anchoring | sum-to-zero on skill | resolves the skill-vs-pace level trade-off |
| Persistence | pickle SCMs/idata | gcm/PyMC lack a stable save; no netCDF backend installed |

## 8. Artifacts & persistence

**Tracked:** all source code, `*.md` docs, `requirements.txt`, `data/f1db.version`,
`models/reliability_rates.json`, everything in `outputs/` and `figures/`.

**Gitignored (regenerable):** `.venv/`, `data/*.sqlite|parquet|csv`, `models/*.pkl`,
`.claude/settings.local.json`.

Fitted artifacts: `models/scm_finish.pkl`, `models/scm_reliability.pkl` (v1),
`models/v2_idata.pkl` (v2 posterior). Reports in `outputs/*.txt` + `intervention_grid.csv`.
Figures: `dag.png`, `attribution_diagnostic.png` (v1), `v2_driver_skill.png`,
`v2_attribution_diagnostic.png` (v2).

## 9. Reproduction

```bash
brew install python@3.12 && python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

python scripts/download_data.py        # -> data/f1db.sqlite (pinned v2026.7.0)
python scripts/build_dataset.py        # -> data/f1_results.parquet

# v1 line
python scripts/dag.py                   # DAG + figure
python scripts/fit_model.py             # -> models/scm_*.pkl
python scripts/validate_model.py        # validation report
python scripts/attribution.py           # ICC + counterfactuals + unified metric

# v2 line
python v2/build_quali.py                # -> data/f1_quali.parquet
python v2/fit_skill.py                   # PyMC -> models/v2_idata.pkl
python v2/build_scm_data.py             # merge latents -> data/f1_scm_v2.parquet   (PR #7)
python v2/attribution_v2.py             # race-outcome attribution w/ latents       (PR #7)
```

## 10. Known limitations & open problems

1. **Selection confounding (pitfall #1).** driver⊥constructor is false (V 0.84). v1 cannot
   identify; v2 reduces but does not eliminate it (latent corr 0.49).
2. **Residual skill/pace entanglement (v2).** Thinly-connected backmarkers absorb car-badness
   into "skill," overstating the driver share. The race attribution inherits this.
3. **DNF censoring.** Handled via Option A; driver-error DNFs are not yet a modelled driver-risk
   term (only mechanical reliability is).
4. **Era restriction.** Cross-regulation comparison is explicitly out of scope.
5. **Quali session mixing (v2).** `best` lap mixes Q1/Q2/Q3; session-matched normalization is a
   refinement.
6. **Point estimates (v2 SCM).** Addressed in step 3 (`uncertainty_propagation.py`): propagating
   posterior uncertainty widens the attribution and shifts it modestly toward the car, but does not
   rescue car-dominance — confirming the bias is **upstream** in the latent identification, not in
   SCM-stage point estimates.

## 11. Tech stack

Python 3.12. `dowhy==0.14` (`gcm`), `pandas==3.0.3`, `numpy==2.4.6`, `scikit-learn==1.9.0`,
`networkx==3.6.1`, `scipy==1.15.3`, `statsmodels==0.14.6`, `matplotlib==3.11.0`,
`pyarrow==24.0.0`; v2 adds `pymc==6.0.1`, `arviz==1.2.0`, `pytensor==3.0.7`. Full pins in
`requirements.txt`.

## 12. Roadmap

Uncertainty propagation is done (step 3) and localized the remaining bias upstream. Next
candidates (see §10) target the **latent identification** itself: **widen the era** for more
team-switching (better scale chaining); session-matched quali normalization; model **race pace**
directly as a second signal; add a driver-error-DNF risk term.
