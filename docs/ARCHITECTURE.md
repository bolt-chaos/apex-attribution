# Architecture

System map for **apex-attribution** — a causal driver-vs-car attribution metric for Formula 1.

This is the **system map**: data flow, both modelling lines (v1 archived, v2 current through the
joint quali+race fit, incidents, cross-era, and the interactive site — §13). Everything described
here is on `main`. The **authority on what's current** is `README.md` (status + phase log +
headline results, repo root); this file explains *how the pieces fit*, and §§5–6 deliberately keep
the v1→v2 history because the non-identification story is part of the design rationale. See
`SCHEMA_NOTES.md` (this folder) for the data schema and `CLAUDE.md` (repo root) for working
conventions.

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
├── README.md             status + phase log + headline results (the authority on what's current)
├── CLAUDE.md             agent conventions (PR-based workflow)
├── requirements.txt      pinned deps (verified env)
├── docs/                 reference docs — see the README reading map
│   ├── ARCHITECTURE.md     this file (system map)
│   ├── SCHEMA_NOTES.md     f1db entity→column map, DNF taxonomy, gcm-0.14 API notes
│   ├── CONCEPTS.md         plain-language concept guide
│   ├── BOOK_OF_WHY.md      mapping onto Pearl's ladder
│   ├── WRITEUP_NOTES.md    running write-up notes
│   ├── IDEAS.md            Pearl-style critique + backlog
│   └── RESUME.md           fast re-entry: current state + next action
├── scripts/              v1 pipeline (archived; §5) + shared data prep + site export
│   ├── download_data.py    cache f1db SQLite (idempotent, version-pinned)
│   ├── build_dataset.py    -> data/f1_results.parquet (cohort + DNF taxonomy)
│   ├── dag.py              v1 causal DAG (networkx.DiGraph) + figure
│   ├── fit_model.py        assign gcm mechanisms + fit -> models/scm_*.pkl
│   ├── validate_model.py   evaluate_causal_model + falsify_graph + chi-square
│   ├── attribution.py      v1 ICC + counterfactual swaps (superseded by v2)
│   └── export_site.py      bake models -> site/public/data/*.json
├── v2/                   hierarchical latent skill/pace line — full pipeline in §6
│   ├── build_quali.py / build_race_pace.py    identifying signals (quali %, race pace)
│   ├── fit_skill*.py       PyMC fits (pooled / _rw random-walk / _joint quali+race)
│   ├── build_scm_data.py   merge posterior latents -> data/f1_scm_v2_*.parquet
│   ├── attribution_v2.py   race-outcome SCM: ICC, do()-sweeps, rung-3 necessity
│   ├── fit_incident.py / unified_metric.py    incident-proneness + E_all
│   ├── cross_era.py / era_connectivity.py     era work
│   └── backtest*.py / insights.py / predict.py / attribution_eiv.py / uncertainty_propagation.py
├── site/                 Vite + React + TS interactive site (§13); data schema in site/DATA.md
├── data/                 (gitignored except f1db.version) source DB + derived frames
├── models/               (gitignored *.pkl) fitted SCMs + idata; *_rates.json tracked
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

**Cohort:** always the **largest teammate-connected component** within the chosen window, so the
comparison scale is identified; detached backmarkers are dropped (logged). v1 fixed the window at
2022–2025 (25 drivers); in v2 the **era is a CLI parameter** — the canonical modern window is
**2018–2025**, with wider windows (2006–2025, 1988–2025) fitted separately for the era slider and
cross-era work (the table above shows the original 2022–2025 row counts). DNFs split into
`mechanical` / `driver_error` / `other` so a car failure is never charged to the driver
(Option A — see §7).

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

**Step 2 — feed latents into the SCM** (`v2/build_scm_data.py` + `v2/attribution_v2.py`).
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

**Step 4 — widen the era to 2018–2025 → car-dominance reproduced.** The narrow era fragments into
3 teammate components; 2018–2025 is **one 41-driver component**, de-confounding car pace from driver
skill. Re-running the whole pipeline (era is a CLI param: `--start/--end/--out-tag`/`--tag`, artifacts
tagged `_2018_2025`, baseline preserved) flips every metric to car-dominant: quali systematic split
car **89%**/driver 11% (was 46/54); race ICC car **36.5%** vs driver **14.4%** → **PASS** (was
6.5 vs 31.6); OLS `finish ~ skill 0.38, pace 0.43` (car larger); counterfactual Sargeant Williams→
Ferrari **P12→P4**. Confirms the step-3 diagnosis: the driver-heaviness was the thin-connectivity
artifact, fixed by full connectivity. (`figures/v2_attribution_diagnostic_2018_2025.png`,
`figures/v2_driver_skill_2018_2025.png`)

**Step 5 — time-varying skill** (`v2/fit_skill_rw.py`). Replaces constant career skill with a
per-driver Gaussian random walk across seasons: `skill[d,s] = skill_base[d] + drift[d,s]`,
`drift[d,s] = drift[d,s-1] + σ_rw·z`. The data supports σ_rw ≈ 0.11%/season (HDI excludes 0) and
recovers believable arcs (Verstappen improving; Vettel/Ricciardo declining; rookies maturing).
`build_scm_data.py` detects the `season` dim and maps skill by `(driver, year)`. Car-dominance is
robust: race ICC car 26.6% vs driver 23.0% → PASS (moves toward parity from 36.5/14.4 because
season-specific skill captures more real driver signal). Anchoring caveat: the RW is anchored at
2018, so a late-debut driver's pre-debut (no-data) cells accumulate prior drift — minor at small
σ_rw. (`figures/v2_skill_trajectories_2018_2025_rw.png`)

**Step 6 — uncertainty on the current best model** (`v2/uncertainty_propagation.py`, generalized to
the time-varying model: it detects the `season` dim and maps each draw's skill by `(driver, year)`;
the point-estimate baseline is computed in-script via ICC on the posterior mean). 30 joint draws on
the wide+RW model give **car median 31.9%** (90% CrI [23.3, 42.3]) vs **driver 21.4%** [12.9, 29.3],
**P(car > driver) = 73%**. So "car leads" is the modal-but-not-decisive outcome — the CrIs overlap;
the honest statement is a range, not a verdict. This tempers the Step-5 point headline and is the
prerequisite for any cross-era claim. (`figures/v2_uncertainty_2018_2025_rw.png`)

**Step 7 — walk the era back** (`v2/era_connectivity.py` + the parameterized pipeline; `build_quali.py`
now reads `qualifying_time_millis` so the signal reaches 1980). (1) A teammate-graph connectivity
sweep: the main component stays 82–95% of drivers back to 1980, and **Senna joins the modern grid's
component once the era starts ≤1994** — connectivity is not the cross-era blocker. (2) The split is
**era-dependent**: 2006–2025 (20 yr) gives car 43.6% [35,48] vs driver 12.4% [6,15], P(car>driver)=
**100%**, vs 2018–2025's 32/21 at 73% — a wider window contains more car variation, so car-dominance
strengthens and sharpens. No single "X%/Y%"; it depends on the window. (3) Believable 20-yr arcs
(Hamilton peak+decline; Alonso dip+resurgence; Vettel decline). Remaining blockers: 2006 quali-format
change (single-session `qt` vs knockout best-of-q123), era-varying skill spread, off-support
counterfactual. (`figures/era_connectivity.png`, `figures/v2_skill_trajectories_2006_2025_rw.png`)

**Step 8a — session-consistent quali normalization** (`v2/build_quali.py --gap-method session`;
default stays `pole`). Measures each driver's gap to the fastest lap **in the same session** (min
over sessions reached), not to the overall Q3 pole — removing the knockout track-evolution bias that
over-penalized Q1-eliminated backmarkers by **~0.8%** (verified: backmarkers shrink ~1.26% vs
frontrunners ~0.49%), and unifying the pre-2006 single-session / 2006+ knockout formats. Skill
magnitudes become realistic (spread ~1.7%→~1.0%, order preserved); the 2018–2025 attribution is
**robust** (car 26.8% / driver 24.6% ≈ pole's 26.6/23.0). The first cross-era blocker, and the
foundation for 8b/8c.

## 7. Key design decisions

| decision | choice | why |
|---|---|---|
| Era | CLI parameter; canonical **2018–2025** (v1 used 2022–2025) | wide enough for grid connectivity within one hybrid era; wider windows (2006–, 1988–2025) fitted separately because the car/driver split is era-dependent — that dependence is itself a finding |
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
python v2/build_scm_data.py             # merge latents -> data/f1_scm_v2.parquet
python v2/attribution_v2.py             # race-outcome attribution w/ latents
```

## 10. Known limitations & open problems

1. **Selection confounding (pitfall #1).** driver⊥constructor is false (V 0.84). v1 cannot
   identify; v2 latents reduce it (corr 0.49→0.41 on the wide era) and the 2018–2025 single-component
   model resolves it enough to reproduce car-dominance.
2. **Residual skill/pace entanglement.** A thin-connectivity problem (narrow era): backmarkers absorb
   car-badness into "skill." **Largely resolved by widening the era (step 4)** — one connected
   component de-confounds the latents. Still present for any driver who never switches teams.
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

Steps 4–6 reproduced a car-leaning split (P(car>driver)=73%, CrIs overlap) with believable driver
arcs and honest intervals. Next: ~~session-matched quali normalization~~ [done, 8a]; ~~model **race
pace** directly as a second signal~~ [done, see below]; add a driver-error-DNF risk term.

### Race pace as a second signal (done)

`build_race_pace.py` → `fit_skill_joint.py` add race pace alongside qualifying as a **joint model**
of two correlated driver abilities (`quali_skill`, `racecraft`; soft LKJ link, two pace terms
`pace_q`/`pace_r`, race gets a larger σ). `build_scm_data.py` gained `--skill-source
{quali,race,combined}` / `--pace-source {quali,race}` (default **race**, since `finish_pos` is a race
outcome) and auto-detects a joint idata (`"racecraft" in posterior`); `uncertainty_propagation.py`
gained `--var-skill/--var-pace`. Node names and the 5-node DAG are unchanged, so attribution is
directly comparable. **Finding:** race pace attributes *more to the car* than qualifying — feeding
`racecraft`/`pace_r` gives car median **31.1%** / driver **13.9%**, **P(car>driver)=100%** (vs the
quali-based 73%); on Sundays the machinery is the decisive factor. (Quali-sourced from the same joint
model is near-parity, ~25/28.) Caveat: the racecraft signal is lapped-car sensitive (see PR-2 / the
joint report's merchant-table caveat).

### Confounder in the graph + ICC demotion (Pearl review, done)

Two "specification + reporting" fixes from a Pearl-style review ([`IDEAS.md`](IDEAS.md)), in
`attribution_v2.py`: (1) the roots are correlated (good drivers → good cars), so we add the hiring
edge `driver_skill → car_pace` (`--hiring-edge`, default on) — putting the confounder in the graph;
(2) ICC is demoted to a caveated descriptive number and the report now leads with graph-**robust**
measures: the interventional `do()`-sweeps, a rung-3 **necessity** query (`necessity_query`: "would
this podium have happened but for the car / the driver?", computed on the separable independent-roots
graph), and OLS. **Finding:** ICC assumes independent root noise, so the edge swings it **~25pp and
flips it** (car 26%/driver 16% → car 1%/driver 58%), while the interventional spread moves 0.0
positions — the old "car dominates by ICC" was partly an independence artifact. Graph-robust verdict
(wide era): near-parity, car slightly ahead (spread car 10.6 vs driver 10.1; podium needs car 82% vs
driver 68%; OLS pace 0.47 > skill 0.35). `uncertainty_propagation.py` gained a matching `--hiring-edge`
flag (default off) + an ICC-is-demoted caveat.

### Errors-in-variables de-attenuation (done)

`attribution_eiv.py` corrects the attenuation from feeding *estimated* latents into the finish
regression as if exact. Using the posterior draws to estimate the measurement-error covariance Σ_uu
and applying the multivariate correction `β_corr = (Σ_WW − Σ_uu)⁻¹ Σ_WY`. **Finding:** the driver
latent is noisier (reliability 0.78 vs car 0.93), so de-attenuation raises the *driver* — standardized
betas 0.35/0.47 → 0.40/0.40 (parity). It corrects a real bias but doesn't rescue the car; it confirms
car ≈ driver on the wide era. (Corrects the linear seam only; the gcm measures would shift the same
way.)

### Cross-era comparison ("Senna in a modern Red Bull")

The query is already expressible as `do(car_pace = 2024-Red-Bull, driver_skill = Senna)` → predict
finish; the architecture supports its *shape*. The blockers are not plumbing:
1. **Teammate chain must connect the eras** — **CONFIRMED (Step 7):** the connectivity sweep shows
   Senna joins the modern grid's component at era-start ≤1994; 1984–2025 keeps 90% in one component.
2. **Quali-format change at 2006** — pre-2006 is single-session `qualifying_time_millis`; 2006+ is
   knockout best-of-q1/q2/q3. **Largely addressed by Step 8a** (`--gap-method session`): a
   session-relative gap removes the within-knockout track bias and treats pre-2006 as one session, so
   the two formats are computed consistently. Residual: knockout = best-of-N sessions vs one single
   lap (8b's era-varying noise absorbs the scale).
3. **Cross-era scale comparability** — a 0.3% teammate gap may not mean the same thing in 1990 vs
   2024 (field spreads, tire wars, refueling). Needs an era-varying skill spread; only partly identified.
4. **Off-support counterfactual** — Step 6's interval machinery is the prerequisite; the cross-era
   query is an extrapolation reported with a wide CrI and a "what the model implies, not an identified
   effect" caveat.

Staged path: (A) uncertainty machinery [done, Step 6] → (B) walk era back / connectivity [done,
Step 7] → (C) session-consistent quali normalization [done, Step 8a] + era-varying skill spread
[8b] → (D) a thin cross-era counterfactual wrapper [8c].

## 13. Interactive site (`site/`)

A fully static, browser-side layer over the fitted models — no runtime Python. Live at
**https://bolt-chaos.github.io/apex-attribution/**.

- **Export** — [`scripts/export_site.py`](../scripts/export_site.py) runs locally (models are
  gitignored) and writes ~300 KB of JSON to `site/public/data/`: downsampled posterior draws per
  driver/car, a precomputed `E[finish]` mesh over `(driver_skill, car_pace)` (built via `exp_finish`
  at high sample count for a smooth surface), per-era ICC shares + interventional spreads, the
  cross-era legend draws + their own (1988-model-scale) mesh, and the full shared-teammate graph.
  Schema: [`site/DATA.md`](../site/DATA.md). The committed JSON is the source of truth for CI.
- **Why static works** — every interactive is a lookup into posterior draws or a smooth function of
  `(skill, pace)`: car-swap = bilinear interpolation of the mesh + posterior draws → credible band;
  era slider / cross-era / H2H / career arcs are all closed-form over the shipped draws. Two scales
  are kept strictly separate (main = joint racecraft/pace_r; cross-era = 1988 sess_rw skill).
- **App** — Vite + React + TypeScript in `site/`. `src/lib/` holds the reusable math (`mesh.ts`
  bilinear interp + band, `posterior.ts` P(A ahead), `graph.ts` BFS pathfinding); `src/components/`
  one file per interactive. d3-force lays out the teammate network once (static positions).
- **Deploy** — [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds `site/` and
  publishes to GitHub Pages on pushes touching `site/**`. No Python in CI.
