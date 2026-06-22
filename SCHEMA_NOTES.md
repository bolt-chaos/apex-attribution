# SCHEMA_NOTES.md — f1db schema mapping for causal driver-vs-car attribution

**Data source:** [f1db](https://github.com/f1db/f1db), CC-BY-4.0.
**Version pinned:** `v2026.7.0` (asset `f1db-sqlite.zip` → `data/f1db.sqlite`, 73 MB).
**Refresh:** `python scripts/download_data.py [--tag vX] [--force]` (idempotent, caches by version tag in `data/f1db.version`).

The db has **31 tables**. f1db uses **string slug ids** for entities (`driver_id='max-verstappen'`, `constructor_id='red-bull'`, `circuit_id`, `engine_manufacturer_id`, `tyre_manufacturer_id`) and **integer ids** only for `race.id`. There is no surrogate integer PK for drivers/constructors — the slug *is* the key.

---

## The one table that matters most: `race_data` (186,123 rows)

`race_data` is a **long/EAV-style table**: one row per (race, session-type, driver). A `type` discriminator column says which session the row describes. Columns are namespaced by session (`qualifying_*`, `starting_grid_position_*`, `race_*`, `fastest_lap_*`, `pit_stop_*`, `practice_*`), and only the columns for that row's `type` are populated.

Distinct `type` values (counts):

| type | rows | use |
|---|---|---|
| `RACE_RESULT` | 27,445 | **primary outcome table** — finishing pos, grid, points, retirement reason |
| `QUALIFYING_RESULT` | 26,866 | qualifying pace / grid-setting session |
| `STARTING_GRID_POSITION` | 25,682 | grid (also mirrored on RACE_RESULT rows, see below) |
| `PIT_STOP` | 22,293 | per-stop rows |
| `FASTEST_LAP` | 16,999 | fastest-lap session |
| `SPRINT_RACE_RESULT` | 546 | sprint outcomes (deferred for v1) |
| (practice / Q1 / Q2 / sprint-quali variants) | — | not needed for v1 |

> **Key convenience:** the `RACE_RESULT` row already carries the driver's grid slot
> (`race_grid_position_number`), points (`race_points`), and retirement reason
> (`race_reason_retired`) — so the MVP outcome dataframe can be built from a single
> `type='RACE_RESULT'` slice without joining the separate `STARTING_GRID_POSITION` rows.
> Qualifying *pace* (millis) still requires joining the `QUALIFYING_RESULT` slice.

---

## Entity → real column mapping (the entities the project needs)

| Concept we need | Table | Column(s) | Notes |
|---|---|---|---|
| driver id | `race_data` | `driver_id` (slug) | join `driver.id` for names/DOB |
| constructor id (car proxy) | `race_data` | `constructor_id` (slug) | join `constructor.id` |
| engine | `race_data` | `engine_manufacturer_id` | available per row if we want an engine node later |
| tyre supplier | `race_data` | `tyre_manufacturer_id` | spec-tyre era → near-constant; likely dropped |
| season / year | `race` | `year` | `race_data.race_id → race.id` |
| round | `race` | `round` | |
| circuit id | `race` | `circuit_id` | join `circuit.id` |
| **circuit_type** (exogenous) | `race` | `circuit_type` | values: `RACE` (permanent, 854), `STREET` (234), `ROAD` (83) |
| finishing position | `race_data` | `position_number` (INT, **NULL if unclassified**) | `type='RACE_RESULT'` |
| finishing pos (raw) | `race_data` | `position_text` | `'1'..'20'`, or `'DNF'`,`'DNQ'`,`'DNS'`,`'NC'`,`'DSQ'` |
| **grid position** | `race_data` | `race_grid_position_number` | on the RACE_RESULT row; pit-lane starts can be NULL/0 |
| **points** | `race_data` | `race_points` (DECIMAL) | candidate alt outcome target |
| **DNF / classification** | `race_data` | `position_text` + `race_reason_retired` | see reliability mapping below |
| qualifying pace | `race_data` | `qualifying_time_millis`, `qualifying_q1/q2/q3_millis` | `type='QUALIFYING_RESULT'` |
| weather | — | **NOT in schema** | f1db has no weather table → **drop the `weather` node for v1** (prompt allowed this) |

Supporting dimension tables: `driver`, `constructor`, `circuit`, `engine_manufacturer`,
`season_entrant_driver` (year × entrant × constructor × driver lineups, incl. `rounds` and a
`test_driver` flag — useful for filtering reserve/one-off drivers).

---

## DNF censoring → the `reliability` node (pitfall #3)

`position_number IS NULL` ⇔ not classified. `race_reason_retired` (free text, populated only on
retirements) lets us split a DNF into **machine failure** vs **driver-caused**, which is exactly the
distinction the `reliability` node needs. Observed reasons (2014+, by frequency) bucket as:

- **Mechanical / reliability (not the driver):** Engine, Power unit, Gearbox, Brakes, Suspension,
  Hydraulics, Electrical, Oil leak/pressure, Power loss, Overheating, Wheel, Exhaust, Turbo,
  Water leak/pressure, Undertray, Fuel system, Battery, Clutch, Transmission, Mechanical, …
- **Driver-caused (a real driver outcome):** Collision, Collision damage, Accident, Accident damage,
  Spun off, Puncture(*ambiguous*).
- **Other / disqualification / withdrawal:** Withdrew, Illegal skid block wear, DSQ.

Plan: build a `dnf_cause ∈ {finished, mechanical, driver_error, other}` label via a keyword map
(stored as a reviewable dict in the prep script). A **mechanical** DNF feeds `reliability`, not the
driver's `race_execution`; we never charge an engine failure to the driver.

---

## Teammate-structure / connectivity finding (pitfall in scope step 2)

For the proposed v1 era **2022–2025**: all 10 constructors fielded ≥2 drivers each season (some 3–4
due to mid-season swaps), so the teammate-difference design is well populated.

Union-find over the teammate graph (edge = two drivers shared a constructor-season with ≥3 races
each) gives **3 connected components**, not one:

- **Main component: 25 drivers** ← this is the v1 cohort.
- Detached: **Haas/Sauber backmarker cluster** — `kevin-magnussen`, `nico-hulkenberg`,
  `valtteri-bottas`, `guanyu-zhou`, `mick-schumacher`, `gabriel-bortoleto` (6 drivers) never shared
  a car with the main grid in this window.

**Decision for v1:** restrict to the largest connected component (prompt step 2 explicitly requires a
single component). The 6 dropped drivers are logged, not silently excluded. Widening the season
window (e.g. 2018–2025) would likely merge them but reintroduces a regulation-era confound, so it's
deferred.

---

## gcm API notes — verified against the **installed** versions (pitfall: breaking changes)

Installed & pinned: `dowhy==0.14`, `pandas==3.0.3`, `numpy==2.4.6`, `scikit-learn==1.9.0`,
`networkx==3.6.1`, `scipy==1.15.3`, `statsmodels==0.14.6`.

Verified by importing `dowhy.gcm` in the venv:

- ✅ Top-level: `gcm.fit`, `gcm.intrinsic_causal_influence`, `gcm.interventional_samples`,
  `gcm.counterfactual_samples`, `gcm.arrow_strength`, `gcm.evaluate_causal_model`,
  `gcm.StructuralCausalModel`, `gcm.InvertibleStructuralCausalModel`, `gcm.ProbabilisticCausalModel`,
  and `gcm.auto.assign_causal_mechanisms`.
- ⚠️ **`gcm.falsify_graph` is NOT top-level in 0.14.** Import it as
  `from dowhy.gcm.falsify import falsify_graph` (also re-exported by `dowhy.gcm.model_evaluation`).
- Counterfactual queries need an **`InvertibleStructuralCausalModel`**, not the plain `StructuralCausalModel`.
- Signatures we'll rely on (0.14):
  - `intrinsic_causal_influence(causal_model, target_node, prediction_model='approx', ..., num_samples_randomization=250, num_samples_baseline=1000)`
  - `interventional_samples(causal_model, interventions: Dict[node, Callable], observed_data=None, num_samples_to_draw=None)`
  - `counterfactual_samples(causal_model, interventions: Dict[node, Callable], observed_data=None, noise_data=None)`
  - Interventions are `{node: lambda x: value}` callables (the `do(constructor=X)` swap).
- ⚠️ `pandas==3.0.3` is very new; watch for dtype/Copy-on-Write friction when feeding frames to gcm.

---

## Proposed v1 dataframe shape (`build_dataset.py`, phase b — NOT yet built)

One row per **classified-or-retired race entry** for drivers in the v1 cohort/era. Tentative columns:

| column | source | role in DAG |
|---|---|---|
| `year`, `round`, `race_id` | `race` | keys / fixed effects, filtering |
| `circuit_id` | `race.circuit_id` | key |
| `circuit_type` | `race.circuit_type` | exogenous root (`RACE`/`STREET`/`ROAD`) |
| `driver_id` | `race_data.driver_id` | categorical driver node |
| `constructor_id` | `race_data.constructor_id` | categorical constructor (car proxy) node |
| `engine_manufacturer_id` | `race_data` | optional engine node (v1: probably unused) |
| `grid` | `race_data.race_grid_position_number` | mediator (`car_pace`+`quali_exec`→grid) |
| `quali_millis` | join `QUALIFYING_RESULT.qualifying_time_millis` | qualifying-execution signal |
| `finish_pos` | `race_data.position_number` | **outcome** (NULL ⇒ DNF) |
| `points` | `race_data.race_points` | alt outcome |
| `classified` | `position_number IS NOT NULL` | censoring flag |
| `dnf_cause` | derived from `race_reason_retired` | drives `reliability` vs driver split |

Open question to resolve in phase (b): outcome target = **finishing position** (needs a DNF-handling
convention) vs **points** (already 0 for DNF, but zero-inflated and points-system-dependent). Leaning
toward modelling **classified finishing position with `reliability` as a separate censoring node**, per
pitfall #3.

> Estimated cohort size: ~25 drivers × ~10 teams × (2022–2025 ≈ 90+ races) ≈ a few thousand
> result rows after the connected-component + classification filters. Confirmed adequate for the
> small categorical DAG; exact N reported once the prep script runs.
