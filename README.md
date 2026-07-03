# apex-attribution

A **causal attribution metric for Formula 1 drivers** — separating a driver's own
contribution from the car they happened to be driving. Instead of a hand-weighted
box-score sum (à la basketball's PER), the "weights" fall out of a fitted **structural
causal model**, so we can pose the eternal F1 bar argument — *how much is the driver vs.
the machinery?* — formally, as a `do(constructor = X)` intervention holding the driver fixed.

Built on [DoWhy's `gcm` module](https://github.com/py-why/dowhy) over the
[f1db](https://github.com/f1db/f1db) open dataset (CC-BY-4.0).

New to the ideas here (causal inference, the teammate trick, Bayesian uncertainty)? Start with
[`CONCEPTS.md`](CONCEPTS.md) — a plain-language guide to every concept the project rests on.
For the full system map (data flow, both modelling lines, artifacts, reproduction) see
[`ARCHITECTURE.md`](ARCHITECTURE.md). To pick the project back up quickly (current state + next
action + reproduce commands) see [`RESUME.md`](RESUME.md). Running notes for the eventual write-up
(story, results, decisions, bugs, limitations) are in [`WRITEUP_NOTES.md`](WRITEUP_NOTES.md).

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

**Phase 4 — widen the era to 2018–2025 → car-dominance reproduced.** ✅ The narrow era fragments
into 3 teammate components; **2018–2025 is a single 41-driver component**, so car pace and driver
skill are properly de-confounded (e.g. Albon now spans Red Bull/Toro Rosso/Williams). Re-running the
*whole* pipeline on the wider era flips every metric to car-dominant:

| metric | v1 (categorical) | v2 narrow 2022–25 | **v2 wide 2018–25** |
|---|---|---|---|
| quali variance (systematic) | — | car 46% / driver 54% | **car 89% / driver 11%** |
| ICC car vs driver (race) | 1.3% vs 45% | 6.5% vs 31.6% | **36.5% vs 14.4%** ✅ PASS |
| OLS `finish ~ skill, pace` | — | 0.56, 0.20 | **0.38, 0.43** (car larger) |
| counterfactual (Sargeant→Ferrari) | flat | — | **P12→P4** |

This confirms the phase-3 diagnosis: the driver-heaviness was the **thin-connectivity artifact**,
and full teammate-graph connectivity fixes it. The era is a CLI parameter (`--start/--end/--out-tag`);
2018–2025 artifacts are tagged `_2018_2025`, the 2022–2025 baseline is preserved.
See [`figures/v2_attribution_diagnostic_2018_2025.png`](figures/v2_attribution_diagnostic_2018_2025.png),
[`figures/v2_driver_skill_2018_2025.png`](figures/v2_driver_skill_2018_2025.png).

**Phase 5 — time-varying (per-season) skill** ([`v2/fit_skill_rw.py`](v2/fit_skill_rw.py)). Replaces
the constant career skill with a per-driver **Gaussian random walk across seasons** (temporally
smooth = slowly varying). The data supports a small, real drift (**σ_rw ≈ 0.11%/season**, HDI
excludes 0), and the recovered career arcs are believable — **Verstappen** steadily improving
(−0.55→−1.01), **Vettel/Ricciardo declining**, **Norris/Piastri** maturing, **Hamilton** flat at his
plateau (see [`figures/v2_skill_trajectories_2018_2025_rw.png`](figures/v2_skill_trajectories_2018_2025_rw.png)).
**Car-dominance is robust** to the refinement: race ICC **car 26.6% vs driver 23.0% → PASS**
(interventional car 10.3 vs driver 7.5; counterfactual Albon Williams→Red Bull P13→P7). The split
moves toward parity vs constant-skill (was 36.5/14.4) because season-specific skill legitimately
captures form the career average flattened — a more accurate, more nuanced picture, car still ahead.

**Phase 6 — uncertainty on the current best model** ([`v2/uncertainty_propagation.py`](v2/uncertainty_propagation.py),
now generalized to the time-varying model). The wide+RW point estimate (car 26.6 / driver 23.0) is a
~3-point lead — too close to take on faith. Re-running ICC over 30 *joint* posterior draws gives the
honest picture: **car median 31.9% (90% CrI [23.3, 42.3]) vs driver 21.4% [12.9, 29.3]**, and
**P(car > driver) = 73%** (see [`figures/v2_uncertainty_2018_2025_rw.png`](figures/v2_uncertainty_2018_2025_rw.png)).
So *"the car leads"* is the **modal outcome (≈3 of 4 draws) but not decisive** — the credible
intervals overlap, so the right statement is a **range, not a verdict**: the car is *probably* the
bigger factor, but the data don't pin a confident "car dominates." This appropriately tempers the
point-estimate headline, and is the prerequisite for any future cross-era claim (which would carry
even wider intervals).

**Phase 7 — walk the era back (toward cross-era)** ([`v2/era_connectivity.py`](v2/era_connectivity.py)).
`build_quali.py` now also reads `qualifying_time_millis`, so the skill signal reaches back to **1980**
(pre-2006 single-session quali; 2006+ knockout). Three findings:

- **Connectivity reaches Senna.** A teammate-graph sweep shows the main connected component stays at
  **82–95% of drivers** all the way back to 1980. Extending to ≤1994 places **Senna in the same
  component as Verstappen/Hamilton**; the full 1984–2025 span keeps 217/242 (90%) connected. So
  connectivity is **not** the cross-era blocker. ([`figures/era_connectivity.png`](figures/era_connectivity.png))
- **The split is era-dependent — car-dominance sharpens as the window widens.** Re-running the whole
  pipeline at 2006–2025 (92 drivers, all-knockout era):

  | era | car (median, 90% CrI) | driver | P(car>driver) |
  |---|---|---|---|
  | 2018–2025 (4 yr) | 31.9% [23, 42] | 21.4% [13, 29] | 73% (overlapping) |
  | 2006–2025 (20 yr) | 43.6% [35, 48] | 12.4% [6, 15] | **100%** (separated) |

  A 20-year window spans huge car variation (multiple reg eras, dominant vs terrible cars), so the
  car explains more of the finishing-position variance; the converged cost-cap 2018–2025 cars give
  near-parity. **There is no single "X% driver / Y% car" — it depends on the window.**
- **Believable 20-year career arcs** ([`figures/v2_skill_trajectories_2006_2025_rw.png`](figures/v2_skill_trajectories_2006_2025_rw.png)):
  Hamilton peaks ~2016 then declines; Alonso's McLaren-Honda dip and Aston resurgence; Vettel's decline.

Remaining cross-era blockers (connectivity solved): the 2006 quali-format change (single-session vs
knockout best-lap — the `qt` fallback unifies the column but mixing the two needs care), era-varying
skill spread, and the off-support counterfactual. See [`ARCHITECTURE.md`](ARCHITECTURE.md) §12.

**Phase 8a — session-consistent quali normalization** ([`v2/build_quali.py`](v2/build_quali.py)
`--gap-method session`). The first cross-era blocker, and a real fix to current results. Instead of
each driver's gap to the race's overall (Q3) pole, measure their gap to the fastest lap **in the same
session** (min over the sessions they reached). This:
- **Removes a measured bias.** Q1-eliminated backmarkers were over-penalized **~0.8%** purely by
  track evolution (their Q1 lap vs the much-faster Q3 pole). Verified: backmarkers shrink ~1.26% vs
  frontrunners ~0.49%. Skill magnitudes become realistic (spread compresses ~1.7%→~1.0%; Latifi
  +0.76→+0.51) with the **ranking order preserved**.
- **Unifies the 2006 format boundary** (pre-2006 single-session reduces to one session) — the
  foundation cross-era needs.
- **Leaves the headline robust:** 2018–2025 attribution is car 26.8% / driver 24.6%, ≈ the pole
  method's 26.6/23.0.

Default stays `pole` (all prior baselines reproduce exactly). (Fit R-hat 1.02, marginally above
target — the session-normalized 4-yr window supports ~zero skill drift, a mild sampling funnel; the
longer-span fits in 8b have more drift to estimate.)

## Does it actually predict? (out-of-sample validation)

A model is only trustworthy if it predicts data it never saw. [`v2/backtest.py`](v2/backtest.py)
fits driver skills on **2018–2023** and predicts every **teammate qualifying head-to-head in the
held-out 2024–2025** — who out-qualifies their teammate, and by how much. Teammates share the car, so
car pace cancels and the gap is a pure skill difference (no future car data needed). Results:

- **Head-to-head accuracy: 67%** race-level, **80%** season-long (vs a 50% coin-flip). The confident
  calls (Verstappen > Tsunoda, Alonso > Stroll, Norris > Piastri) are right; the genuinely close
  pairs are coin-flips — exactly as honest uncertainty should behave.
- **Correlation 0.40** between predicted and actual gaps; beats a predict-nothing baseline on error.
- **Calibration:** the model's credible intervals cover reality at **50%→74%, 80%→92%, 90%→97%** —
  well-behaved but slightly *conservative* (intervals a touch too wide, i.e. the model modestly
  over-states its own uncertainty). See [`figures/backtest.png`](figures/backtest.png).

This is the validation that turns "a model that fits the past" into "a model that predicts the
future" — and it does, for the part it claims to measure (relative driver pace).

## Forecasting (Phase B): project the skills forward

[`v2/predict.py`](v2/predict.py) turns the validated capability into a forward-looking forecast.
Skills are carried from each driver's last trained season to a target season via the **fitted
random walk** (a martingale — same expected skill, but the uncertainty widens by
`sqrt(forward_seasons) * sigma_rw`, an honest "we know less about next year" penalty). It emits:

- a **projected driver-skill power ranking** (pure qualifying pace, fastest first, with 90% CrIs)
  — Verstappen clear #1, then Leclerc/Norris/Sainz/Piastri; and
- **per-team teammate H2H predictions** — expected gap + `P(faster driver out-qualifies teammate)`,
  using the same posterior-draw + StudentT race-noise machinery the backtest calibrated. Default
  line-ups are each constructor's last-season pairing; pass `--lineup pairs.json` for a hypothetical
  grid (e.g. two aces in one car). 2026 projection: Verstappen > Tsunoda (P=78%), Alonso > Stroll
  (P=68%), the genuinely close pairs (Norris/Piastri, Bortoleto/Hülkenberg) near coin-flips — exactly
  the honest behaviour the backtest showed. See [`figures/predict_2026.png`](figures/predict_2026.png).

Only teammate H2H is claimed (car cancels, no future car data needed); absolute finishing position
would require the unbuilt target-season cars, so it is not.

**Over-/under-rated drivers** ([`v2/insights.py`](v2/insights.py)) — the fun question the model
uniquely answers: *is a driver flattered or robbed by their machinery?* For each driver it contrasts
their **actual** average finish (reputation, car included) with where the SCM says they'd finish on
their own skill in an **average car** (`do(car_pace = median)`). The gap is the car effect in
finishing positions: Hamilton/Piastri/Bottas/Pérez are *flattered* (dominant cars worth ~+4 places),
while Albon/Sargeant/Bortoleto are *held back* by weak machinery. A pace-vs-results table keeps it
honest ("over-rated" means results overstate the driver's share, not that they lack skill — Hamilton
is flattered yet still pace-rank #5). See [`figures/insights_over_under_2018_2025.png`](figures/insights_over_under_2018_2025.png).

## "Senna in a modern Red Bull" (Phase B, illustrative only)

The crowd-pleaser — and the most caveated thing in the repo ([`v2/cross_era.py`](v2/cross_era.py)).
The obstacle is the **era scale**: a driver's skill is a % qualifying gap to the field, but the
field's *spread* has collapsed (grid SD ~0.71% in 1988–94 vs ~0.25% in 2018–25 as cars and drivers
converged), so a raw −1.3% gap in 1990 is **not** the same achievement as −1.3% in 2024. We make the
assumption visible by showing two translations side by side:

- **naive** (raw skill, no adjustment) — meaningless, shown only to demonstrate why: it scatters the
  legends from P2.5 to P11.4 in the same car; and
- **era-normalized** (z-score) — how many SDs the legend led *their* field, mapped onto the modern
  field's spread. Senna was **−2.18 SD** ahead → modern-equivalent skill ≈ −1.59%, a hair past
  Verstappen's 2024 −1.51%.

Fed (with the real `red-bull@2024` car pace) through the modern race-outcome SCM, the era-normalized
greats all land **≈ P3–3.6**, right alongside Verstappen's P3.4 — *utterly dominant but not
physically impossible.* **Caveats are loud and in the output:** this is an off-support extrapolation,
not an identified effect; cross-era skill rests on the era-scale assumption; and the source skill
model is at R-hat 1.04 (not fully converged). The honest answer to the bar argument is "a
front-running, title-winning combo — with an asterisk." See [`figures/cross_era.png`](figures/cross_era.png).

## Two abilities: qualifying pace vs racecraft (race pace as a 2nd signal)

So far skill was identified from **qualifying only**. [`v2/build_race_pace.py`](v2/build_race_pace.py)
adds a **race-pace** signal (`race_pct_gap` = % time deficit to the winner; same teammate-cancels
logic, noisier), and [`v2/fit_skill_joint.py`](v2/fit_skill_joint.py) fits a model where each driver
has **two correlated latent abilities** — `quali_skill` (Saturday) and `racecraft` (Sunday) — linked
by a soft LKJ prior so each is identified by its own data (the random-walk + teammate machinery,
twice). Race pace gets its own larger noise term, so it's a complement, not a replacement.

Result (2018–2025, **R-hat 1.01 — converged**): the two abilities are **highly correlated,
`rho = +0.92`** (90% CrI [0.82, 0.99]) — one-lap pace strongly predicts race pace — but with a small
real residual. The per-driver `racecraft − quali_skill` **"qualifying-merchant" delta** has credible
mid-grid signal: **Pérez races better than he qualifies** (quali ~P14 pace → racecraft top-7),
Tsunoda/Bottas/Albon read as quali specialists. **Honest caveat (in the output):** the deltas are
small vs noise and **lapped-car sensitive** (a `--drop-lapped` sensitivity moves the racecraft
ranking, Spearman ~0.78); the backmarker tail is partly race-gap *compression* (a lapped car can't
lose more), not pure racecraft, and even Verstappen's "races better" softens when lapped cars are
dropped. See [`figures/v2_skill_joint_quali_vs_race_2018_2025_joint.png`](figures/v2_skill_joint_quali_vs_race_2018_2025_joint.png).

**Does racecraft predict?** [`v2/backtest_race.py`](v2/backtest_race.py) repeats the out-of-sample
test for race pace: fit on **2018–2023**, predict held-out **2024–2025** teammate race-pace H2H.
Result: **race-level 52%** (barely above coin-flip — single-race pace is dominated by strategy/traffic
noise), but **season-long 60%** and **correlation 0.46** (above even the quali backtest's 0.40)
confirm the signal is real, just noisier. Every confident call lands — Verstappen ≫ Tsunoda/Pérez,
Alonso > Stroll, Norris > Piastri — and the misses are genuinely close pairs. So racecraft is a real,
predictive-but-noisy signal: trust it season-long, not race-by-race. See [`figures/backtest_race.png`](figures/backtest_race.png).

**Threading it into the attribution** ([`v2/build_scm_data.py`](v2/build_scm_data.py) `--skill-source
race`) shows race pace attributes **more to the car** than qualifying: car median **31.1%** / driver
**13.9%**, **P(car>driver)=100%** (vs the quali-based 73%) — on Sundays the machinery is decisive.

## Incident-proneness: how often a driver throws a result away

Speed isn't the whole of a driver — *not binning the car* is a skill too. The DNF taxonomy already
splits retirements into `mechanical` (charged to the car, never the driver) and `driver_error`
(collision / accident / spun off). [`v2/fit_incident.py`](v2/fit_incident.py) models the per-driver
**driver-error-DNF risk** as a hierarchical logistic (`logit p = alpha + incident[driver] +
hazard[circuit_type]`), with **partial pooling** so rare/short-career rates shrink toward the field
and circuit hazard is controlled (street circuits crash ~30% more).

The headline is the **shrinkage**: raw rates look dramatic (Grosjean 15.8%, Latifi 11.5% vs Norris
2.6%), but once small-sample noise is corrected the credible per-start risks compress to a narrow
**~5.4–7.0%** band with heavily overlapping CrIs — *incident-proneness is real and robustly ordered
(Grosjean/Latifi/Albon worst; Norris/Hamilton/Alonso/Verstappen cleanest) but a smaller effect than
raw rates suggest.* R-hat 1.00. See [`figures/v2_incident_proneness_2018_2025.png`](figures/v2_incident_proneness_2018_2025.png).
(Caveat: attributed to the driver controlling for circuit only — a small car-induced-error component
may fold in.) Per-driver rates are saved to `models/incident_rates_2018_2025.json` for the unified
expected-result metric.

**Unified expected-result metric** ([`v2/unified_metric.py`](v2/unified_metric.py)) folds both DNF
routes into one number: `E_all = (1 − p_mech − p_inc)·E[finish|classified] + (p_mech + p_inc)·DNF`
(combined at reporting only — Option A). It exposes the **incident tax** (positions/race lost to a
driver's own errors), which is *stakes-dominated*: because shrunk incident rates barely vary, the tax
tracks how high a driver would finish, not how often they crash — Verstappen loses the most per race
(~0.9 pos) to a *low* crash rate × a costly fall from P3. A **cleanliness dividend** isolates the
proneness itself (positions saved vs a field-average driver in the same seat): Norris/Hamilton save
the most (~0.06 pos/race), Grosjean/Sainz lose the most (~0.05) — real but tiny, so incident-proneness
is a modest tiebreaker, most valuable when it protects a front-runner's podium. See
[`figures/v2_unified_metric_2018_2025.png`](figures/v2_unified_metric_2018_2025.png).

## Putting the confounder in the graph — and why ICC is no longer the headline

A [Pearl-style review](IDEAS.md) flagged two "specification + reporting" fixes, now implemented in
[`v2/attribution_v2.py`](v2/attribution_v2.py):

1. **Specification.** `driver_skill` and `car_pace` were drawn as *independent* roots, but they're
   correlated (~0.5) because good drivers are hired into good cars. We put that confounder *in the
   graph* via the hiring edge `driver_skill → car_pace` (`--hiring-edge`, default on).
2. **Reporting.** Demote ICC (the variance share) from headline to a descriptive number, and lead
   with the graph-**robust** measures.

**Why it matters — the striking result:** ICC assumes *independent* root noise, so adding the hiring
edge **swings the split ~25pp — from car 26% / driver 16% to car 1% / driver 58%, flipping the
verdict** — while the interventional car-vs-driver spread moves **0.0 positions**. The `do()`-sweeps
and counterfactuals set *both* roots, so they're invariant to how the confounding is modelled; only
the variance share isn't. So the earlier "car dominates X% by ICC" headline was partly an
independent-roots **artifact**.

**The graph-robust answer (wide era, race pace):** by every measure that doesn't depend on the
confounding spec, the car and driver are **near-parity, car slightly ahead** — interventional car
**10.6** vs driver **10.1** positions; OLS `finish ~ pace 0.47 > skill 0.35`; and a new **rung-3
necessity query** ("would this podium have happened *but for* the car / the driver?") says a podium
needs the **car 82%** vs the **driver 68%** of the time (Räikkönen/Bottas/Piastri most car-dependent;
Alonso/Verstappen most driver-dependent). See [`figures/v2_necessity_2018_2025_joint.png`](figures/v2_necessity_2018_2025_joint.png).
ICC is kept only as a clearly-caveated descriptive number.

**Errors-in-variables — correcting the estimated-latents bias** ([`v2/attribution_eiv.py`](v2/attribution_eiv.py)).
`driver_skill`/`car_pace` are posterior *estimates* fed into the finish regression as if exact —
a measurement-error problem that **attenuates** the noisier latent's coefficient. Using the posterior
draws to estimate the error covariance and applying the standard multivariate correction: the
**driver** latent is the noisier one (reliability **0.78** vs the car's **0.93**), so de-attenuation
raises the *driver* — standardized `finish ~ skill+pace` goes from naive **0.35 / 0.47** to corrected
**0.40 / 0.40 (parity)**. So the attenuation was suppressing the *driver*, not the car; correcting it
confirms car ≈ driver rather than rescuing the car. See [`figures/v2_eiv_2018_2025_joint.png`](figures/v2_eiv_2018_2025_joint.png).

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
