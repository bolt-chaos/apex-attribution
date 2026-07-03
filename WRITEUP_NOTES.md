# Write-up notes

Running scratchpad of the things worth putting in the eventual write-up / report / college essay.
Add to it as the project grows; Phase C (`WRITEUP.md`) expands these bullets into prose.
Keep it honest and specific (numbers + which figure), because specificity is what reads as real.

## One-line pitch
Built a causal + Bayesian model to settle the eternal F1 argument — *how much of a result is the
driver vs. the car?* — and, just as importantly, figured out exactly when it can and can't answer.

## The story in five beats (the narrative arc)
1. **The question.** "Is it the driver or the car?" Points/wins can't tell you, because the best
   drivers are *in* the best cars (they're confounded).
2. **The naive attempt fails — informatively.** A causal model with raw driver/car labels gives a
   backwards answer (says the *driver* explains ~45% of the result and the car ~1%). Diagnosed why:
   each driver is locked to ~one car, so the model can't tell their effects apart and dumps the
   car's pace onto the driver's name.
3. **The fix.** A hierarchical Bayesian model that uses **teammates** (who share a car) to separate
   driver skill from car pace as two distinct hidden quantities, then feeds those into the causal
   model. Now it reproduces the known answer: the **car dominates** the modern era.
4. **Honesty about the answer.** Quantified the uncertainty (the split is a *range*, not a number)
   and discovered it's **era-dependent** — over a long span with big car differences the car
   dominates more; in the converged modern era it's closer.
5. **The limits.** Tried to compare across eras ("Senna in a modern Red Bull") and hit a *fundamental*
   identification wall — the same wall pro analysts hit (baseball cross-era, chess Elo). Then
   **validated** the model predicts genuinely-unseen races out-of-sample.

## Headline results (with numbers)
- **Naive model is backwards:** driver 45% vs car 1.3% of finishing-position variance — a clear
  failure of the project's own sanity check (the car should dominate). Cause: driver↔car statistical
  dependence is near-total (**Cramér's V = 0.84**). (`figures/attribution_diagnostic.png`)
- **Fixed model reproduces car-dominance:** wide-era ICC **car 36.5% vs driver 14.4%**.
  (`figures/v2_attribution_diagnostic_2018_2025.png`)
- **With honest uncertainty:** car median 31.9% (90% CrI [23, 42]) vs driver 21.4% [13, 29];
  **P(car > driver) = 73%** — "car leads" is probable, not certain. (`figures/v2_uncertainty_2018_2025_rw.png`)
- **Era-dependent:** 2006–2025 (20 yr) gives car 44% / driver 12%, P=100%; the longer the window,
  the more car variation it contains, the more the car dominates. *There is no single "X%/Y%."*
- **Believable driver ranking & career arcs:** Verstappen #1, then Leclerc/Norris/Russell/Hamilton…
  with arcs that match reality — Hamilton peaks ~2016 then declines, Alonso's mid-career McLaren-Honda
  dip and late Aston resurgence, Vettel's decline. (`figures/v2_skill_trajectories_2006_2025_rw.png`)
- **Out-of-sample validation:** trained on 2018–2023, predicted held-out 2024–2025 teammate
  qualifying battles at **67% (race) / 80% (season-long)** vs a 50% coin-flip; correlation 0.40;
  intervals calibrated but slightly conservative (50%→74%, 90%→97%). (`figures/backtest.png`)

## Techniques & skills demonstrated (the "what I learned" list)
- **Causal inference:** structural causal models, `do()` interventions, counterfactuals (DoWhy `gcm`);
  the difference between *correlation* and a *causal* "what if we swapped the car" query.
- **Bayesian hierarchical modeling (PyMC):** partial pooling, a Gaussian **random walk** for skill
  that drifts over a career, non-centered parameterization, a robust **Student-t** likelihood,
  a sum-to-zero constraint for identifiability, per-era noise.
- **Identification / confounding reasoning:** selection bias (good drivers → good cars), collinearity/
  nesting, why a variance decomposition can be unidentified, and how a *design* (teammates) fixes it.
- **Uncertainty quantification:** posteriors, credible intervals, propagating uncertainty into a
  downstream metric, and **calibration** (are the error bars honest?).
- **Model validation:** out-of-sample backtesting against a baseline — the thing that separates
  "fits the past" from "predicts the future."
- **Graph theory:** modeling the grid as a teammate graph; connected components decide whether two
  drivers are even comparable (`networkx`).
- **Data engineering & reproducibility:** schema introspection of a real SQLite dataset, a
  CLI-parameterized pipeline, version-pinned environment, git + pull-request workflow.

## Key decisions & the reasoning (the judgment calls)
- **Use qualifying pace as the skill signal**, not race results: it's the cleanest car-equalized
  measure (one lap, less luck/strategy/traffic).
- **Teammates as the identification lever**: same car, so their gap is (mostly) pure skill; drivers
  switching teams chain the comparisons across the grid.
- **Never charge a mechanical DNF to the driver** ("Option A"): a blown engine is the car, not the
  driver — modeled reliability separately.
- **Session-relative qualifying gaps**: compare each driver to the fastest lap *in the same session*,
  not the overall pole (see bug/insight below).
- **Widen the era** to de-confound driver from car (more team-switching → cleaner separation).
- **Report a range, not a number**, once it was clear the split is genuinely uncertain & era-dependent.

## Bugs, gotchas & things I had to debug (rigor — this stuff is GOOD essay material)
- **Teammate pair-ordering bug (backtest).** When tallying head-to-heads I treated "A vs B" and
  "B vs A" as different pairs, which split each rivalry in two with opposite signs and scrambled the
  result. Canonicalizing the pair order fixed it — and the predicted-vs-actual **correlation jumped
  0.22 → 0.40**. *Lesson: canonicalize identifiers before aggregating pairwise data.*
- **Student-t scale-vs-standard-deviation bug (calibration).** I scaled the prediction noise by
  `sqrt(ν/(ν−2))`, confusing a Student-t's *scale* parameter with its *standard deviation*. With
  ν≈2–3 this hugely over-inflated the intervals, so calibration looked absurd (a 50% interval covered
  94%). Using the scale directly gave sensible coverage (50% → 74%). *Lesson: know whether a
  distribution's parameter is a scale or an SD — for Student-t they differ by `sqrt(ν/(ν−2))`.*
- **The qualifying-format trap.** Pre-2006 qualifying is a single session (`qualifying_time_millis`);
  2006+ is the knockout Q1/Q2/Q3. The first version read only Q1/Q2/Q3, which would have **silently
  dropped every race before 2006.** Caught it by checking data availability *by decade* before
  trusting the build. *Lesson: probe your data's coverage before you model it.*
- **Track-evolution bias in qualifying.** Comparing a Q1-eliminated driver's lap to the much-faster
  Q3 pole over-penalized backmarkers by **~0.8%** (as big as the whole driver-skill spread!), purely
  because the track speeds up through the session. Fixed with the session-relative gap.
- **"Doohan in a McLaren."** The attribution metric secretly generated impossible driver×car pairings
  (it assumed driver and car are independent), which biased the answer. Understanding this required
  reading *how the metric computes its number*, not just looking at the output.
- **Convergence (R-hat).** Some Bayesian fits flirted with non-convergence (R-hat ~1.02–1.04),
  especially when a parameter was near zero (a "funnel"). Knowing to *check* R-hat and not trust an
  unconverged fit is itself a skill.
- **Library reality:** `gcm.falsify_graph` moved namespaces between versions; pandas 3.0 needed an
  extra parquet engine; no NetCDF backend → pickle the model. Small, but real "make the tools work."

## Honest limitations (often the STRONGEST part of a science essay)
- **Cross-era comparison is only partly identifiable.** You never observe an old and a modern driver
  in the same car, and the chain linking them is thin and noisy → old-era legends get *shrunk* toward
  average (Prost ranked #46). This isn't a bug; it's a known hard problem (cross-era baseball, chess
  Elo inflation). Recognizing and reporting it is the scientifically mature move.
- **The split is era-dependent**, so any single "driver vs car %" is incomplete without the window.
- **Selection confounding** (driver↔car) is never fully removed, only mitigated by the teammate design.
- **Counterfactuals like "Senna in a Red Bull" are extrapolation** beyond observed data — illustrative,
  not an identified effect; reported with a wide interval and a caveat.
- **Calibration is slightly conservative** — the model modestly over-states its own uncertainty.

## How a causal-inference purist (Judea Pearl) would critique this — and the roadmap it implies
*(A steelman of Pearl's likely reaction, not a quote. This is strong essay material: it situates the
project in the field's rigorous frame and shows you know its frontier. His criticisms are mostly of
the form "you found the right problems by hand — now formalize them with the tools built for exactly
this," which is praise disguised as a to-do list.)*

**What he'd praise.** It climbs his **Ladder of Causation** — most "who's best" analysis is rung-1
association (win counts); this reaches rung 2 (`do(constructor=X)`) and rung 3 (counterfactuals). It
**commits to an explicit DAG** (assumptions drawn, not hidden) and even runs `falsify_graph`. And the
v1 **non-identification finding** is his central thesis in miniature — no amount of data/ML separates
driver from car when each driver sits in one car; identification is a property of *structure*, not
sample size. He'd applaud that the project refuses to trust a non-identified number.

**What he'd criticize (each = a concrete improvement):**
- **It's really a *positivity/overlap* failure, not just confounding.** With each driver nested in ~one
  car (Cramér's V 0.84), `do(car = Red Bull)` asks about a driver never observed near a Red Bull — no
  overlap. The v2 fixes (team-switchers, wider era) are *overlap-improvement* strategies; frame them
  that way.
- **The associational→causal seam.** Skill/pace are fit by a rung-1 regression, then fed into the SCM
  as if they were clean manipulable causes. What does `do(car_pace = −1.5%)` physically intervene on?
  It's a fitted construct, and skill/pace are still entangled (corr 0.49–0.54).
- **ICC variance-shares aren't structural — and the era-dependence proves it.** "Car 32% / driver 21%"
  is a population-specific variance decomposition, so of course it shifts with the era window. Pearl
  would read the era-dependence as a *symptom* of reporting a non-transportable summary, and push toward
  effect/mediation measures that are more invariant.
  **✅ IMPLEMENTED (`v2/attribution_v2.py`, the two "spec + reporting" fixes):** (1) put the confounder
  in the graph via the hiring edge `driver_skill→car_pace` (`--hiring-edge`, default on); (2) demote ICC
  to a descriptive number and lead with graph-robust measures. The payoff is the strongest possible
  vindication of the criticism: **adding the edge swings the ICC split ~25pp — from car 26%/driver 16%
  to car 1%/driver 58%, flipping the verdict — while the interventional spread moves 0.0 positions.**
  So the old "car dominates by ICC" headline was largely an independent-roots artifact. By every
  *graph-robust* measure the car and driver are near-parity on the wide era (car 10.6 vs driver 10.1
  positions; a **rung-3 necessity query** says a podium needs the car 82% vs the driver 68%; OLS pace
  0.47 > skill 0.35). Figure: `figures/v2_necessity_2018_2025_joint.png`.
- **`grid` is a mediator with no formal mediation analysis.** "How much of a driver's edge is Saturday
  (grid) vs Sunday (racecraft)?" is a textbook natural-direct/indirect-effect decomposition (his
  mediation formula); the quali-vs-racecraft split answers it only informally.
- **Cross-era = a *transportability* problem.** The "Senna" z-score era-normalization is exactly what
  Pearl & Bareinboim formalized with **selection diagrams** and transport formulas. He'd call the
  z-score an ad-hoc patch and want the era gap modeled structurally.
- **DNF/incident selection lives outside the graph.** "Option A" (combine at reporting) is pragmatic;
  the principled route is a **selection node** in the DAG + recoverability-from-selection-bias theory.
- **Causal sufficiency & sensitivity.** Likely unmeasured common causes (team budget → car pace *and*
  hiring *and* strategy; weather/tires omitted). He'd want an explicit **sensitivity analysis to
  unmeasured confounding**, not just an acknowledgment.
- **Identifiability found empirically, not proven.** Run the **ID algorithm / do-calculus** on the graph
  up front to certify whether `E[finish | do(car_pace)]` is identifiable *before* fitting.

**Concrete next steps this implies:** add a driver→team **assignment/selection node** + check positivity;
run the **ID algorithm** to certify the query; replace ICC with a **mediation decomposition** through
`grid`; recast cross-era as **transportability** with selection diagrams; put **DNF selection in the
graph**; add a **confounding sensitivity analysis**.

## Fun results & hooks (to engage a reader)
- "Senna in a modern Red Bull" — the dream query, shipped as an explicitly playful, caveated demo.
- Driver career arcs that match what fans remember (Hamilton's peak, Alonso's resurgence, Vettel's fade).
- "Most over/under-rated driver" — where model skill disagrees with reputation/results.
- The teammate-cancellation trick: because teammates share a car, their gap is *pure* skill — an
  elegant idea that makes the validation clean.

## Tips for the actual essay/report
- **Lead with the question and the naive failure** — the "the obvious approach gives a backwards
  answer, and here's why" hook is gripping and shows insight.
- **Make the validation and the limits the centerpiece**, not the fanciest model. "I tested it on data
  it never saw, and I know exactly where it breaks" is what reads as a scientist.
- Use **specific numbers and one or two figures**, not adjectives.
- Tell the **debugging stories** (the two bugs above) — they show real, honest problem-solving.
- One paragraph on **what you'd do next** (race pace as a second signal; de-biasing cross-era) shows
  you understand the frontier.

## Figure inventory (what each one shows)
- `figures/attribution_diagnostic.png` — the naive model failing (flat car effect).
- `figures/v2_attribution_diagnostic_2018_2025.png` — the fix: car effect restored.
- `figures/v2_uncertainty_2018_2025_rw.png` — the split as a distribution (P(car>driver)=73%).
- `figures/v2_skill_trajectories_2006_2025_rw.png` — believable career arcs.
- `figures/era_connectivity.png` — the teammate chain reaching back to Senna.
- `figures/backtest.png` — out-of-sample predicted vs actual (the validation).
