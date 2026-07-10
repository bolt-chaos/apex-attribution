# apex-attribution × *The Book of Why*

A chapter-by-chapter map between Judea Pearl & Dana Mackenzie's *The Book of Why* (2018) and this
project. It's meant to be read two ways: as a **guided tour** of the repo for someone who knows the
book, and as a **concrete case study** of the book's ideas for someone who knows the repo. Where the
project has already acted on one of the book's warnings, that's marked **✅ acted on** with a pointer
to the code.

The through-line: *the whole v1→v2 arc is a literal climb up Pearl's Ladder of Causation, and the
project's biggest self-corrections are the book's biggest lessons, rediscovered on F1 data.*

For the plain-language version of every concept named here, see [`CONCEPTS.md`](CONCEPTS.md). For the
full Pearl-referee critique (the source of several fixes below), see [`IDEAS.md`](IDEAS.md) §1 and the
critique section of [`WRITEUP_NOTES.md`](WRITEUP_NOTES.md).

---

## The spine: the Ladder of Causation (Chapter 1)

Pearl's central image is a three-rung ladder. You cannot answer a higher-rung question with
lower-rung tools, no matter how much data you have. The entire project is organized around knowing
which rung a question lives on.

| Rung | Book's verb | The project's version | Where |
|---|---|---|---|
| **1 — Association** | *seeing* | Win counts; Bell et al.'s mixed-effects work; the repo's own rung-1 instruments (OLS standardized betas, backtest correlations) | `v2/attribution_v2.py` (OLS block), `v2/backtest.py` |
| **2 — Intervention** | *doing* | `do(constructor = Red Bull)` holding the driver fixed; the interventional pace/skill sweeps; `insights.py`'s `do(car_pace = median)` | `v2/attribution_v2.py` (`exp_finish`, sweeps), `v2/insights.py` |
| **3 — Counterfactual** | *imagining* | "Albon's actual P13 → where in a Red Bull, same day, same luck?"; "Senna in a modern Red Bull"; the Probability-of-Necessity "but for" query | `v2/attribution_v2.py` (`gcm.counterfactual_samples`, `necessity_query`), `v2/cross_era.py` |

The reason the model can even *pose* a never-observed pairing ("Doohan in a McLaren") is **graph
surgery** — Pearl's mechanism for rung 2. The reason it can replay a *specific* race with one thing
changed is **abduction → action → prediction** — his recipe for rung 3 (Chapter 8). Both are things
plain win-counting (rung 1) provably cannot do.

---

## Chapter-by-chapter

### Ch. 1 — The Ladder of Causation
See the table above. The single most important thing the project inherits from Chapter 1 is the
*discipline of asking which rung a question is on before answering it*. "How much is the driver vs.
the car?" sounds like a rung-1 counting question; the project insists it's a rung-2/3 question and
builds the machinery to match.

### Ch. 2 — From Buccaneers to Guinea Pigs (Galton, Pearson, and the birth of correlation)
Pearl's history of how statistics *talked itself out of causation* — Pearson declaring causation a
mere "limit of correlation." The project's v1 failure is a miniature re-enactment: pure
correlational structure (a driver's name, a car's name) could not recover the causal split, and no
amount of the same kind of data would fix it. The cure wasn't a better correlation — it was a causal
redesign.

### Ch. 2 (cont.) — Sewall Wright's path diagrams
Wright's path coefficients are the direct ancestor of both SCMs and regression betas. In this
project the **OLS standardized betas literally *are* path coefficients** on the DAG (`finish ~
driver_skill + car_pace`). They're kept in the report as an honest rung-1 sanity check that lines up
with the rung-2/3 answers (naive skill 0.35 / pace 0.47).
→ `v2/attribution_v2.py` (OLS block).

### Ch. 3 — From Evidence to Causes (Bayes, and Bayesian networks)
Two distinct Bayesian ideas, both live here:
- **Bayes' rule / belief updating** — the entire skill/pace measurement model is Bayesian (PyMC):
  prior → likelihood → posterior, and the output is a *distribution*, not a point. Every error bar in
  the repo descends from this. → `v2/fit_skill.py`, `v2/fit_skill_joint.py`.
- **Bayesian networks and their testable implications** — a DAG implies conditional independences you
  can *check against data*. The project does exactly this with `falsify_graph`, and — unusually — has
  a validation script that *expects* `driver ⊥ constructor` to **fail** and quantifies the leak
  rather than hoping it passes. → `scripts/dag.py`, the graph-falsification / validation scripts.

### Ch. 4 — Confounding and Deconfounding (and the heritability trap)
This chapter does the most work in the project, in two directions:

- **Confounding is *the* problem.** "Good drivers are hired into good cars" is the project's
  pitfall #1 — a textbook confounder inducing corr ≈ 0.41–0.49 between the two latents. Pearl's ideal
  answer is the RCT; the project's substitute is the **teammate trick** (two teammates share a car, so
  their gap cancels the car — a natural experiment), stitched into a grid-wide scale via the
  **connected component** of shared teammates. This is deconfounding *by design* rather than by
  backdoor adjustment. → `CONCEPTS.md` Part B, `scripts/build_dataset.py`, `v2/era_connectivity.py`.
- **The heritability / ANOVA trap — and ✅ acting on it.** Pearl's long-running distrust of
  variance-partition objects (heritability, "% due to genes vs. environment") is aimed squarely at
  **ICC** (Intrinsic Causal Influence), which was once this project's headline. The book's point:
  *a variance share is a property of the population, not the mechanism.* The repo rediscovered this
  empirically ("there is no single X%/Y%" — the split moves with the era), then **demoted ICC** and
  promoted the interventional sweeps + a necessity query to headline instead. The clinching demo:
  putting the confounder in the graph (the hiring edge, below) swings the ICC split ~25pp and flips
  it, while the interventional/counterfactual answers barely move.
  → `v2/attribution_v2.py` (report reordered; ICC demoted with an inline caveat), `CONCEPTS.md`
  "Why it's now *demoted*".

### Ch. 5 — The Smoke-Filled Debate (Hill's viewpoints, causation without a mechanism)
Not directly modeled, but the spirit shows up in the project's **triangulation** habit: rather than
rest on one estimator, it argues car-vs-driver from several independent angles (interventional
spread, probability of necessity, OLS betas, errors-in-variables correction) and reports where they
agree. That "consilience across methods" is very much the Bradford-Hill instinct applied to a single
model.

### Ch. 6 — Paradoxes Galore (Simpson, Berkson, and colliders)
- **Berkson / collider bias — a known open seam.** Finishing a race is a *common effect* of the car
  (mechanical failure) and the driver (crashing). Conditioning on `classified` (survivors) therefore
  opens a collider and can induce a spurious skill–pace dependence on top of the hiring confounder —
  Berkson's paradox in F1 clothing. The project handles DNFs deliberately (skill estimated on
  classified rows only; breakdown risk recombined at reporting) and **documents this as a tradeoff**,
  but the referee's stricter ask — draw the selection node in the DAG with a Bareinboim–Pearl
  recoverability argument — is noted as future work. → criticism #3 in [`IDEAS.md`](IDEAS.md);
  `SCHEMA_NOTES.md` DNF section.
- **Simpson's paradox** is the cousin lurking behind the era-dependence result: aggregate vs.
  per-era splits tell different stories, and the project's answer ("report the era, don't average
  over it") is the Simpson-aware move.

### Ch. 7 — Beyond Adjustment (the do-calculus)
Present *in spirit, not in machinery.* The project never runs do-calculus to find an adjustment set,
because it doesn't buy identification through backdoor adjustment — it buys it through **data design**
(the natural experiment + connectivity engineering). That's a legitimate and, here, stronger route:
the teammate trick is closer to an RCT than to an adjustment formula. Worth stating plainly in the
write-up as *"identification by design, not by do-calculus."*

### Ch. 8 — Counterfactuals (the invertible SCM)
The technical heart of rung 3. The project uses `gcm`'s **InvertibleStructuralCausalModel** to run
Pearl's three-step counterfactual recipe:
1. **Abduction** — infer the specific race's noise from what actually happened,
2. **Action** — surgically change one thing (`do(car_pace = midfield)`),
3. **Prediction** — replay to get the counterfactual result.

This is exactly "same driver, same day, same luck; only the car changes," and it powers both the
Albon-style swaps and the ✅ **Probability-of-Necessity** query — a rung-3, population-robust
"but for the car?" contrast that now co-headlines the attribution (on the wide era: car needed 82%
of the time vs. driver 68% — near-parity, car slightly ahead). → `v2/attribution_v2.py`
(`necessity_query`), `CONCEPTS.md` "Probability of Necessity".

### Ch. 9 — Mediation (direct vs. indirect effects)
The DAG contains an explicit mediator: `driver_skill → grid → finish_pos`. Some of a driver's effect
on the result flows **through Saturday** (qualifying position) and some is direct (race-day
overtaking, tyre management). Pearl's mediation formulas (natural direct/indirect effects) could
decompose "how much of skill reaches the result via grid vs. around it" — a well-posed, computable
question the project has set up but **not yet asked**. Flagged as a clean next query. → `grid` node in
`v2/attribution_v2.py`.

### Ch. 10 — Big Data, Robots, and the Promise of Causal Inference
Pearl's warning: *more data won't climb the ladder for you.* v1 is a live demonstration — the
qualifying/finishing data was abundant and clean, yet the answer was confidently backwards because
the *information* to separate driver from car wasn't in the data's **shape** (Cramér's V 0.84 — each
driver nested in essentially one car). The fix was a causal-design change (widen the era for
connectivity, extract latents via the teammate structure), not more rows. This is the single best
"Chapter 10 in the wild" story the repo has, and the write-up leads the validation section with it.
→ `CONCEPTS.md` "Identification"; `README.md` "the attribution is not identified".

---

## Beyond the book: where the project adds machinery Pearl leaves to others

The book is deliberately light on *estimation* — Pearl hands the statistics off once identification is
settled. This project fills that half in, and in two places goes past the book's toolkit:

- **Identification by design, not adjustment.** A natural experiment (teammates) plus deliberate
  *connectivity engineering* (which seasons to include so the shared-teammate graph is one connected
  component) — a route to identification the book discusses less than backdoor/frontdoor adjustment.
- **Honest Bayesian uncertainty end-to-end.** Posteriors, credible intervals, R-hat convergence
  checks, out-of-sample backtesting and calibration, and `P(car > driver)` as a probability rather
  than a point estimate. The book would call all of this "the statistician's department" — but it's
  what lets the project make *forward-looking* claims responsibly.
- **Errors-in-variables — a measurement-error patch the book gestures at but doesn't work.** Because
  `driver_skill` and `car_pace` are *estimated latents* plugged into the finish regression, their
  estimation error **attenuates** their coefficients (regression dilution). The project measures each
  latent's reliability (skill 0.78, car 0.93) and de-attenuates. Honest result: the *driver* was the
  noisier latent, so correcting it lifts the driver to parity — it doesn't rescue the car. This is the
  principled response to the referee's criticism #4 (the causal heavy-lifting happening *outside* the
  SCM, latents plugged in as if observed). → `v2/attribution_eiv.py`.

---

## The transportability problem the book names but doesn't finish

"Senna in a modern Red Bull" (`v2/cross_era.py`) is a **transportability** question in the
Bareinboim–Pearl sense: which mechanisms are invariant across eras and which differ. The project's
z-score era-normalization is an *encoding* of one such assumption (a legend's SDs-ahead-of-field map
onto the modern spread), and the naive-vs-normalized side-by-side makes the assumption **visible**.
The book's formalism — **selection diagrams** — would make it **formal** ("identified under these
stated invariances, and not otherwise"). The project stops at visible + loudly caveated (the cross-era
fit sits at R-hat 1.04, flagged as not-yet-trustworthy); selection diagrams are the natural rigorous
upgrade. → criticism #6 in [`IDEAS.md`](IDEAS.md); `CONCEPTS.md` R-hat note.

---

## Scorecard: the book's warnings, and what the project did about them

| Book lesson | Status in this repo |
|---|---|
| Put the confounder **in the graph**, not the caveats (Ch. 4) | ✅ **acted on** — hiring edge `driver_skill → car_pace`; ICC shown to swing under it. `v2/attribution_v2.py` |
| Don't let a **variance share headline a causal model** (Ch. 4) | ✅ **acted on** — ICC demoted to descriptive; interventional sweeps + PN promoted. `v2/attribution_v2.py` |
| Ask **rung-3 counterfactual** questions when they're the right ones (Ch. 8) | ✅ **acted on** — Probability-of-Necessity "but for" query. `necessity_query` |
| Don't plug **estimated latents in as if observed** (measurement error) (Ch. 3/4) | ✅ **acted on** — errors-in-variables de-attenuation. `v2/attribution_eiv.py` |
| Beware **collider/Berkson** bias from conditioning on survival (Ch. 6) | ⏳ **documented, not yet formal** — DNF handling is deliberate; selection node not yet in the DAG |
| **Mediation**: split direct vs. indirect effects (Ch. 9) | ⏳ **set up, not yet asked** — `grid` is an explicit mediator; decomposition uncomputed |
| **Transportability** across eras via selection diagrams (Bareinboim–Pearl) | ⏳ **visible, not yet formal** — z-score normalization + loud caveats; diagrams would formalize |
| **Big data won't climb the ladder** (Ch. 10) | ✅ **the founding lesson** — v1's non-identification finding *is* this chapter, on F1 |

> Pearl's parting line to a project like this would be the one already quoted in `IDEAS.md`: *"you
> asked whether your question was answerable before answering it, and rebuilt the data until it was.
> Now put your confounder in the graph instead of the caveats, and stop letting a variance share
> headline a causal model."* The scorecard above is the project taking him up on it.

---

*Companion docs: [`CONCEPTS.md`](CONCEPTS.md) (plain-language concept guide), [`IDEAS.md`](IDEAS.md)
(the full Pearl-referee review + future directions), [`WRITEUP_NOTES.md`](WRITEUP_NOTES.md) (running
write-up notes), [`ARCHITECTURE.md`](ARCHITECTURE.md) (system map), [`README.md`](../README.md) (phase
log & results).*
