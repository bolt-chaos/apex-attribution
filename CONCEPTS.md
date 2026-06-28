# CONCEPTS — the ideas behind apex-attribution

A plain-language guide to the concepts this project is built on. It's meant to make the rest of
the repo (`README.md`, `ARCHITECTURE.md`, the code) readable even if you've never met these ideas
before. Each concept follows the same shape: **the intuition** → **how this project uses it** →
**where to look** in the code/docs.

If you read top to bottom it tells a story: the *question* → why it's *hard* (causation) → the
*trick* that makes it possible (teammates) → the *statistics* that make it honest → how we *check*
that it actually works.

---

## The question, in one line

> In Formula 1, how much of a result is the **driver** and how much is the **car**?

A fan's bar argument. We turn it into something you can actually compute: hold the driver fixed,
swap the car, and measure how much the predicted result changes — and vice versa.

---

## Part A — Causal inference (why this is hard)

### Correlation vs. causation
**Intuition.** Two things moving together doesn't mean one *causes* the other. Ice-cream sales and
drownings both rise in summer — heat drives both; ice cream doesn't drown anyone.

**In this project.** Verstappen wins a lot *and* drives a Red Bull. Is he winning *because* he's
great, or *because* the car is great? Plain win-counting can't separate these — they're tangled.
Untangling them is the entire project.

### Confounding
**Intuition.** A **confounder** is a hidden cause sitting behind two things, creating a misleading
link between them (the "summer" in the ice-cream example).

**In this project.** The big confounder is that **good drivers tend to get good cars** (teams sign
the best, the best want the fastest teams). So driver skill and car quality are correlated *in the
data* even before any race happens. If you ignore this, you'll credit the driver for the car's
speed (or vice versa).

### Structural Causal Model (SCM) and the DAG
**Intuition.** An SCM is a diagram of *what causes what*, drawn as a **DAG** (Directed Acyclic
Graph — a flowchart of arrows with no loops). Each arrow means "this directly influences that."
Once you commit to the diagram, math can answer cause-and-effect questions on it.

**In this project.** The DAG says: `driver_skill` and `car_pace` and `circuit_type` all feed into
`grid` (qualifying position) and `finish_pos` (race result). We use the `gcm` module of the DoWhy
library to fit and query it.

→ See [`ARCHITECTURE.md`](ARCHITECTURE.md) §5–6, [`scripts/dag.py`](scripts/dag.py),
  [`v2/attribution_v2.py`](v2/attribution_v2.py) (the `NODES`/`EDGES` lists).

### Intervention — the `do()` operator
**Intuition.** There's a difference between *observing* and *doing*. "People who take this medicine
are healthier" (observing — maybe healthy people just take more medicine) vs. "we *gave* people the
medicine" (doing — now we know it's the medicine). Causal notation writes the second as
`do(medicine = yes)`: reach in and *set* a value, breaking its normal causes.

**In this project.** The core query is `do(constructor = Red Bull)` while holding the driver fixed:
*put this exact driver in that exact car* and predict the result. That's a thing you can't observe
(the driver was never in that car) but the SCM can compute.

### Counterfactual
**Intuition.** A counterfactual is a rewind-and-change-one-thing question: "*given what actually
happened*, what *would* have happened if one detail were different?" It's more specific than a plain
intervention because it keeps the original race's luck and circumstances fixed.

**In this project.** "Albon actually finished P13 in the Williams — where would *that same race* have
put him in a Red Bull?" → the model says ~P7. Same driver, same day, same luck; only the car changes.

→ See `gcm.counterfactual_samples(...)` in [`v2/attribution_v2.py`](v2/attribution_v2.py).

### Identification — the deepest idea here
**Intuition.** A question is **identified** if the data can actually pin down the answer; it's
**non-identified** if many different truths all fit the data equally well, so no amount of clever
math can separate them. Example: I tell you two numbers sum to 10. You can't recover them — `7+3`
and `6+4` both fit. The problem isn't your math; the information just isn't there.

**In this project.** This is the plot twist of the whole repo. **v1 failed** not because of a bug
but because the question wasn't identified: each driver sat in essentially *one* car (Cramér's V
0.84 — almost perfectly nested), so "driver skill" and "car pace" were the unknown two numbers that
sum to 10. The model couldn't split them and dumped the car's speed onto the driver. Everything in
v2 is about *engineering identification* — getting the data into a shape where the split is
recoverable.

→ See [`README.md`](README.md) "the attribution is not identified", [`ARCHITECTURE.md`](ARCHITECTURE.md) §6.

---

## Part B — The F1 trick that rescues identification

### Teammates share a car
**Intuition.** The cleanest controlled experiment F1 hands you for free: two teammates drive the
**same car**. So when one out-qualifies the other, the car can't be the reason — it cancels. The
gap between them is **pure driver skill**.

**In this project.** This single fact is the backbone. It's what makes skill *identifiable* (the
teammate gap is a direct skill measurement) and what makes forecasting the future possible (the
unknown future car cancels). The whole `predict.py` forecaster rides on it.

### Connecting the grid into one chain (the connected component)
**Intuition.** Teammate gaps only compare two drivers at a time. But drivers *change teams* across
seasons: A raced with B, then B raced with C, then C with D… Chain those overlaps and you can
compare A to D even though they were never teammates — like ranking chess players who never played
each other directly through a chain of common opponents. (Same idea as chess Elo.)

**In this project.** We keep only the **largest "connected component"** — the biggest group of
drivers all linked by some chain of shared teammates. A driver floating with no shared-teammate
links can't be placed on the same scale, so they're excluded. Crucially, *how wide a span of
seasons you include* changes connectivity, and that turned out to drive the headline result.

→ See `largest_connected_cohort` in [`scripts/build_dataset.py`](scripts/build_dataset.py),
  [`v2/era_connectivity.py`](v2/era_connectivity.py), [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md) connectivity section.

### Latent variables — skill and pace
**Intuition.** A **latent** variable is something real but not directly measured — you infer it from
its effects. You never see "intelligence" directly; you infer it from test scores. Here, you never
see "driver skill" or "car pace" on a stat sheet; you infer them from lap times.

**In this project.** v1's mistake was using the driver's *name* and the car's *name* as raw
categories (which are perfectly nested → non-identified). v2 replaces those with two inferred
**continuous** numbers — `driver_skill` and `car_pace` — estimated so that they're only mildly
correlated (~0.49) instead of nearly identical. *Now* the SCM can tell them apart.

### Qualifying pace as the cleanest signal
**Intuition.** To measure skill you want the least-contaminated signal. Race results are muddied by
crashes, strategy, traffic, reliability. A single flying qualifying lap is much closer to "pure
pace."

**In this project.** The skill model is trained on **qualifying gap** — how far off the fastest lap
each driver was, as a percentage (`pct_gap`; 0% = pole). Percentages make different circuits
comparable. A later refinement (`--gap-method session`) compares each lap to the fastest lap *in the
same session* to remove a subtle track-evolution bias.

→ See [`v2/build_quali.py`](v2/build_quali.py).

---

## Part C — The statistics that make it honest

### Bayesian inference: prior, likelihood, posterior
**Intuition.** Bayesian reasoning is "update your beliefs with evidence." You start with a **prior**
(what's reasonable before seeing data — e.g. "driver skills are within a few percent of average"),
the data has a **likelihood** (how well any guess explains what you saw), and combining them gives a
**posterior** (your updated belief). The key feature: the posterior isn't a single number, it's a
*whole distribution* of plausible values — uncertainty is built in, not bolted on.

**In this project.** The skill/pace model is Bayesian (built with PyMC). It doesn't output "Norris's
skill = −0.34%"; it outputs thousands of plausible values for it. Every error bar in the project
comes from that.

→ See [`v2/fit_skill.py`](v2/fit_skill.py), [`v2/fit_skill_rw.py`](v2/fit_skill_rw.py).

### Hierarchical (multilevel) model
**Intuition.** Data often has structure: drivers nested in teams, races nested in seasons. A
**hierarchical** model respects that structure and lets related estimates "borrow strength" from
each other — a driver with few races gets sensibly pulled toward the grid average instead of giving
a wild estimate.

**In this project.** Skill is per-driver, pace is per-team-per-season, and they're tied together by
the shared-teammate structure. This "crossed random effects" setup is what lets the chain-of-
teammates logic actually identify the scale.

### Random walk — letting skill change over time
**Intuition.** A random walk models something that drifts slowly and directionlessly: each step is
the previous value plus a small random nudge. Best guess for next year = this year; but uncertainty
grows the further out you go (and grows like **√time**, because random steps partly cancel).

**In this project.** Drivers improve and decline, so skill isn't frozen. Each driver's skill follows
a per-season random walk. The model *learns* the step size (`sigma_rw ≈ 0.11%/season` — small, so
skill is fairly stable). This is also what lets `predict.py` project skill into next season while
honestly widening the error bars.

→ See [`v2/fit_skill_rw.py`](v2/fit_skill_rw.py); recovered career arcs in
  [`figures/v2_skill_trajectories_2018_2025_rw.png`](figures/v2_skill_trajectories_2018_2025_rw.png).

### Credible interval
**Intuition.** A range that the true value is *probably* in. "90% credible interval `[−0.69, −0.39]`"
means: given the data, there's a 90% chance the real value sits in that range. Wide = "unsure";
narrow = "confident." It's uncertainty made into a number you can read. (Its classical cousin, the
*confidence interval*, sounds the same but technically can't be read that intuitively — the credible
interval is the one that means what you'd want it to mean.)

**In this project.** Computed by taking the Bayesian posterior's cloud of values and chopping off the
extreme 5% on each end (`np.quantile(draws, [0.05, 0.95])`). Every `±` and error bar you see is one
of these.

### Convergence and R-hat
**Intuition.** Bayesian models are fit by an algorithm (MCMC) that *explores* the space of plausible
answers by wandering through it with several independent walkers. You need to check the walkers
actually agree and settled down — otherwise the "posterior" is garbage. **R-hat** is the standard
check: it should be ≈ 1.00. Values like 1.04 are a yellow flag that the fit didn't fully settle.

**In this project.** Good fits report R-hat 1.00–1.02. The stalled cross-era attempt (the hard
"Senna's era" extension) sat at R-hat 1.04 — which is exactly why it's flagged as *not trustworthy
yet* rather than quietly shipped.

### Robust noise (Student-t)
**Intuition.** Real data has outliers (a wet qualifying lap, a botched run). A normal "bell curve"
assumption panics at outliers and lets them distort everything. A **Student-t** distribution has
"fatter tails" — it expects the occasional weird value and shrugs it off.

**In this project.** Lap-time gaps are modeled with a Student-t so freak sessions don't warp the
skill estimates.

---

## Part D — Earning trust (validation)

### Out-of-sample backtesting
**Intuition.** A model that fits the past can just be memorizing it. The real test is **prediction**:
hide some data, train on the rest, and see if it predicts the hidden part. If it does, it learned
something real.

**In this project.** This is the project's strongest evidence. Train skills on **2018–2023**, then
predict every teammate qualifying head-to-head in the **held-out 2024–2025** (data the model never
saw). Result: **67% race-level / 80% season-long** accuracy vs. a 50% coin-flip, correlation 0.40.
It predicts the future, for the thing it claims to measure.

→ See [`v2/backtest.py`](v2/backtest.py), [`figures/backtest.png`](figures/backtest.png).

### Calibration
**Intuition.** Separate from being *accurate*, a model should be *honest about its confidence*. If you
look at all the times it said "90% sure," it should be right about 90% of those times. Over-confident
models claim 90% and are right 70%; under-confident ones are the reverse.

**In this project.** The backtest checks this: the model's 50/80/90% intervals actually cover reality
50/80/90%+ of the time (slightly *conservative* — its error bars are a touch too wide, which is the
safe direction to err).

### Variance attribution (ICC)
**Intuition.** Once the model is trusted, you ask the headline question: of all the variation in race
results, what *share* is explained by the car vs. the driver? **ICC** (Intrinsic Causal Influence)
splits the total variance into each cause's contribution.

**In this project.** This produces the famous answer — and the honest finding that **there is no
single answer**: it depends on the era. Over the tightly-matched 2018–2025 cars, car ≈ 32% / driver
≈ 21% (close, overlapping). Over the wild-variation 2006–2025 span, car ≈ 44% / driver ≈ 12%
(car clearly dominant). More car variety in the window → more the car explains.

→ See [`v2/attribution_v2.py`](v2/attribution_v2.py), `intrinsic_causal_influence`.

### DNF censoring
**Intuition.** Missing data is rarely random. If a driver retires (DNF — Did Not Finish) on lap 1,
you don't know where they'd have finished — and *whether* they DNF'd may itself depend on the car
(unreliable engine) or the driver (crash). Naively dropping or zeroing these biases the result.

**In this project.** Handled deliberately, not papered over: skill is estimated on classified
finishes only, and "expected finish *including* breakdown risk" is recombined separately at the
reporting stage. The choice is documented as a known tradeoff.

→ See [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md) DNF section, [`README.md`](README.md) DNF handling note.

---

## One-screen glossary

| Term | One-line meaning |
|---|---|
| **Confounder** | A hidden common cause that fakes a link (good drivers get good cars). |
| **SCM / DAG** | A flowchart of what-causes-what that math can query. |
| **`do(x)` / intervention** | *Setting* a value and breaking its normal causes ("put this driver in that car"). |
| **Counterfactual** | Rewind a real event, change one thing, replay ("…in a Red Bull instead"). |
| **Identification** | Whether the data *can* answer the question at all (v1's fatal flaw). |
| **Teammate trick** | Same car ⇒ the gap between teammates is pure skill. |
| **Connected component** | Chaining shared-teammate links to compare drivers who never met. |
| **Latent variable** | A real quantity you infer rather than measure (skill, pace). |
| **Prior / posterior** | Belief before / after seeing the data (Bayesian). |
| **Hierarchical model** | Respects nesting (drivers in teams) so estimates borrow strength. |
| **Random walk** | Slow directionless drift; uncertainty grows like √time. |
| **Credible interval** | The range the true value is probably in (e.g. 90%). |
| **R-hat** | Convergence check; should be ≈ 1.00. |
| **Student-t** | Outlier-tolerant "fat-tailed" bell curve. |
| **Backtest** | Predict held-out data to prove it's not just memorizing. |
| **Calibration** | Is the model's stated confidence honest? |
| **ICC** | Share of result-variance due to each cause (car vs. driver). |
| **DNF censoring** | Non-random missing results that must be handled carefully. |

---

*This doc explains the ideas; for the system map and pipeline see [`ARCHITECTURE.md`](ARCHITECTURE.md),
for the phase-by-phase story and results see [`README.md`](README.md), and for the data schema see
[`SCHEMA_NOTES.md`](SCHEMA_NOTES.md).*
