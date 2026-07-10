# CONCEPTS — the ideas behind apex-attribution

A plain-language guide to the concepts this project is built on. It's meant to make the rest of
the repo (`README.md`, `ARCHITECTURE.md`, the code) readable even if you've never met these ideas
before. Each concept follows the same shape: **the intuition** → **how this project uses it** →
**where to look** in the code/docs.

If you read top to bottom it tells a story: the *question* → why it's *hard* (causation) → the
*trick* that makes it possible (teammates) → the *statistics* that make it honest → how we *check*
that it actually works → and how we *extend* it to measure more of the driver (race pace, crashes).

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

→ See [`ARCHITECTURE.md`](ARCHITECTURE.md) §5–6, [`scripts/dag.py`](../scripts/dag.py),
  [`v2/attribution_v2.py`](../v2/attribution_v2.py) (the `NODES`/`EDGES` lists).

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

→ See `gcm.counterfactual_samples(...)` in [`v2/attribution_v2.py`](../v2/attribution_v2.py).

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

→ See [`README.md`](../README.md) "the attribution is not identified", [`ARCHITECTURE.md`](ARCHITECTURE.md) §6.

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

→ See `largest_connected_cohort` in [`scripts/build_dataset.py`](../scripts/build_dataset.py),
  [`v2/era_connectivity.py`](../v2/era_connectivity.py), [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md) connectivity section.

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

→ See [`v2/build_quali.py`](../v2/build_quali.py).

### A second signal — quali_skill vs. racecraft
**Intuition.** If you have *two* different tests of the same ability, use both — but trust the
noisier one less. Qualifying (one flying lap) is a clean speed test; a *race* is a messier test
(traffic, tyre wear, strategy) that reveals something slightly different — **racecraft**. You can
model both at once as two related abilities per driver, each pinned by its own test, with a
**correlation** linking them (people fast on Saturday tend to be fast on Sunday — but not
identically).

**In this project.** The joint model gives each driver two latents — `quali_skill` (Saturday) and
`racecraft` (Sunday). Race pace gets its *own, larger* noise term, so it's a complement, not a
replacement (see *Robust noise*, and the down-weighting idea below). The two come out highly
correlated (`rho ≈ 0.92`), and the small gap `racecraft − quali_skill` is the **"qualifying
merchant"** signal — who races better than they qualify (Pérez) vs. quali specialists. Honest caveat:
that gap is small and sensitive to how lapped cars are treated, so it's read as suggestive, not
definitive. It also feeds the attribution: measuring the driver by *race* pace attributes even more
of a result to the car than qualifying does.

→ See [`v2/build_race_pace.py`](../v2/build_race_pace.py), [`v2/fit_skill_joint.py`](../v2/fit_skill_joint.py),
  [`v2/backtest_race.py`](../v2/backtest_race.py).

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

→ See [`v2/fit_skill.py`](../v2/fit_skill.py), [`v2/fit_skill_rw.py`](../v2/fit_skill_rw.py).

### Hierarchical (multilevel) model
**Intuition.** Data often has structure: drivers nested in teams, races nested in seasons. A
**hierarchical** model respects that structure and lets related estimates "borrow strength" from
each other — a driver with few races gets sensibly pulled toward the grid average instead of giving
a wild estimate.

**In this project.** Skill is per-driver, pace is per-team-per-season, and they're tied together by
the shared-teammate structure. This "crossed random effects" setup is what lets the chain-of-
teammates logic actually identify the scale.

### Partial pooling and shrinkage
**Intuition.** When some groups have very little data, their raw averages lie: flip a coin 3 times,
get 3 heads, and the raw rate screams "100%." **Partial pooling** pulls each group's estimate toward
the overall average by an amount that depends on how little data it has — tiny samples shrink a lot,
big samples barely budge. It's the statistically honest way to say "that extreme number is mostly
luck." (It's the *mechanism* that makes the hierarchical model above work.)

**In this project.** This is the headline of the incident model. Raw driver-error-crash rates look
dramatic — Grosjean 15.8%, Norris 2.6% — but they come from few events. After shrinkage the credible
rates compress into a narrow **~5.4–7.0%** band: the *ordering* is real and believable (Grosjean/
Latifi worst, Norris/Hamilton cleanest), but most of the eye-popping raw spread was small-sample
noise. Shrinkage turns "Grosjean crashes 6× more!" into the truer, duller "he's somewhat more
incident-prone."

→ See [`v2/fit_incident.py`](../v2/fit_incident.py),
  [`figures/v2_incident_proneness_2018_2025.png`](../figures/v2_incident_proneness_2018_2025.png).

### Modeling a yes/no outcome (logistic regression)
**Intuition.** Some outcomes aren't numbers, they're yes/no: did the driver crash out this race? You
model the *probability* of "yes" and let it depend on factors (who's driving, which circuit). Since a
probability must stay between 0 and 1, the math works on the **log-odds** scale — a transform that
lets you add up effects without ever falling off the 0–1 edge.

**In this project.** The incident model is Bayesian **logistic**: `logit(crash probability) =
baseline + driver effect + circuit effect`. The circuit term controls for the fact that street
circuits (walls close) crash more, so a driver isn't penalised just for racing at Monaco.

→ See [`v2/fit_incident.py`](../v2/fit_incident.py).

### Random walk — letting skill change over time
**Intuition.** A random walk models something that drifts slowly and directionlessly: each step is
the previous value plus a small random nudge. Best guess for next year = this year; but uncertainty
grows the further out you go (and grows like **√time**, because random steps partly cancel).

**In this project.** Drivers improve and decline, so skill isn't frozen. Each driver's skill follows
a per-season random walk. The model *learns* the step size (`sigma_rw ≈ 0.11%/season` — small, so
skill is fairly stable). This is also what lets `predict.py` project skill into next season while
honestly widening the error bars.

→ See [`v2/fit_skill_rw.py`](../v2/fit_skill_rw.py); recovered career arcs in
  [`figures/v2_skill_trajectories_2018_2025_rw.png`](../figures/v2_skill_trajectories_2018_2025_rw.png).

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

→ See [`v2/backtest.py`](../v2/backtest.py), [`figures/backtest.png`](../figures/backtest.png).

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

**Why it's now *demoted*.** A variance share isn't a property of the world — it depends on the
population (which is why it moves with the era) *and* on the graph. ICC assumes the root causes have
*independent* noise; but skill and car pace are correlated (good drivers get good cars), and once you
put that confounder in the graph the ICC split **swings ~25pp and flips** (car 26%/driver 16% →
car 1%/driver 58%) — while the interventional/counterfactual answers don't move at all. So the project
now leads with those robust measures and keeps ICC only as a clearly-caveated descriptive number.

→ See [`v2/attribution_v2.py`](../v2/attribution_v2.py), `intrinsic_causal_influence`; and *Probability
  of Necessity* below.

### Probability of Necessity — the "but for" question
**Intuition.** A sharp, rung-3 way to ask "how much was it the car vs. the driver" *without* a
variance share: of the podiums a driver actually got, how many would have **vanished but for** the
car (imagine the same driver, same day, in a midfield car) — versus but for the driver (a median
driver, same car)? This is a counterfactual applied to the events that *happened*, so it doesn't
depend on the population's spread the way ICC does.

**In this project.** On the wide era a podium needs the **car 82%** of the time vs the **driver 68%**
— near-parity, car slightly ahead, agreeing with the interventional spread and OLS (and unlike the
wildly-swinging ICC). Face-valid: Räikkönen/Bottas/Piastri's podiums are the most car-dependent;
Alonso's and Verstappen's the most driver-dependent.

→ See `necessity_query` in [`v2/attribution_v2.py`](../v2/attribution_v2.py),
  [`figures/v2_necessity_2018_2025_joint.png`](../figures/v2_necessity_2018_2025_joint.png).

### DNF censoring
**Intuition.** Missing data is rarely random. If a driver retires (DNF — Did Not Finish) on lap 1,
you don't know where they'd have finished — and *whether* they DNF'd may itself depend on the car
(unreliable engine) or the driver (crash). Naively dropping or zeroing these biases the result.

**In this project.** Handled deliberately, not papered over: skill is estimated on classified
finishes only, and "expected finish *including* breakdown risk" is recombined separately at the
reporting stage. The choice is documented as a known tradeoff.

→ See [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md) DNF section, [`README.md`](../README.md) DNF handling note.

---

## Part E — Beyond pace: the rest of a driver's contribution

The models above measure how *fast* a driver is. But a driver's value is more than speed — it also
includes *finishing the race*. These last ideas fold that in.

### Incident-proneness — not crashing is a skill
**Intuition.** Being fast isn't everything; *finishing* matters. Binning the car into a wall hands
back points, and some drivers do it more than others. That's a real, separate part of a driver's
value that the pace models can't see.

**In this project.** Retirements are split into **mechanical** (the car's fault — never charged to
the driver) and **driver-error** (crash/spin — a genuine driver outcome). The incident model
estimates each driver's driver-error rate (via *partial pooling*, above), controlling for circuit
hazard. The honest verdict: incident-proneness is real and sensibly ordered, but a *modest*
differentiator once small-sample noise is stripped out.

→ See [`v2/fit_incident.py`](../v2/fit_incident.py).

### Expected cost = chance × stakes (the "incident tax")
**Intuition.** What a mistake *costs* isn't just how often you make it — it's how much you lose each
time. A leader who crashes throws away a win; a backmarker who crashes throws away 17th. Same crash,
very different cost. Expected loss = probability × consequence.

**In this project.** The **incident tax** (positions lost per race to your own errors) turns out to
be *stakes-dominated*: because crash rates barely differ between drivers, the tax mostly tracks how
high you'd have finished. Verstappen's tax is the highest — not because he's reckless (he's clean)
but because each of his rare mistakes costs a podium. A separate **cleanliness dividend** isolates
the driver's own contribution, and it's tiny — the honest conclusion that incident-proneness is a
minor tiebreaker between similarly-fast drivers.

### The unified metric — one number, combined only at the end
**Intuition.** A driver's real expected result blends three things: how high they'd finish, the
chance the car breaks, and the chance they crash. You keep these *separate* while modelling (so an
engine failure is never blamed on the driver), then combine them only at the reporting step:
`expected result ≈ (chance of finishing) × (expected finish) + (chance of a DNF) × (back of field)`.

**In this project.** This is "Option A" — pace, mechanical risk, and incident risk are estimated
independently and mixed only in the final metric, so no imputed crash ever pollutes the skill
estimate.

→ See [`v2/unified_metric.py`](../v2/unified_metric.py), [`v2/fit_incident.py`](../v2/fit_incident.py).

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
| **Second signal (racecraft)** | A noisier extra measurement (race pace), down-weighted, not a replacement for quali. |
| **Qualifying merchant** | Fast on Saturday, worse on Sunday — the `racecraft − quali_skill` gap. |
| **Prior / posterior** | Belief before / after seeing the data (Bayesian). |
| **Hierarchical model** | Respects nesting (drivers in teams) so estimates borrow strength. |
| **Partial pooling / shrinkage** | Pull noisy small-sample rates toward the average; extremes are mostly luck. |
| **Logistic regression** | Model a yes/no outcome's probability (on the log-odds scale). |
| **Random walk** | Slow directionless drift; uncertainty grows like √time. |
| **Credible interval** | The range the true value is probably in (e.g. 90%). |
| **R-hat** | Convergence check; should be ≈ 1.00. |
| **Student-t** | Outlier-tolerant "fat-tailed" bell curve. |
| **Backtest** | Predict held-out data to prove it's not just memorizing. |
| **Calibration** | Is the model's stated confidence honest? |
| **ICC** | Share of result-variance due to each cause — *demoted*: population- and graph-dependent. |
| **Probability of Necessity** | "But for" the car/driver, would this podium have happened? A robust rung-3 contrast. |
| **Hiring edge** | `driver_skill→car_pace` — puts the "good drivers get good cars" confounder in the graph. |
| **Attenuation / errors-in-variables** | A noisy *estimated* predictor biases its effect toward zero; correcting it de-attenuates. |
| **Reliability** | Fraction of a measured variable's spread that's real signal (skill 0.78, car 0.93 here). |
| **DNF censoring** | Non-random missing results that must be handled carefully. |
| **Incident-proneness** | Per-driver driver-error-crash rate — finishing is a skill too. |
| **Expected cost = chance × stakes** | A mistake costs more the higher you'd have finished (the incident tax). |
| **Unified metric** | Combine pace + mechanical + crash risk into one expected result, at reporting only. |

---

*This doc explains the ideas; for the system map and pipeline see [`ARCHITECTURE.md`](ARCHITECTURE.md),
for the phase-by-phase story and results see [`README.md`](../README.md), and for the data schema see
[`SCHEMA_NOTES.md`](SCHEMA_NOTES.md).*
