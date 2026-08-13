# Pitch draft — a DoWhy example notebook

Working drafts of the outreach message for [`py-why/dowhy`](https://github.com/py-why/dowhy),
proposing this project as a `gcm` example notebook. The direction and its rationale are in
[`IDEAS.md` §3](IDEAS.md); this file is just the message itself, in two lengths.

**Status: unsent.** Nothing here has been posted. Pick a version, fill the placeholder, send.

Before sending:
- [ ] Choose short or long (see the trade-off below)
- [ ] Replace `<REPO_URL>` with this repo's URL — decide first whether it's ready for traffic
- [ ] Re-check the gallery hasn't gained an identification-failure example since 2026-08-13
- [ ] Post as a GitHub issue or a py-why Discussion (CONTRIBUTING says open an issue with
      questions; it does not specify a Discussions-first process)

## What the gallery actually contains (checked 2026-08-13)

`IDEAS.md` §3 claimed no gcm example shows the tools failing. That is now *nearly* true, and both
drafts are positioned against the two exceptions rather than ignoring them — claiming a void that
isn't there is the fastest way to look like we didn't read the docs.

- **"When Accuracy Lies: Causal Inference on a Chest X-Ray CNN"** — a real cautionary tale, but
  about a *predictive* model (AUC 0.72; ~30% of predictions driven by scanner brightness rather
  than pathology). The gcm pipeline is the **hero** that diagnoses it. Ours is the inverse: the
  **causal model itself** is wrong and the reader watches it fail. It also headlines ICC, which is
  what makes our hiring-edge finding a useful contribution rather than a footnote.
- **"Falsification of User-Given DAGs"** — complementary, and worth naming: our failure is one that
  falsification does **not** catch. The graph is correct; the *variable encoding* is what destroys
  identifiability.

Also note: `CONTRIBUTING.md` lists notebooks as a contribution type but specifies nothing about
dependencies, CI, notebook runtime, or data size. There is no published bar to design against —
which is exactly why both drafts lead with questions rather than a finished artifact.

## Trade-off between the two

The **short** version is a single-ask message: the failure arc, the gap, the design constraints,
three questions. It drops the ICC-graph-dependence lesson and the out-of-sample validation, holding
both for the reply. Better odds of actually being read end-to-end; risks reading as a thinner
contribution than it is.

The **long** version adds those two sections. It makes the stronger case but asks for meaningfully
more of a maintainer's attention on first contact.

Default recommendation: **send short.** The validation and the ICC lesson are what you say when
they write back, and having something substantial in reserve is worth more than front-loading it.

---

# Version A — short (~370 words)

**Title:** Example notebook proposal: what to do when your SCM is silently non-identified (F1
driver-vs-car attribution)

Checking interest before investing in polish, per CONTRIBUTING's note to open an issue with
questions first.

I have a `gcm` model that separates a Formula 1 driver's contribution from the car they drove —
the eternal bar argument, posed as `do(constructor = X)` holding the driver fixed. I'd like to
distil it into one self-contained example notebook. The reason isn't the domain; it's the failure
mode it teaches.

My first version did everything by the book: `auto.assign_causal_mechanisms` → `fit` →
`evaluate_causal_model` → `intrinsic_causal_influence`. Every call succeeded, no warnings, and the
answer was **confidently backwards** — 45% driver / 1.3% car, in an era where the car is famously
dominant.

The cause is general. Each driver is nested in essentially one constructor (Cramér's V 0.84), so a
categorical SCM **cannot** separate latent driver skill from latent car pace, and quietly loads the
car's contribution onto driver identity. Worth noting: DAG falsification does not catch this — the
graph is fine, it's the *variable encoding* that destroys identifiability.

The fix keeps the same graph and the same API calls: identify skill and pace as continuous latents
from **teammate contrasts** (teammates share a car, so the car term cancels; drivers switching
teams chain those comparisons into a connected graph — the trick behind chess Elo), feed those in,
and ~25 percentage points of attribution move to something that matches known ground truth.

So the arc is: **naive spec fails informatively → diagnose why → fix identification → re-run the
same calls → sane answer, honestly caveated.**

How it differs from what's already there: "When Accuracy Lies" is a cautionary tale where `gcm` is
the hero diagnosing a bad *predictive* model — here the *causal* model is the thing that's wrong.
"Falsification of User-Given DAGs" is complementary, for the reason above.

Design: one self-contained `.ipynb`, fixed seed, reduced ICC sample counts for docs-CI runtime. **No
PyMC dependency** — the latent model is Bayesian, but the fitted latents ship as a small checked-in
parquet with a paragraph explaining the teammate trick. Data is [f1db](https://github.com/f1db/f1db),
CC-BY-4.0.

Three questions:

1. Is a contribution in this shape welcome in the gallery?
2. Is "here's how the tooling fails" a framing you want in an examples section, or a tone mismatch?
3. Any constraint to design around up front — runtime budget, data-size limits, whether a
   checked-in parquet of precomputed latents is acceptable?

Happy to open a draft PR instead if that's easier to evaluate. Full project at `<REPO_URL>`.

---

# Version B — long (~850 words)

**Title:** Example notebook proposal: an identification failure you can see, using F1
driver-vs-car attribution

Hi — I'd like to check interest before investing in polish, per CONTRIBUTING's note to open an
issue with questions first.

## The proposal

I've built a `gcm`-based causal attribution model for Formula 1: **how much of a race result is the
driver, and how much is the car they happened to be driving?** It's the eternal bar argument, posed
formally as `do(constructor = X)` holding the driver fixed. I'd like to distil it into one
self-contained example notebook.

The reason I think it earns a slot isn't the domain — it's the **failure mode it teaches**.

## The teaching angle: a pipeline that runs perfectly and answers backwards

My first version did everything the docs say. `auto.assign_causal_mechanisms` → `fit` →
`evaluate_causal_model` → `intrinsic_causal_influence`. Every call succeeded, no warnings, and the
answer was **confidently wrong**: ICC attributed 45% to the driver and 1.3% to the car — under
modern regulations, where the car is famously dominant. The interventional sweeps agreed with each
other and with nothing in reality.

The cause is structural, and it's general: each driver is nested in essentially one constructor
(Cramér's V 0.84), so a categorical SCM **cannot** separate latent driver skill from latent car
pace, and quietly dumps the car's contribution onto driver identity. The signal was in the data the
whole time — the fix is to identify skill and pace as continuous latents from *teammate contrasts*
(teammates share a car, so the car term cancels; drivers switching teams chain those comparisons
into a connected graph, the same trick behind chess Elo) and feed those into the SCM. Same graph,
same API calls, ~25 percentage points of attribution moved, and the result finally matches the
known ground truth.

That's the arc I'd want the notebook to walk: **naive spec fails informatively → diagnose *why* →
fix identification → re-run the same gcm calls → sane answer, honestly caveated.**

## Why I think this is a gap, having read the gallery

The existing gcm examples are, as far as I can tell, all cases where the causal tooling *succeeds*
— online shop attribution, microservice RCA, the medical counterfactual, the wage-gap
decomposition. The two closest are:

- **"When Accuracy Lies" (chest X-ray CNN)** — a genuine cautionary tale, but the caution is about a
  *predictive* model; the gcm pipeline is the hero that diagnoses it. Mine is the inverse: the
  **causal model itself** is the thing that's wrong, and the reader watches it fail.
- **"Falsification of User-Given DAGs"** — adjacent and complementary. My failure is one that
  falsification does *not* catch: the graph is fine; it's the *variable encoding* that destroys
  identifiability.

So I'd position this as "what to do when your SCM is silently non-identified," which I don't think
anything currently covers.

## A second lesson I'd want to include: ICC is graph-dependent

While validating, I hit something that surprised me and might be worth surfacing in the docs
generally. `driver_skill` and `car_pace` are correlated (~0.5) because good drivers get hired into
good cars. Modelling that confounding as an explicit edge rather than independent roots swings the
ICC split **~25pp — from car 26% / driver 16% to car 1% / driver 58%, flipping the verdict** — while
the interventional car-vs-driver spread moves **0.0 positions**, because `do()` sets both roots
regardless.

The practical takeaway — *ICC assumes independent root noise, so lead with interventional and
counterfactual measures when your roots are confounded* — seems broadly useful, and I'd rather
demonstrate it than have readers rediscover it. (I'd also show the rung-3 necessity query: "would
this podium have happened **but for** the car / the driver?" — 82% vs 68%, and unlike ICC it's
robust to the confounding spec.)

## Practical design

- **One self-contained `.ipynb`**, fixed seed, reduced ICC sample counts for docs-CI runtime.
- **No PyMC dependency.** The latent skill/pace model is Bayesian, but I'd ship the fitted latents
  as a small checked-in parquet with a paragraph explaining the teammate trick and a link to the
  full repo. Incidentally, "gcm consuming domain-derived latents from an upstream model" may itself
  be a pattern worth showing.
- **Real, open, permissive data**: [f1db](https://github.com/f1db/f1db), CC-BY-4.0.
- **Honest caveats in-notebook**, not hidden: the split is era-dependent (there is no single
  "X% driver / Y% car" — it depends on how much car variation your window spans), and the roots
  are correlated.
- The model is **out-of-sample validated**, if that helps credibility: trained on 2018–2023, it
  predicts held-out teammate qualifying head-to-heads at 80% season-long (vs a 50% baseline), with
  credible intervals that cover at 50%→74%, 80%→92%, 90%→97%. A forecast published before the 2026
  season then scored 70% season-long against the races that followed.

## What I'm asking

1. Is a contribution in this shape welcome in the gallery?
2. Is the "here's how the tooling fails" framing something you *want* in the docs, or a tone
   mismatch for an examples section?
3. Any constraint I should design around up front — runtime budget, data-size limits, whether a
   checked-in parquet of precomputed latents is acceptable?

Happy to open a draft PR if that's an easier way to evaluate it. Full working project (code,
validation, write-ups) is at `<REPO_URL>` if you want to see the substance before deciding.
