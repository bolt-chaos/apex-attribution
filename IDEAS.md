# IDEAS — directions & external-review notes

Brainstorming notes from a review session (2026-07-02): how a Pearl-style reviewer would read
the project, how its concepts map to *The Book of Why*, and two concrete directions (a DoWhy
example notebook, an interactive site). Kept in the same spirit as `WRITEUP_NOTES.md`: honest
and specific. None of this is committed work — it's a menu.

---

## 1. The Pearl review (what a causal-inference referee would say)

### What he'd applaud

- **The question is posed on the right rung.** "Driver vs. car" is framed as
  `do(constructor = X)` holding the driver fixed — an interventional query, explicitly
  distinguished from Bell et al.'s associational mixed-effects work. The counterfactuals in
  `v2/attribution_v2.py` correctly use rung 3 (abduct the race's noise, mutate the car, replay
  — "same driver, same day, same luck").
- **Identification before estimation.** v1's headline — *the machinery works, but the
  attribution is not identified* — is the discipline most applied work skips. The fix came from
  re-engineering the **data design** (teammate chains, era-widening for connectivity), not from
  a fancier estimator.
- **Assumptions on the table.** An explicit DAG (`scripts/dag.py`), conditional-independence
  falsification (`falsify_graph`), a validation script that *expects* driver⊥constructor to fail
  and quantifies the leak, loud caveats on the off-support Senna query.

### What he'd criticize

1. **The DAG contradicts its own validation report** (the sharpest one). `driver_skill` and
   `car_pace` are drawn as *independent roots*, while the docs call selection confounding
   "pitfall #1" and measure corr ≈ 0.41–0.49 between the latents. The hiring mechanism belongs
   *in the graph*: a latent common cause (bidirected edge → semi-Markovian model) or an explicit
   `skill → team_assignment → car_pace` node. Concretely: `gcm` fits a Markovian SCM with
   independent noise, and `intrinsic_causal_influence` — the headline metric — decomposes
   variance under exactly that assumption. With dependent exogenous variables the ICC shares are
   partly an artifact. (The `do()` sweeps and counterfactual swaps are less affected, since both
   roots are being *set*; it's the variance attribution that inherits the bias.)
2. **Variance decomposition is not a causal invariant — and the project proved it.** ICC is the
   heritability-style ANOVA object Pearl has always distrusted: variance shares depend on the
   *distribution* of the causes in the window, not just the mechanism (2006–2025 "shows more
   car" because it contains more car variance). The repo discovered this empirically ("there is
   no single X%/Y%") and reported it honestly — but the referee's push is: that finding is the
   *definition* of why variance shares are the wrong headline. Promote the `do()`-sweep spreads
   and counterfactual contrasts; demote ICC to descriptive. Consider a **probability of
   necessity/sufficiency** query — "would Verstappen have been champion *but for* the Red
   Bull?" — rung-3, well-posed, and the invertible-SCM machinery already exists.
3. **Conditioning on "classified" is conditioning on a collider.** Finishing is a common effect
   of the car (mechanical failure) and the driver (crashing); restricting to survivors induces
   Berkson-style spurious dependence between skill and pace, on top of the selection
   confounding. Option A is a reasonable design, but the selection node should be drawn in the
   DAG with a recoverability argument (Bareinboim–Pearl selection-bias theory), not left as a
   design note. Same concern one level up: survivorship in who stays on the grid at all.
4. **The causal heavy lifting happens outside the SCM.** The identification victory (the
   teammate trick) lives in the PyMC measurement model; the latents are plugged into the DAG as
   if observed — a measurement-error / plug-in problem. Phases 3/6 (joint posterior draws
   through the ICC) are a good patch; the principled version is one SCM with latent nodes.
   Related circularity: `driver_skill` is *estimated from* qualifying gaps, then used as a
   *cause of* `grid` (the qualifying outcome) — the same data sits on both ends of that edge.
5. **Hidden cycles.** Top drivers improve the car (development feedback); a DAG must be acyclic,
   so the honest fix is temporal unrolling (`car_pace[t]` depends on driver feedback at t−1).
   The per-season random walk on skill is a step in that direction; per-team-season `car_pace`
   silently averages a within-season development trajectory the driver partly causes.
6. **"Senna in a modern Red Bull" is a transportability problem with an existing formalism.**
   The era-normalization z-score is exactly what Bareinboim–Pearl selection diagrams encode:
   which mechanisms are invariant across eras and which differ. The naive-vs-normalized
   side-by-side makes the assumption *visible*; selection diagrams would make it *formal* —
   "identified under these stated invariances, and not otherwise."

### The takeaway

The two biggest criticisms (missing bidirected edge; ICC as headline) are **specification and
reporting fixes, not rebuilds**. The interventional/counterfactual machinery that survives them
is the strongest part of the repo, and the backtest/calibration work — outside Pearl's theory
but good hygiene — already guards the model's forward-looking claims.

> *"You did the rare thing — you asked whether your question was answerable before answering
> it, and you rebuilt the data until it was. Now finish the job: put your confounder in the
> graph instead of in the caveats, and stop letting a variance share headline a causal model —
> the variance belongs to the population, only the mechanism belongs to the world."*

---

## 2. Mapping to *The Book of Why*

> A fuller, chapter-by-chapter version of this mapping — with a scorecard of which of the book's
> warnings the project has since acted on — now lives in [`BOOK_OF_WHY.md`](BOOK_OF_WHY.md). The
> sketch below is the seed it grew from.

The v1→v2 arc is a literal ascent of the **Ladder of Causation** (Ch. 1):

- **Rung 1 — association:** win counts; Bell et al.; the project's own rung-1 instruments (OLS
  standardized betas, backtest correlations).
- **Rung 2 — intervention:** `do(constructor = Red Bull)`; the interventional sweeps;
  `insights.py`'s `do(car_pace = median)`. Graph surgery is why the model can pose
  never-observed pairings ("Doohan in a McLaren").
- **Rung 3 — counterfactuals:** `gcm.counterfactual_samples` = the abduction–action–prediction
  recipe of Ch. 8; "Senna in a modern Red Bull" is the showpiece.

| *Book of Why* | In this project |
|---|---|
| Ch. 2 — Wright's path diagrams | The OLS standardized betas *are* path coefficients |
| Ch. 3 — Bayes nets, testable implications | `falsify_graph` tests the DAG's implied conditional independences |
| Ch. 4 — confounding & RCTs | Pitfall #1 (drivers hired into good cars); the teammate trick as the RCT substitute |
| Ch. 6 — Berkson's paradox / colliders | Conditioning on `classified` (see criticism #3 above) |
| Ch. 7 — do-calculus | Present in spirit, not machinery — identification bought by *design*, not adjustment |
| Ch. 8 — counterfactuals | Invertible SCM, "same luck" replay |
| Ch. 9 — mediation | `grid` is an explicit mediator; "how much of skill flows through Saturday?" is computable and unasked |
| Ch. 10 — big data won't save you | v1 is a live demo: the information wasn't in the data's *shape* (Cramér's V 0.84) |

Where the project goes beyond the book's toolkit: **identification by design** (natural
experiment + connectivity engineering rather than backdoor adjustment) and **honest Bayesian
uncertainty** (posteriors, calibration, P(car>driver) — the estimation half Pearl leaves to
statisticians). Where it still uses what the book warns about: the ICC headline is the
heritability object of Ch. 4, and the era-dependence finding is the project independently
rediscovering Pearl's critique of it — a nice write-up point.

---

## 3. Direction: a DoWhy example notebook (upstream contribution)

**Verdict: strong candidate, after reshaping into one notebook.**

Why it fits the `py-why/dowhy` gallery:

- **Fills a real gap.** Existing gcm examples (online shop, microservice RCA, medical
  counterfactual) all show the tools *working*. No example demonstrates **identification
  failure** — v1's "pipeline runs perfectly, answer is confidently backwards, here's the
  diagnosis" is the cautionary tale the docs lack.
- **Wide API coverage in one story:** `auto.assign_causal_mechanisms` → `fit` →
  `evaluate_causal_model` / `falsify_graph` → `intrinsic_causal_influence` →
  `interventional_samples` → `counterfactual_samples`.
- **Real, open, licensed data** (f1db, CC-BY-4.0) and a zero-onboarding domain question.

Required reshaping:

- **One self-contained `.ipynb`:** question + 5-node DAG → naive categorical SCM fails
  informatively → load **precomputed** skill/pace latents → separation works → ICC + sweeps +
  the Albon counterfactual → caveats (era-dependence, correlated roots).
- **The PyMC stage becomes shipped data.** DoWhy won't take a PyMC dependency and notebooks run
  in docs CI — check in a small parquet of fitted latents with a paragraph on the teammate
  trick and a link back to this repo. ("gcm consuming domain-derived latents" is itself a
  pattern their docs don't show.)
- **Runtime discipline:** reduced ICC sample counts, fixed seed.

Risks: the correlated-roots issue (criticism #1) — surface it in the notebook rather than hope
nobody notices; maintenance expectations (static data mitigates); tone ("how to use gcm well,"
not "my research project").

Path: CONTRIBUTING.md explicitly welcomes example notebooks; process is flexible. **Open an
issue / py-why Discussion first** with a two-paragraph pitch (the F1 question, the
identification-failure teaching angle, the precomputed-latents design) before polishing.
Fallback if it stalls: the same distillation is a PyData-style talk or blog post.

---

## 4. Direction: an interactive site (GitHub Pages / Vercel)

**Key architectural fact:** every interactive query the project supports is either a lookup
into posterior draws or a smooth function of (skill, pace) — none of it needs live Python. The
expensive machinery runs offline; a **fully static site** serves its outputs. No backend, no
cold starts, no costs.

Build step exports three JSON artifact kinds:

1. **Downsampled posterior draws** (~200 joint draws of `skill[driver, season]`,
   `pace[team, season]` ≈ a few hundred KB gzipped) — client-side JS computes credible
   intervals, P(A beats B), and random-walk projections. *Honest uncertainty survives the port.*
2. **A precomputed E[finish] mesh** over (skill, pace, circuit_type) from the SCM's
   interventional samples — bilinear interpolation answers "any driver in any car" instantly.
3. **Small tables that already exist:** `models/incident_rates_2018_2025.json`, teammate-graph
   edges, per-window attribution results.

Interactives, in rough order of payoff:

- **The car-swap machine** — driver × car dropdowns → expected finish with credible band.
- **Senna-in-a-Red-Bull with the assumption as a toggle** — naive vs. era-normalized switch;
  the user *experiences* why the era-scale assumption matters (legends scatter to P11 naive,
  converge to P3 normalized).
- **Teammate-chain pathfinder** — shortest shared-teammate path between any two drivers
  (Senna → … → Verstappen); the identification story as a visual.
- **Era-window slider** — drag the start year, watch the car/driver split move
  (31.9/21.4 @ 2018 → 43.6/12.4 @ 2006) with a P(car>driver) gauge; precompute ~6–8 windows.
- **Career arcs + H2H lineup builder** — trajectories with credible bands; P(A out-qualifies B)
  from draws (`predict.py --lineup`, in-browser).

Platform: Pages vs. Vercel barely matters for static; **Vercel** if per-PR preview deploys are
wanted (fits the repo's PR workflow), otherwise Pages keeps it all in one repo with a CI step
regenerating JSON when models change. Small React or vanilla+D3 app; **no Pyodide** (posteriors
are precomputed). A Streamlit/HF Space would be faster to stand up but isn't a real site and
adds a running dependency.

Design principles carried over from the repo's DNA: **uncertainty always visible** (bands and
probabilities, never bare point estimates) and **off-support queries wear their asterisk in the
UI**, not a footnote.

Effort: weekend-to-a-week. Walking skeleton = export script + the car-swap machine.
