// Plain-language "how it works" + the honest caveats in one place. Links out to the repo's deeper
// writeups (docs/CONCEPTS.md, docs/BOOK_OF_WHY.md) for anyone who wants the full story.

import type { Necessity } from "../lib/data";

const REPO = "https://github.com/bolt-chaos/apex-attribution";

export function About({ necessity }: { necessity?: Necessity }) {
  return (
    <section className="feature about">
      <h2>How this works</h2>

      <p className="feature__lede">
        Verstappen wins a lot <em>and</em> drives a great car. So is he winning because he's brilliant,
        or because the machine is? Plain win-counting can't tell them apart. This whole project is one
        long attempt to separate the two — and then let you play with the result.
      </p>

      <h3>The trick: teammates share a car</h3>
      <p>
        Two teammates drive the <strong>same car</strong>, so when one out-qualifies the other, the car
        can't be the reason — it cancels. That gap is pure driver skill. Chain those comparisons across
        seasons (A raced B, B raced C…) and you can rank drivers who never met — the{" "}
        <a href={`${REPO}#readme`}>connectivity</a> the “teammate chain” tab draws out. A Bayesian model
        turns those gaps into a skill number for every driver and a pace number for every car, each with
        honest uncertainty.
      </p>

      {necessity && (
        <>
          <h3>Why the headline doesn't sum to 100%</h3>
          <p>
            The front page says a podium needed the car <strong>{necessity.carPct}%</strong> of the
            time and the driver <strong>{necessity.driverPct}%</strong> — more than 100%, on purpose.
            They answer two <em>separate</em> counterfactual questions, asked of all{" "}
            {necessity.nPodiums} podiums in {necessity.era}: rewind the race keeping everything else
            the same, downgrade <em>just the car</em> to mid-field — does the podium survive? Then
            rewind again and downgrade <em>just the driver</em>. One podium can fail both tests: a
            fire needs the match <em>and</em> the oxygen, so "necessary" isn't a pie chart. In fact,
            since {necessity.carPct} + {necessity.driverPct} &gt; 100, at least{" "}
            {necessity.carPct + necessity.driverPct - 100}% of podiums needed <strong>both</strong> —
            remove either ingredient and the result evaporates. Pundit percentages ("that win was 70%
            car") assume every result has one necessary cause; the model says that's the wrong
            question. Most driver-dependent podiums:{" "}
            {necessity.mostDriverDependent.map((d) => d.name).join(", ")} — most car-dependent:{" "}
            {necessity.mostCarDependent.map((d) => d.name).join(", ")}.
          </p>
        </>
      )}

      <h3>What each tab does</h3>
      <ul className="about__list">
        <li>
          <strong>Car swap</strong> — hold a driver fixed, change the car: an interventional{" "}
          <em>do()</em>-query answering “where would they finish?”
        </li>
        <li>
          <strong>Era slider</strong> — how the car-vs-driver split shifts with the window of seasons
          (there is no single number — that's the finding).
        </li>
        <li>
          <strong>Cross-era</strong> — a legend in a modern car, under an explicit, visible era-scale
          assumption.
        </li>
        <li>
          <strong>Teammate chain</strong> — the shortest shared-car path between any two drivers.
        </li>
        <li>
          <strong>Career arcs & head-to-head</strong> — skill over time, and P(A out-qualifies B).
        </li>
      </ul>

      <h3>Read the numbers honestly</h3>
      <ul className="about__list about__list--caveats">
        <li>
          Every estimate carries <strong>uncertainty</strong> — trust the band, not the point.
        </li>
        <li>
          Expected finish is a <strong>circuit average</strong>; it doesn't yet know Monaco from Monza.
        </li>
        <li>
          Car-vs-driver shares are <strong>era-dependent</strong> and descriptive, not a fixed law.
        </li>
        <li>
          Cross-era is an <strong>illustration under an assumption</strong>, not a measurement — its
          source model isn't fully converged, so its bands are wide on purpose.
        </li>
        <li>This is a model of <strong>pace</strong>, not a prophecy. It's a good bar-argument settler, not an oracle.</li>
      </ul>

      <h3>Go deeper</h3>
      <p className="about__links">
        <a href={`${REPO}/blob/main/docs/CONCEPTS.md`}>Concepts guide</a>
        <a href={`${REPO}/blob/main/docs/BOOK_OF_WHY.md`}>Mapping to “The Book of Why”</a>
        <a href={`${REPO}/blob/main/README.md`}>Full methodology &amp; results</a>
        <a href={REPO}>Source code</a>
      </p>
      <p className="feature__note">
        Built with a Bayesian hierarchical skill model (PyMC) feeding a structural causal model (DoWhy{" "}
        <code>gcm</code>) over the open f1db dataset. Everything you see here is precomputed — the site
        is fully static.
      </p>
    </section>
  );
}
