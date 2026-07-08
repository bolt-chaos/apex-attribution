// Plain-language "how it works" + the honest caveats in one place. Links out to the repo's deeper
// writeups (CONCEPTS.md, BOOK_OF_WHY.md) for anyone who wants the full story.

const REPO = "https://github.com/bolt-chaos/apex-attribution";

export function About() {
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
        <a href={`${REPO}/blob/main/CONCEPTS.md`}>Concepts guide</a>
        <a href={`${REPO}/blob/main/BOOK_OF_WHY.md`}>Mapping to “The Book of Why”</a>
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
