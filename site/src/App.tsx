import { useEffect, useRef, useState } from "react";
import { loadCore, type CoreData } from "./lib/data";
import { CarSwap } from "./components/CarSwap";
import { EraSlider } from "./components/EraSlider";
import { CrossEra } from "./components/CrossEra";
import { Pathfinder } from "./components/Pathfinder";
import { CareerArcs } from "./components/CareerArcs";
import { Lineup } from "./components/Lineup";
import { About } from "./components/About";

const TABS = [
  { id: "car-swap", label: "Car swap" },
  { id: "era", label: "Era slider" },
  { id: "cross-era", label: "Cross-era" },
  { id: "chain", label: "Teammate chain" },
  { id: "arcs", label: "Career arcs" },
  { id: "h2h", label: "Head-to-head" },
  { id: "about", label: "How it works" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [data, setData] = useState<CoreData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("car-swap");
  // Which edges of the mobile tab strip get a fade hint: none / more-right / both / more-left.
  const [navFade, setNavFade] = useState<"none" | "right" | "both" | "left">("none");
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    loadCore().then(setData).catch((e) => setError(String(e)));
  }, []);

  // Show the fade only on the side that actually has more tabs, so the last pill (How it works)
  // isn't left dimmed under a permanent right-edge fade once you've scrolled to the end.
  const updateNavFade = () => {
    const nav = navRef.current;
    if (!nav) return;
    const scrollable = nav.scrollWidth - nav.clientWidth;
    if (scrollable <= 1) return setNavFade("none");
    const atStart = nav.scrollLeft <= 1;
    const atEnd = nav.scrollLeft >= scrollable - 1;
    setNavFade(atStart ? "right" : atEnd ? "left" : "both");
  };

  useEffect(() => {
    updateNavFade();
    window.addEventListener("resize", updateNavFade);
    return () => window.removeEventListener("resize", updateNavFade);
  }, [data]);

  // On mobile the tabs are a horizontal scroll strip; keep the active pill in view after a tap.
  // Scroll ONLY the nav strip — not with Element.scrollIntoView, which also scrolls the document
  // horizontally to centre a right-side pill and drags the whole page left.
  useEffect(() => {
    const nav = navRef.current;
    const active = nav?.querySelector<HTMLElement>(".is-active");
    if (!nav || !active) return;
    const navRect = nav.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    const target =
      nav.scrollLeft + (activeRect.left - navRect.left) - (nav.clientWidth - activeRect.width) / 2;
    nav.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
  }, [tab]);

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead__inner">
          <h1>
            Apex<span className="masthead__accent">Attribution</span>
          </h1>
          <p className="masthead__tagline">
            How much of a Formula 1 result is the <strong>driver</strong>, and how much is the{" "}
            <strong>car</strong>? A causal model you can play with.
          </p>
        </div>
      </header>

      <nav className="tabs" aria-label="Interactives" ref={navRef} data-fade={navFade} onScroll={updateNavFade}>
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tabs__btn ${tab === t.id ? "is-active" : ""}`}
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        {error && (
          <p className="status status--error">
            Couldn't load the model data ({error}). If you're viewing this locally, make sure the site
            was built from the repo root so <code>public/data</code> is present.
          </p>
        )}
        {!error && !data && tab !== "about" && <p className="status">Loading the model…</p>}
        {data && tab === "car-swap" && <CarSwap data={data} />}
        {data && tab === "era" && <EraSlider data={data} />}
        {data && tab === "cross-era" && <CrossEra data={data} />}
        {data && tab === "chain" && <Pathfinder data={data} />}
        {data && tab === "arcs" && <CareerArcs data={data} />}
        {data && tab === "h2h" && <Lineup data={data} />}
        {tab === "about" && <About />}
      </main>

      <footer className="footer">
        <p>
          Built from a Bayesian causal model (DoWhy&nbsp;<code>gcm</code>) over the f1db dataset. Every
          number carries uncertainty on purpose. This is a model of pace, not a prophecy —{" "}
          <a href="https://github.com/bolt-chaos/apex-attribution">read how it works</a>.
        </p>
      </footer>
    </div>
  );
}
