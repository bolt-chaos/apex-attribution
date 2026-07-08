// The teammate-chain pathfinder — the identification story as a picture. Two drivers who never
// shared a car can still be compared by chaining through common teammates (A raced B, B raced C, …).
// We BFS the shortest such chain and light it up inside the full shared-teammate network.

import { useMemo, useState } from "react";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
} from "d3-force";
import type { CoreData } from "../lib/data";
import { buildAdjacency, shortestPath } from "../lib/graph";
import { Select, type Option } from "./shared/Select";

interface SimNode extends SimulationNodeDatum {
  id: string;
}
interface SimLink {
  source: string;
  target: string;
}

const W = 900;
const H = 560;

// Run the force layout to a settled state ONCE (positions are static; the path only restyles).
function layout(nodeIds: string[], edges: { a: string; b: string }[]) {
  const nodes: SimNode[] = nodeIds.map((id) => ({ id }));
  const links: SimLink[] = edges.map((e) => ({ source: e.a, target: e.b }));
  const sim = forceSimulation(nodes)
    .force("link", forceLink<SimNode, SimLink>(links).id((d) => d.id).distance(42).strength(0.45))
    .force("charge", forceManyBody().strength(-42))
    .force("center", forceCenter(W / 2, H / 2))
    .force("collide", forceCollide(7))
    .stop();
  for (let i = 0; i < 320; i++) sim.tick();
  const pos = new Map<string, { x: number; y: number }>();
  for (const n of nodes) pos.set(n.id, { x: n.x ?? W / 2, y: n.y ?? H / 2 });
  return pos;
}

export function Pathfinder({ data }: { data: CoreData }) {
  const tm = data.teammates;
  const nameOf = useMemo(() => new Map(tm.nodes.map((n) => [n.id, n.name])), [tm]);
  const options: Option[] = useMemo(
    () =>
      tm.nodes
        .map((n) => ({ value: n.id, label: n.name }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    [tm],
  );
  const adj = useMemo(() => buildAdjacency(tm.edges), [tm]);
  const pos = useMemo(() => layout(tm.nodes.map((n) => n.id), tm.edges), [tm]);

  const has = (id: string) => nameOf.has(id);
  const [a, setA] = useState(has("ayrton-senna") ? "ayrton-senna" : options[0].value);
  const [b, setB] = useState(has("max-verstappen") ? "max-verstappen" : options[options.length - 1].value);

  const path = useMemo(() => shortestPath(adj, a, b), [adj, a, b]);
  const pathNodes = useMemo(() => new Set(path?.nodes ?? []), [path]);
  const pathLinks = useMemo(
    () => new Set((path?.links ?? []).map((l) => `${l.a}|${l.b}`).flatMap((k) => [k, k.split("|").reverse().join("|")])),
    [path],
  );

  return (
    <section className="feature">
      <h2>The teammate chain</h2>
      <p className="feature__lede">
        Senna and Verstappen never shared a car — so how can the model compare them? Through a chain of
        common teammates. Every line below is two drivers who drove the <em>same car</em>; that shared
        machinery is what cancels out to leave pure skill. Pick any two.
      </p>

      <div className="controls">
        <Select label="From" value={a} options={options} onChange={setA} />
        <span className="controls__in">to</span>
        <Select label="To" value={b} options={options} onChange={setB} />
      </div>

      {path ? (
        <div className="chain">
          {path.nodes.map((id, k) => (
            <span key={id} className="chain__step">
              <span className={`chain__chip ${id === a || id === b ? "is-end" : ""}`}>{nameOf.get(id)}</span>
              {k < path.links.length && (
                <span className="chain__link">
                  <span className="chain__arrow">→</span>
                  <span className="chain__team">{path.links[k].teams[0].replace("@", " ")}</span>
                </span>
              )}
            </span>
          ))}
          <p className="chain__caption">
            {path.links.length === 1
              ? "Direct teammates — one shared car links them."
              : `${path.links.length} hops: comparable through ${path.links.length - 1} intermediate driver${path.links.length - 1 === 1 ? "" : "s"}.`}
          </p>
        </div>
      ) : (
        <p className="chain__none">No shared-teammate chain connects these two in the data.</p>
      )}

      <svg className="net" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Shared-teammate network with the chosen chain highlighted">
        {tm.edges.map((e, i) => {
          const p = pos.get(e.a)!;
          const q = pos.get(e.b)!;
          const on = pathLinks.has(`${e.a}|${e.b}`);
          return (
            <line
              key={i}
              x1={p.x}
              y1={p.y}
              x2={q.x}
              y2={q.y}
              className={on ? "net__link is-path" : "net__link"}
            />
          );
        })}
        {tm.nodes.map((n) => {
          const p = pos.get(n.id)!;
          const on = pathNodes.has(n.id);
          const end = n.id === a || n.id === b;
          return (
            <g key={n.id}>
              <circle cx={p.x} cy={p.y} r={end ? 7 : on ? 5 : 2.5} className={end ? "net__node is-end" : on ? "net__node is-path" : "net__node"} />
              {on && (
                <text x={p.x} y={p.y - 10} className={`net__label ${end ? "is-end" : ""}`}>
                  {nameOf.get(n.id)}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <p className="feature__note">
        The whole grid is one connected web — that connectivity (engineered by widening the seasons
        included) is precisely what makes “driver skill” recoverable on a single scale. A driver with
        no shared-teammate link to the rest can't be placed, and is left out of the model.
      </p>
    </section>
  );
}
