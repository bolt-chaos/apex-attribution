// Head-to-head: put two drivers on equal machinery and ask who's faster over a flying lap. Because
// the model carries full posterior uncertainty, the answer is a probability, not a verdict —
// P(A out-qualifies B) is the share of plausible worlds where A's skill draw beats B's.

import { useMemo, useState } from "react";
import type { CoreData } from "../lib/data";
import { pAhead } from "../lib/posterior";
import { Select, type Option } from "./shared/Select";

export function Lineup({ data }: { data: CoreData }) {
  const options: Option[] = useMemo(
    () =>
      Object.entries(data.drivers)
        .map(([value, d]) => ({ value, label: d.name, last: Math.max(...d.seasons) }))
        .sort((a, b) => b.last - a.last || a.label.localeCompare(b.label))
        .map(({ value, label }) => ({ value, label })),
    [data],
  );
  const has = (id: string) => !!data.drivers[id];
  const [aId, setA] = useState(has("lando-norris") ? "lando-norris" : options[0].value);
  const [bId, setB] = useState(has("max-verstappen") ? "max-verstappen" : options[1].value);

  const a = data.drivers[aId];
  const b = data.drivers[bId];
  const p = useMemo(() => pAhead(a.career.draws, b.career.draws), [a, b]);
  const pct = Math.round(p * 100);
  const leader = p >= 0.5 ? a : b;
  const leaderPct = p >= 0.5 ? pct : 100 - pct;

  return (
    <section className="feature">
      <h2>Head-to-head</h2>
      <p className="feature__lede">
        Same car, one flying lap — who comes out ahead? The model answers with a probability, because
        it's honestly unsure. This pools each driver's whole career.
      </p>

      <div className="controls">
        <Select label="Driver A" value={aId} options={options} onChange={setA} />
        <span className="controls__in">vs</span>
        <Select label="Driver B" value={bId} options={options} onChange={setB} />
      </div>

      <div className="result">
        <div className="h2h__headline">
          <strong>{leader.name}</strong> out-qualifies <strong>{leader === a ? b.name : a.name}</strong>{" "}
          in <span className="h2h__pct">{leaderPct}%</span> of plausible worlds
        </div>
        <div className="h2h__bar" role="img" aria-label={`${a.name} ${pct} percent versus ${b.name} ${100 - pct} percent`}>
          <div className="h2h__side h2h__side--a" style={{ width: `${pct}%` }}>
            {pct >= 12 && <span>{a.name.split(" ").slice(-1)} {pct}%</span>}
          </div>
          <div className="h2h__side h2h__side--b" style={{ width: `${100 - pct}%` }}>
            {100 - pct >= 12 && (
              <span>
                {b.name.split(" ").slice(-1)} {100 - pct}%
              </span>
            )}
          </div>
        </div>
        {aId === bId && <p className="h2h__same">Pick two different drivers.</p>}
      </div>

      <p className="feature__note">
        A coin-flip (≈50%) means the model genuinely can't separate them. Lopsided numbers reflect both
        a real skill gap <em>and</em> how confidently the data pins it down — two closely-matched greats
        can still land near 50/50.
      </p>
    </section>
  );
}
