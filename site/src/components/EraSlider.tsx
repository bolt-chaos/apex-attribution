// Drag the start year and watch the car/driver split move. The point of the feature IS the movement:
// a variance share is a property of the population, not a fixed law of the sport — widen the window
// to include more varied cars and the car's share grows. This is the project's "there is no single
// X%/Y%" finding made tangible (and why ICC is a demoted, descriptive number).

import { useState } from "react";
import type { CoreData, EraRow } from "../lib/data";

const BARW = 520;
const PAD = 110;

function Bar({ y, pct, max, color, label }: { y: number; pct: number; max: number; color: string; label: string }) {
  const w = (pct / max) * (BARW - PAD - 20);
  return (
    <g>
      <text x={PAD - 12} y={y + 15} className="era__barlabel">
        {label}
      </text>
      <rect x={PAD} y={y} width={Math.max(2, w)} height={22} rx={4} fill={color} />
      <text x={PAD + Math.max(2, w) + 8} y={y + 16} className="era__barval">
        {pct.toFixed(0)}%
      </text>
    </g>
  );
}

export function EraSlider({ data }: { data: CoreData }) {
  const eras: EraRow[] = data.era;
  const [i, setI] = useState(0);
  const e = eras[i];
  const maxPct = Math.ceil(Math.max(...eras.map((r) => Math.max(r.carPct, r.driverPct))) / 10) * 10;

  return (
    <section className="feature">
      <h2>The era slider</h2>
      <p className="feature__lede">
        Of all the variation in race results, how much does the <strong>car</strong> explain versus
        the <strong>driver</strong>? Drag the window's start year. There is no single answer — and
        that's the finding: the wider and more varied the field, the more the car explains.
      </p>

      <div className="era__slider">
        <input
          type="range"
          min={0}
          max={eras.length - 1}
          step={1}
          value={i}
          onChange={(ev) => setI(Number(ev.target.value))}
          aria-label="Era window"
          list="era-ticks"
        />
        <datalist id="era-ticks">
          {eras.map((r, k) => (
            <option key={r.label} value={k} />
          ))}
        </datalist>
        <div className="era__ticks">
          {eras.map((r, k) => (
            <button
              key={r.label}
              className={`era__tick ${k === i ? "is-active" : ""}`}
              onClick={() => setI(k)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="result">
        <div className="era__headline">
          Over <strong>{e.label}</strong>: the car explains{" "}
          <span className="era__num era__num--car">{e.carPct.toFixed(0)}%</span> of the variance in
          results, the driver <span className="era__num era__num--drv">{e.driverPct.toFixed(0)}%</span>.
        </div>
        <svg className="era__chart" viewBox={`0 0 ${BARW} 90`} role="img" aria-label={`Car ${e.carPct}% versus driver ${e.driverPct}% of variance`}>
          <Bar y={10} pct={e.carPct} max={maxPct} color="var(--accent)" label="Car" />
          <Bar y={46} pct={e.driverPct} max={maxPct} color="var(--band)" label="Driver" />
        </svg>
        <div className="era__spread">
          Held up another way — the graph-robust one — over this window the car swings expected finish
          by about <strong>{e.carSpread.toFixed(1)} places</strong> end to end, the driver by{" "}
          <strong>{e.driverSpread.toFixed(1)}</strong>.
        </div>
      </div>

      <p className="feature__note">
        These variance shares (an <em>intrinsic causal influence</em> decomposition) are{" "}
        <strong>descriptive, not a law</strong>: they depend on how varied the cars and drivers are in
        the window, which is exactly why the number moves as you drag. The interventional
        “places swung” figure is the more stable, graph-robust measure. Each stop is its own
        era-specific model fit — the slider snaps between measured windows rather than guessing between
        them.
      </p>
    </section>
  );
}
