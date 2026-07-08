// Skill trajectories over time, with credible bands. Each driver's skill follows a per-season random
// walk in the model, so we can draw the arc of a career — with the uncertainty that widens where a
// driver has fewer races. Overlay several to compare.

import { useMemo, useState } from "react";
import type { CoreData } from "../lib/data";
import { Select } from "./shared/Select";

const PALETTE = ["#e10600", "#3a8ee6", "#2ecc71", "#f39c12", "#b07cff", "#1abac6"];
const W = 720;
const H = 380;
const PADX = 44;
const PADY = 28;

export function CareerArcs({ data }: { data: CoreData }) {
  const allSeasons = useMemo(() => {
    const s = new Set<number>();
    Object.values(data.drivers).forEach((d) => d.seasons.forEach((y) => s.add(y)));
    return [...s].sort((a, b) => a - b);
  }, [data]);
  const y0 = allSeasons[0];
  const y1 = allSeasons[allSeasons.length - 1];

  const defaults = ["max-verstappen", "lewis-hamilton", "lando-norris"].filter((id) => data.drivers[id]);
  const [selected, setSelected] = useState<string[]>(
    defaults.length ? defaults : [Object.keys(data.drivers)[0]],
  );

  const addOpts = useMemo(
    () =>
      Object.entries(data.drivers)
        .filter(([id]) => !selected.includes(id))
        .map(([id, d]) => ({ value: id, label: d.name }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    [data, selected],
  );

  // y-range over selected drivers' bands (skill: lower = faster; we invert so faster is up)
  const { yMin, yMax } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    selected.forEach((id) => {
      Object.values(data.drivers[id].bySeason).forEach((b) => {
        lo = Math.min(lo, b.lo);
        hi = Math.max(hi, b.hi);
      });
    });
    if (!isFinite(lo)) {
      lo = -0.5;
      hi = 0.5;
    }
    const pad = (hi - lo) * 0.08 || 0.1;
    return { yMin: lo - pad, yMax: hi + pad };
  }, [data, selected]);

  const x = (year: number) => PADX + ((year - y0) / Math.max(1, y1 - y0)) * (W - 2 * PADX);
  const y = (skill: number) => PADY + ((skill - yMin) / (yMax - yMin)) * (H - 2 * PADY);

  return (
    <section className="feature">
      <h2>Career arcs</h2>
      <p className="feature__lede">
        How a driver's underlying pace rises and fades over the seasons — with the model's honest
        uncertainty as a shaded band. Higher is faster. Add or remove drivers to compare.
      </p>

      <div className="arc__add">
        <Select label="Add driver" value="" options={[{ value: "", label: "＋ add a driver…" }, ...addOpts]} onChange={(v) => v && setSelected((s) => [...s, v])} />
        <div className="arc__chips">
          {selected.map((id, i) => (
            <span key={id} className="arc__chip" style={{ borderColor: PALETTE[i % PALETTE.length] }}>
              <span className="arc__dot" style={{ background: PALETTE[i % PALETTE.length] }} />
              {data.drivers[id].name}
              <button className="arc__x" onClick={() => setSelected((s) => s.filter((x) => x !== id))} aria-label={`remove ${data.drivers[id].name}`}>
                ×
              </button>
            </span>
          ))}
        </div>
      </div>

      <svg className="arc" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Driver skill trajectories over time">
        {/* season gridlines */}
        {allSeasons.map((yr) => (
          <g key={yr}>
            <line x1={x(yr)} y1={PADY} x2={x(yr)} y2={H - PADY} className="arc__grid" />
            <text x={x(yr)} y={H - 8} className="arc__axislabel">
              {yr}
            </text>
          </g>
        ))}
        <text x={12} y={PADY + 4} className="arc__axislabel arc__axislabel--y">
          faster
        </text>
        <text x={12} y={H - PADY} className="arc__axislabel arc__axislabel--y">
          slower
        </text>

        {selected.map((id, i) => {
          const d = data.drivers[id];
          const color = PALETTE[i % PALETTE.length];
          const yrs = d.seasons.filter((yr) => d.bySeason[String(yr)]).sort((a, b) => a - b);
          if (!yrs.length) return null;
          const band =
            yrs.map((yr) => `${x(yr)},${y(d.bySeason[String(yr)].lo)}`).join(" ") +
            " " +
            yrs
              .slice()
              .reverse()
              .map((yr) => `${x(yr)},${y(d.bySeason[String(yr)].hi)}`)
              .join(" ");
          const line = yrs.map((yr) => `${x(yr)},${y(d.bySeason[String(yr)].med)}`).join(" ");
          return (
            <g key={id}>
              <polygon points={band} fill={color} opacity={0.15} />
              <polyline points={line} fill="none" stroke={color} strokeWidth={2.4} />
              {yrs.map((yr) => (
                <circle key={yr} cx={x(yr)} cy={y(d.bySeason[String(yr)].med)} r={3} fill={color} />
              ))}
            </g>
          );
        })}
      </svg>

      <p className="feature__note">
        Skill here is qualifying-pace percentile from the joint model, per season. The band is the 90%
        credible interval — it's wider for part-season or rookie campaigns, exactly where the model
        should be less sure.
      </p>
    </section>
  );
}
