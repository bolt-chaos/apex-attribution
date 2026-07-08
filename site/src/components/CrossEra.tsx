// "Senna in a modern Red Bull" — the crowd-pleaser and the most caveated thing in the project.
// A legend's skill was measured against THEIR field, whose spread was far wider (grids have
// converged). Comparing eras needs a scale assumption, so we show two translations side by side:
//   NAIVE  — raw skill straight into the modern model (wrong; shown only to demonstrate why).
//   ERA-NORMALIZED — how many SDs ahead of his field the legend was, mapped onto the modern spread.
// Both feed the cross-era SCM's own mesh (a different scale from the main site — see DATA.md).

import { useMemo, useState } from "react";
import type { CoreData } from "../lib/data";
import { finishBandAtPace, interpFinish } from "../lib/mesh";
import { fmtPos } from "../lib/posterior";
import { Select, type Option } from "./shared/Select";
import { CredibleBand } from "./shared/CredibleBand";

export function CrossEra({ data }: { data: CoreData }) {
  const ce = data.crossEra;
  const legendOpts: Option[] = useMemo(
    () =>
      Object.entries(ce.legends)
        .map(([value, l]) => ({ value, label: `${l.name} (${l.peak[0]}–${l.peak[1]})` }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    [ce],
  );
  const carOpts: Option[] = useMemo(
    () =>
      Object.entries(ce.cars)
        .map(([value, c]) => ({ value, label: c.name, year: Number(value.split("@")[1]) }))
        .sort((a, b) => b.year - a.year || a.label.localeCompare(b.label))
        .map(({ value, label }) => ({ value, label })),
    [ce],
  );

  const [legendId, setLegendId] = useState(ce.legends["ayrton-senna"] ? "ayrton-senna" : legendOpts[0].value);
  const [carId, setCarId] = useState(ce.cars["red-bull@2024"] ? "red-bull@2024" : carOpts[0].value);

  const legend = ce.legends[legendId];
  const car = ce.cars[carId];

  const { naive, norm, zMean, equivMean, versRef } = useMemo(() => {
    const naive = finishBandAtPace(ce.mesh, legend.rawDraws, car.pace);
    const equiv = legend.rawDraws.map((d) => ce.modMean + ((d - legend.eraMean) / legend.eraSd) * ce.modSd);
    const norm = finishBandAtPace(ce.mesh, equiv, car.pace);
    const zMean =
      legend.rawDraws.reduce((s, d) => s + (d - legend.eraMean) / legend.eraSd, 0) / legend.rawDraws.length;
    const equivMean = equiv.reduce((s, d) => s + d, 0) / equiv.length;
    const versRef = ce.verstappen2024 != null ? interpFinish(ce.mesh, ce.verstappen2024, car.pace) : null;
    return { naive, norm, zMean, equivMean, versRef };
  }, [ce, legend, car]);

  return (
    <section className="feature">
      <h2>Cross-era: a legend in a modern car</h2>
      <p className="feature__lede">
        Where would a past great land in today's machinery? Their skill was measured against a{" "}
        <em>much wider</em> field, so the eras only compare after a scale assumption. Watch it matter:
        toggle between the naive translation and the era-normalized one.
      </p>

      <div className="xera__caveat" role="note">
        <strong>Read this first.</strong> {ce.caveat}
      </div>

      <div className="controls">
        <Select label="Legend" value={legendId} options={legendOpts} onChange={setLegendId} />
        <span className="controls__in">in a</span>
        <Select label="Target car" value={carId} options={carOpts} onChange={setCarId} />
      </div>

      <div className="xera__pair">
        <div className="xera__card xera__card--naive">
          <div className="xera__tag">Naive — wrong on purpose</div>
          <div className="xera__big">{fmtPos(naive.med)}</div>
          <div className="xera__range">
            90% {fmtPos(naive.lo)}–{fmtPos(naive.hi)}
          </div>
          <CredibleBand lo={naive.lo} med={naive.med} hi={naive.hi} />
          <p className="xera__explain">
            Raw skill dropped straight into the modern model, as if a gap in {legend.peak[0]} meant the
            same as one today. It doesn't.
          </p>
        </div>

        <div className="xera__card xera__card--norm">
          <div className="xera__tag">Era-normalized</div>
          <div className="xera__big">{fmtPos(norm.med)}</div>
          <div className="xera__range">
            90% {fmtPos(norm.lo)}–{fmtPos(norm.hi)}
          </div>
          <CredibleBand
            lo={norm.lo}
            med={norm.med}
            hi={norm.hi}
            reference={versRef != null ? { pos: versRef, label: `Verstappen ’24 (${fmtPos(versRef)})` } : undefined}
          />
          <p className="xera__explain">
            {legend.name} was <strong>{Math.abs(zMean).toFixed(1)} SD</strong> ahead of his field —
            mapped onto today's tighter spread, an equivalent gap of {equivMean.toFixed(2)}%. That's
            what feeds the model here.
          </p>
        </div>
      </div>

      <p className="feature__note">
        The dashed marker is peak Verstappen in the same car, for scale. Era-normalized, the greats
        land roughly where he does — utterly dominant, but not physically impossible. The honest answer
        to “Senna in a modern Red Bull” is “a front-running, title-winning combination” — with the loud
        asterisk that cross-era skill is an <em>assumption</em>, not a measurement.
      </p>
    </section>
  );
}
