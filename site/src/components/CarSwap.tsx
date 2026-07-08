// The marquee interactive: put any driver in any car and see where they'd finish.
// Driver skill (career-pooled posterior draws) x car pace (posterior draws) -> E[finish] band.

import { useMemo, useState } from "react";
import type { CoreData } from "../lib/data";
import { finishBand } from "../lib/mesh";
import { fmtPos } from "../lib/posterior";
import { Select, type Option } from "./shared/Select";
import { CredibleBand } from "./shared/CredibleBand";

function driverOptions(data: CoreData): Option[] {
  return Object.entries(data.drivers)
    .map(([id, d]) => ({ value: id, label: d.name, last: Math.max(...d.seasons) }))
    .sort((a, b) => b.last - a.last || a.label.localeCompare(b.label))
    .map(({ value, label }) => ({ value, label }));
}

function carOptions(data: CoreData): Option[] {
  return Object.entries(data.cars)
    .map(([id, c]) => ({ value: id, label: c.name, year: c.year }))
    .sort((a, b) => b.year - a.year || a.label.localeCompare(b.label))
    .map(({ value, label, year }) => ({ value, label, group: String(year) }));
}

export function CarSwap({ data }: { data: CoreData }) {
  const drivers = useMemo(() => driverOptions(data), [data]);
  const cars = useMemo(() => carOptions(data), [data]);
  const [driverId, setDriverId] = useState(
    data.drivers["max-verstappen"] ? "max-verstappen" : drivers[0].value,
  );
  const [carId, setCarId] = useState(data.cars["williams@2024"] ? "williams@2024" : cars[0].value);

  const driver = data.drivers[driverId];
  const car = data.cars[carId];

  const band = useMemo(
    () => finishBand(data.mesh, driver.career.draws, car.draws),
    [data.mesh, driver, car],
  );

  return (
    <section className="feature">
      <h2>The car-swap machine</h2>
      <p className="feature__lede">
        Hold the driver fixed, change the car. Where would <strong>{driver.name}</strong> finish in
        the <strong>{car.name}</strong>? The model keeps the driver's skill and swaps in the car's
        pace — a <em>do()</em>-intervention, not a guess.
      </p>

      <div className="controls">
        <Select label="Driver" value={driverId} options={drivers} onChange={setDriverId} />
        <span className="controls__in">in a</span>
        <Select label="Car" value={carId} options={cars} onChange={setCarId} />
      </div>

      <div className="result">
        <div className="result__headline">
          <span className="result__big">{fmtPos(band.med)}</span>
          <span className="result__sub">
            expected finish · 90% range {fmtPos(band.lo)}–{fmtPos(band.hi)}
          </span>
        </div>
        <CredibleBand lo={band.lo} med={band.med} hi={band.hi} />
      </div>

      <p className="feature__note">
        Lower is better (P1 = win). The band is the model's honest uncertainty — it widens for
        drivers with fewer races and for cars whose pace is harder to pin down. Finishing position is
        averaged over circuit types.
      </p>
    </section>
  );
}
