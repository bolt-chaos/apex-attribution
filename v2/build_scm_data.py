"""v2 phase 2a: merge the posterior latents into the race-results dataframe.

Replaces v1's categorical driver_id / constructor_id (perfectly nested -> non-identified)
with the v2 posterior-mean CONTINUOUS quantities:
  driver_skill  — per driver        (lower = faster, % quali pace vs grid average)
  car_pace      — per team-year     (lower = faster, % off pole)

These are decoupled (corr ~0.49, vs v1's ~perfect nesting), so the SCM can separate them.
v1 finish_pos / grid / reliability columns are kept. Uses posterior MEANS (point estimates);
propagating full posterior uncertainty into the SCM is a later refinement.

Usage: python v2/build_scm_data.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "f1_results.parquet"
IDATA = ROOT / "models" / "v2_idata.pkl"
OUT = ROOT / "data" / "f1_scm_v2.parquet"


def main() -> int:
    idata = pickle.load(open(IDATA, "rb"))
    skill = idata.posterior["skill"].mean(("chain", "draw")).to_series()      # index: driver
    pace = idata.posterior["pace"].mean(("chain", "draw")).to_series()        # index: team_year

    df = pd.read_parquet(RESULTS)  # started rows (classified flag inside)
    df["team_year"] = df.constructor_id + "@" + df.year.astype(str)
    df["driver_skill"] = df.driver_id.map(skill)
    df["car_pace"] = df.team_year.map(pace)

    before = len(df)
    df = df.dropna(subset=["driver_skill", "car_pace"]).copy()
    dropped = before - len(df)

    df.to_parquet(OUT, index=False)
    print(f"merged latents -> {OUT.relative_to(ROOT)}")
    print(f"rows: {len(df)} (dropped {dropped} without a latent), classified: {int(df.classified.sum())}")
    print(f"corr(driver_skill, car_pace): {df.driver_skill.corr(df.car_pace):.2f}")
    print(f"driver_skill range: [{df.driver_skill.min():+.2f}, {df.driver_skill.max():+.2f}]  "
          f"car_pace range: [{df.car_pace.min():.2f}, {df.car_pace.max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
