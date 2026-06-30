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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "data" / "f1_results.parquet"))
    ap.add_argument("--idata", default=str(ROOT / "models" / "v2_idata.pkl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "f1_scm_v2.parquet"))
    ap.add_argument("--skill-source", choices=["quali", "race", "combined"], default="race",
                    help="for a JOINT idata: which driver ability feeds driver_skill. race=racecraft "
                         "(default; finish_pos is a race outcome), quali=quali_skill, combined=mean "
                         "of the two standardized. Ignored for a legacy (single-skill) idata.")
    ap.add_argument("--pace-source", choices=["quali", "race"], default="race",
                    help="for a JOINT idata: pace_r (race, default) or pace_q (quali).")
    args = ap.parse_args()
    RESULTS, IDATA, OUT = Path(args.results).resolve(), Path(args.idata).resolve(), Path(args.out).resolve()

    idata = pickle.load(open(IDATA, "rb"))
    post = idata.posterior
    if "racecraft" in post:                       # JOINT quali+race model -> pick latents per flags
        def _z(da):                               # z-score across all (driver[,season]) entries
            return (da - float(da.mean())) / float(da.std())
        if args.skill_source == "combined":
            skill_da = ((_z(post["racecraft"]) + _z(post["quali_skill"])) / 2).mean(("chain", "draw"))
        else:
            skill_da = post["racecraft" if args.skill_source == "race" else "quali_skill"].mean(("chain", "draw"))
        pace = post["pace_r" if args.pace_source == "race" else "pace_q"].mean(("chain", "draw")).to_series()
        print(f"joint idata: driver_skill <- {args.skill_source}, car_pace <- {args.pace_source}")
    else:                                         # legacy single-skill idata
        skill_da = post["skill"].mean(("chain", "draw"))
        pace = post["pace"].mean(("chain", "draw")).to_series()               # index: team_year

    df = pd.read_parquet(RESULTS)  # started rows (classified flag inside)
    df["team_year"] = df.constructor_id + "@" + df.year.astype(str)
    if "season" in skill_da.dims:                     # time-varying skill -> map by (driver, year)
        skill = skill_da.to_pandas()                  # rows=driver, cols=season(year)
        df["driver_skill"] = [skill.loc[d, y] if (d in skill.index and y in skill.columns)
                              else np.nan for d, y in zip(df.driver_id, df.year)]
    else:                                             # constant skill -> map by driver
        df["driver_skill"] = df.driver_id.map(skill_da.to_series())
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
