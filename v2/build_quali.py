"""v2 phase 1: build the qualifying-pace dataframe for the hierarchical skill/pace model.

Why qualifying pace: it's the cleanest car-equalized skill signal (continuous, free of
DNF/strategy/lap-1 noise). Teammates share the car, so within a constructor-season their
pace gap is (mostly) a skill gap; drivers switching teams across seasons chain the
constructor-season pace estimates onto one scale.

Outcome column `pct_gap` = 100 * (driver_best_lap / race_pole_lap - 1), i.e. percent off
the fastest qualifying lap of that weekend (0 = pole). Comparable across circuits.

Known approximation (documented, like v1's caveats): `best` is each driver's fastest lap
across Q1/Q2/Q3, so it mixes sessions with slightly different track/fuel states. A driver
eliminated in Q1 is compared on their Q1 lap vs a pole set in Q3. Good enough for a first
identification model; a session-matched normalization is a later refinement.

Usage: python v2/build_quali.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_dataset import largest_connected_cohort  # reuse v1 connectivity logic

DB = ROOT / "data" / "f1db.sqlite"
GAP_CAP = 10.0  # %; dry-quali gaps above this are aborted/wet/incident artifacts -> drop


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2022)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out-tag", default="", help="suffix for output files, e.g. _2018_2025")
    ap.add_argument("--gap-method", choices=["pole", "session"], default="pole",
                    help="pole: gap to the race's overall pole (legacy). "
                         "session: gap to the fastest lap IN THE SAME SESSION (removes the "
                         "knockout Q1-vs-Q3 track-evolution bias; comparable across the 2006 "
                         "single-session/knockout format boundary).")
    args = ap.parse_args()
    START, END = args.start, args.end
    out_parquet = ROOT / "data" / f"f1_quali{args.out_tag}.parquet"
    out_csv = ROOT / "data" / f"f1_quali{args.out_tag}.csv"

    con = sqlite3.connect(DB)
    q = """
    SELECT rd.race_id, r.year, r.round, r.circuit_id, r.circuit_type,
           rd.driver_id, rd.constructor_id,
           rd.qualifying_q1_millis AS q1, rd.qualifying_q2_millis AS q2,
           rd.qualifying_q3_millis AS q3,
           rd.qualifying_time_millis AS qt, rd.position_number AS qpos
    FROM race_data rd JOIN race r ON r.id = rd.race_id
    WHERE rd.type = 'QUALIFYING_RESULT' AND r.year BETWEEN ? AND ?
    """
    df = pd.read_sql(q, con, params=(START, END))
    con.close()
    n0 = len(df)

    # best lap = fastest available across sessions. 2006+ uses the q1/q2/q3 knockout split;
    # pre-2006 uses the single-session qualifying_time_millis (qt). min() spans both eras
    # (qt is NaN in the modern era, q1/q2/q3 NaN in the old era), enabling a back-to-1980 reach.
    df["best"] = df[["q1", "q2", "q3", "qt"]].min(axis=1)
    df = df[df.best.notna()].copy()
    n_nolap = n0 - len(df)

    if args.gap_method == "pole":
        df["pole"] = df.groupby("race_id").best.transform("min")
        df["pct_gap"] = (df.best / df.pole - 1.0) * 100.0
    else:  # session-relative: each lap vs its own session's fastest; per driver take the best gap
        long = df.melt(id_vars=["race_id", "driver_id"], value_vars=["q1", "q2", "q3", "qt"],
                       var_name="session", value_name="millis").dropna(subset=["millis"])
        long["sess_pole"] = long.groupby(["race_id", "session"]).millis.transform("min")
        long["gap"] = long.millis / long.sess_pole - 1.0
        best_gap = long.groupby(["race_id", "driver_id"]).gap.min().mul(100.0).rename("pct_gap")
        df = df.merge(best_gap, on=["race_id", "driver_id"], how="left")

    n_pre_cap = len(df)
    df = df[df.pct_gap <= GAP_CAP].copy()
    n_capped = n_pre_cap - len(df)

    # restrict to the largest teammate-connected component (same lever as v1)
    cohort, comps = largest_connected_cohort(df, min_shared=3)
    df = df[df.driver_id.isin(cohort)].copy()

    df["team_year"] = df.constructor_id + "@" + df.year.astype(str)
    cols = ["year", "round", "race_id", "circuit_id", "circuit_type",
            "driver_id", "constructor_id", "team_year", "best", "pct_gap", "qpos"]
    df = df[cols].sort_values(["year", "round", "pct_gap"]).reset_index(drop=True)

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print("=" * 60)
    print(f"v2 QUALI-PACE BUILD SUMMARY  ({START}-{END}, gap-method={args.gap_method})")
    print("=" * 60)
    print(f"rows: {len(df)}  | dropped: no-lap {n_nolap}, >{GAP_CAP}% gap {n_capped}")
    print(f"drivers: {df.driver_id.nunique()}  team-years: {df.team_year.nunique()}  "
          f"races: {df.race_id.nunique()}")
    print(f"teammate components (sizes): {[len(c) for c in comps]}")
    print(f"\npct_gap: median {df.pct_gap.median():.2f}%  mean {df.pct_gap.mean():.2f}%  "
          f"p95 {df.pct_gap.quantile(.95):.2f}%  max {df.pct_gap.max():.2f}%")
    print("\nteam-years per constructor:")
    print(df.groupby('constructor_id').team_year.nunique().to_string())
    print("\nentries per driver (head):")
    print(df.driver_id.value_counts().head(8).to_string())
    print(f"\nWrote {out_parquet.relative_to(ROOT)} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
