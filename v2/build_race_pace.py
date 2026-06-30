"""v2: build the RACE-pace dataframe — a second skill signal alongside qualifying.

Qualifying pace ([`build_quali.py`](build_quali.py)) measures one-lap speed. Race pace measures
*racecraft*: tyre management, race-day consistency, pace under fuel. Same teammate logic — within a
constructor-season the two drivers share the car, so their race-pace gap is (mostly) a skill gap —
but race pace is NOISIER (strategy, safety cars, traffic, tyres), so downstream it gets its own,
larger noise term and is treated as a complement to qualifying, not a replacement.

Outcome column `race_pct_gap` = 100 * gap_to_winner_millis / winner_race_time_millis, i.e. the
driver's time deficit to the winner as a percent of race distance (0 = won / matched the winner's
average pace). This is the race analogue of quali's `pct_gap` (both are 100*(t/ref - 1)), and is
comparable across circuits of different lengths.

Known approximations (documented, like build_quali.py's caveats):
- LAPPED cars (~31% of classified finishers) have no millisecond gap, only a whole-lap gap. We fold
  them in as `race_gap_laps * avg_lap_millis` and flag `lapped=True` — coarse (a lapped car sits
  somewhere within that lap) but it preserves the real within-team signal "my teammate got lapped
  and I didn't." Use `--drop-lapped` for a sensitivity run.
- PIT-STOP time is NOT removed. Teammates run similar stop counts, so it largely nets out within a
  team, and the StudentT tails absorb the residual. `race_pit_stops` is carried as a diagnostic only.
- SAFETY CARS / red flags compress gaps; f1db has no SC-lap column to filter on. Mitigation is
  structural, not a filter: the teammate *difference* identifies skill and both teammates are
  compressed together; season averaging smooths event shocks; race pace carries a larger sigma.
- DNFs are dropped: a retirement has no clean pace (it belongs to the reliability/incident track).

Usage: python v2/build_race_pace.py [--start 2018] [--end 2025] [--out-tag _2018_2025]
                                    [--drop-lapped]
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
RACE_GAP_CAP = 8.0  # %; race-distance gaps above this are multi-lap-down / heavily-delayed -> drop


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out-tag", default="", help="suffix for output files, e.g. _2018_2025")
    ap.add_argument("--drop-lapped", action="store_true",
                    help="drop lapped cars instead of folding them in (sensitivity run)")
    args = ap.parse_args()
    START, END = args.start, args.end
    out_parquet = ROOT / "data" / f"f1_race_pace{args.out_tag}.parquet"
    out_csv = ROOT / "data" / f"f1_race_pace{args.out_tag}.csv"

    con = sqlite3.connect(DB)
    q = """
    SELECT rd.race_id, r.year, r.round, r.circuit_id, r.circuit_type,
           rd.driver_id, rd.constructor_id,
           rd.position_number AS finish_pos, rd.race_laps,
           rd.race_time_millis, rd.race_gap_millis, rd.race_gap_laps, rd.race_pit_stops
    FROM race_data rd JOIN race r ON r.id = rd.race_id
    WHERE rd.type = 'RACE_RESULT' AND r.year BETWEEN ? AND ?
    """
    df = pd.read_sql(q, con, params=(START, END))
    con.close()
    n0 = len(df)

    # classified finishers only (DNFs have no clean pace)
    df = df[df.finish_pos.notna()].copy()
    n_dnf = n0 - len(df)

    # per-race winner reference: total time and average lap (for the lapped approximation)
    winners = df[df.finish_pos == 1].set_index("race_id")
    df["winner_time"] = df.race_id.map(winners.race_time_millis)
    df["winner_laps"] = df.race_id.map(winners.race_laps)
    df = df[df.winner_time.notna() & (df.winner_time > 0)].copy()  # need a timed winner
    df["avg_lap"] = df.winner_time / df.winner_laps

    # gap to winner in millis: leader = 0; same-lap = race_gap_millis; lapped ~ laps * avg_lap
    df["lapped"] = df.race_gap_millis.isna() & df.race_gap_laps.notna() & (df.finish_pos > 1)
    gap_ms = df.race_gap_millis.copy()
    gap_ms[df.finish_pos == 1] = 0.0
    gap_ms[df.lapped] = df.race_gap_laps[df.lapped] * df.avg_lap[df.lapped]
    df["gap_ms"] = gap_ms

    if args.drop_lapped:
        df = df[~df.lapped].copy()
    n_pre_drop = len(df)
    df = df[df.gap_ms.notna()].copy()        # drop finishers with neither a ms nor a lap gap
    n_nogap = n_pre_drop - len(df)

    df["race_pct_gap"] = df.gap_ms / df.winner_time * 100.0

    n_pre_cap = len(df)
    df = df[df.race_pct_gap <= RACE_GAP_CAP].copy()
    n_capped = n_pre_cap - len(df)

    # restrict to the largest teammate-connected component (same lever as v1 / build_quali)
    cohort, comps = largest_connected_cohort(df, min_shared=3)
    df = df[df.driver_id.isin(cohort)].copy()

    df["team_year"] = df.constructor_id + "@" + df.year.astype(str)
    cols = ["year", "round", "race_id", "circuit_id", "circuit_type",
            "driver_id", "constructor_id", "team_year", "finish_pos", "race_laps",
            "race_pct_gap", "lapped", "race_pit_stops"]
    df = df[cols].sort_values(["year", "round", "race_pct_gap"]).reset_index(drop=True)

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print("=" * 60)
    print(f"v2 RACE-PACE BUILD SUMMARY  ({START}-{END}"
          f"{', drop-lapped' if args.drop_lapped else ''})")
    print("=" * 60)
    print(f"rows: {len(df)}  | dropped: DNF {n_dnf}, no-gap {n_nogap}, >{RACE_GAP_CAP}% gap {n_capped}")
    print(f"lapped (folded in): {int(df.lapped.sum())}  ({df.lapped.mean():.0%} of kept rows)")
    print(f"drivers: {df.driver_id.nunique()}  team-years: {df.team_year.nunique()}  "
          f"races: {df.race_id.nunique()}")
    print(f"teammate components (sizes): {[len(c) for c in comps]}")
    print(f"\nrace_pct_gap: median {df.race_pct_gap.median():.2f}%  mean {df.race_pct_gap.mean():.2f}%  "
          f"p95 {df.race_pct_gap.quantile(.95):.2f}%  max {df.race_pct_gap.max():.2f}%")
    print("\nentries per driver (head):")
    print(df.driver_id.value_counts().head(8).to_string())
    print(f"\nWrote {out_parquet.relative_to(ROOT)} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
