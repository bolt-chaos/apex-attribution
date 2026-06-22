"""Phase (b): build the clean per-result dataframe for the causal model.

Produces one row per *started* race entry, for the v1 cohort (2022-2025, largest
teammate-connected component), with the columns the DAG (phase c) will consume.

Reproducible end-to-end: reads data/f1db.sqlite, recomputes the connected component
in-script (nothing about the cohort is hardcoded), writes data/f1_results.parquet
(+ .csv for eyeballing), and prints a summary. See SCHEMA_NOTES.md for the schema.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --start 2022 --end 2025 --min-shared 3
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "f1db.sqlite"
OUT_PARQUET = ROOT / "data" / "f1_results.parquet"
OUT_CSV = ROOT / "data" / "f1_results.csv"

PIT_LANE_GRID = 21  # PL starts begin behind the 20-car grid; impute as 21.

# --- DNF cause taxonomy (pitfall #3: never charge a mechanical failure to the driver) ---
# Explicit map of every race_reason_retired observed in 2022-2025. The critical split is
# MECHANICAL (-> reliability node) vs everything else. Unseen future reasons fall through
# the keyword fallback and, failing that, are logged and bucketed 'other'.
REASON_TO_CAUSE: dict[str, str] = {
    # mechanical / car reliability
    "Engine": "mechanical", "Power unit": "mechanical", "Power loss": "mechanical",
    "Gearbox": "mechanical", "Transmission": "mechanical", "Clutch": "mechanical",
    "Differential": "mechanical", "Driveshaft": "mechanical", "Hydraulics": "mechanical",
    "Brakes": "mechanical", "Suspension": "mechanical", "Steering": "mechanical",
    "Throttle": "mechanical", "Electrical": "mechanical", "Battery": "mechanical",
    "Overheating": "mechanical", "Cooling system": "mechanical", "Water leak": "mechanical",
    "Water pressure": "mechanical", "Oil leak": "mechanical", "Oil pressure": "mechanical",
    "Fuel system": "mechanical", "Fuel pump": "mechanical", "Fuel leak": "mechanical",
    "Turbo": "mechanical", "Exhaust": "mechanical", "Vibrations": "mechanical",
    "Handling": "mechanical", "Mechanical": "mechanical", "Wheel": "mechanical",
    "Undertray": "mechanical", "Rear wing": "mechanical", "Front wing": "mechanical",
    "Chassis": "mechanical", "Pit stop issue": "mechanical", "Wheel nut": "mechanical",
    # driver-caused (a real driver/racing outcome — feeds race_execution, not reliability)
    "Collision": "driver_error", "Collision damage": "driver_error", "Accident": "driver_error",
    "Accident damage": "driver_error", "Spun off": "driver_error",
    "Accident in qualifying": "driver_error", "Car damaged in qualifying": "driver_error",
    "Accident on formation lap": "driver_error",
    # other: bad luck, technical disqualification, withdrawal, driver unavailable
    "Puncture": "other", "Debris": "other",
    "Illegal skid block wear": "other", "Car underweight": "other",
    "Received outside assistance": "other", "Withdrew": "other",
    "Unwell": "other", "Injury": "other",
}
# keyword fallback for unseen reasons (checked in order; first hit wins)
_KEYWORDS = [
    ("mechanical", ("engine", "power", "gearbox", "transmission", "clutch", "hydraulic",
                    "brake", "suspension", "steering", "throttle", "electric", "battery",
                    "overheat", "cooling", "water", "oil", "fuel", "turbo", "exhaust",
                    "mechanic", "wheel", "wing", "chassis", "differential", "driveshaft",
                    "pit stop", "vibration", "undertray", "radiator")),
    ("driver_error", ("collision", "accident", "spun", "spin", "crash", "off")),
    ("other", ("underweight", "skid block", "disqualif", "withdrew", "withdrawn",
               "unwell", "injury", "puncture", "debris", "assistance")),
]


def classify_reason(reason: str | None) -> str:
    if reason is None or (isinstance(reason, float) and pd.isna(reason)):
        return "unknown"  # retired but no reason recorded -> treated as 'other' downstream
    if reason in REASON_TO_CAUSE:
        return REASON_TO_CAUSE[reason]
    low = reason.lower()
    for cause, kws in _KEYWORDS:
        if any(k in low for k in kws):
            return cause
    return "other"


def load_race_results(con: sqlite3.Connection, start: int, end: int) -> pd.DataFrame:
    q = """
    SELECT r.year, r.round, r.id AS race_id, r.circuit_id, r.circuit_type,
           rd.driver_id, rd.constructor_id, rd.engine_manufacturer_id,
           rd.position_number   AS finish_pos,
           rd.position_text,
           rd.race_grid_position_number AS grid_num,
           rd.race_grid_position_text   AS grid_text,
           rd.race_qualification_position_number AS quali_pos,
           rd.race_points       AS points,
           rd.race_reason_retired AS reason
    FROM race_data rd JOIN race r ON r.id = rd.race_id
    WHERE rd.type = 'RACE_RESULT' AND r.year BETWEEN ? AND ?
    """
    return pd.read_sql(q, con, params=(start, end))


def largest_connected_cohort(df: pd.DataFrame, min_shared: int) -> tuple[set[str], list]:
    """Teammate graph: edge between two drivers who shared a constructor-season with
    >= min_shared race entries each. Return drivers in the largest component."""
    counts = df.groupby(["year", "constructor_id", "driver_id"]).size().reset_index(name="n")
    counts = counts[counts.n >= min_shared]
    G = nx.Graph()
    for (year, ctor), grp in counts.groupby(["year", "constructor_id"]):
        drivers = list(grp.driver_id)
        G.add_nodes_from(drivers)
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                G.add_edge(drivers[i], drivers[j])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main = comps[0] if comps else set()
    return main, comps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2022)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--min-shared", type=int, default=3,
                    help="min shared race entries to count a teammate edge")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    raw = load_race_results(con, args.start, args.end)
    con.close()
    print(f"Loaded {len(raw)} RACE_RESULT rows for {args.start}-{args.end}.")

    # --- derive fields ---
    raw["classified"] = raw.finish_pos.notna()
    raw["dnf_cause"] = raw.apply(
        lambda r: "finished" if r.classified else classify_reason(r.reason), axis=1)
    raw["reliability_dnf"] = raw.dnf_cause.eq("mechanical")

    # grid: PL (pit lane) -> back of grid; keep NA for genuine no-grid (handled by DNS drop)
    raw["grid"] = raw.grid_num
    raw.loc[raw.grid_text.eq("PL"), "grid"] = PIT_LANE_GRID

    # --- drop entries that never started (no race outcome to attribute) ---
    started = raw[~raw.position_text.eq("DNS")].copy()
    n_dns = len(raw) - len(started)

    # --- restrict to largest teammate-connected component ---
    cohort, comps = largest_connected_cohort(started, args.min_shared)
    df = started[started.driver_id.isin(cohort)].copy()

    # final column order
    cols = ["year", "round", "race_id", "circuit_id", "circuit_type",
            "driver_id", "constructor_id", "engine_manufacturer_id",
            "grid", "quali_pos", "finish_pos", "points",
            "classified", "dnf_cause", "reliability_dnf"]
    df = df[cols].sort_values(["year", "round", "finish_pos"]).reset_index(drop=True)

    # --- persist ---
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    # --- summary ---
    print("\n" + "=" * 64)
    print("BUILD SUMMARY")
    print("=" * 64)
    print(f"rows (started, in cohort): {len(df)}")
    print(f"drivers: {df.driver_id.nunique()}  constructors: {df.constructor_id.nunique()}  "
          f"races: {df.race_id.nunique()}")
    print(f"classified rate: {df.classified.mean():.1%}")
    print(f"dropped DNS rows: {n_dns}")
    print(f"\nteammate components (sizes): {[len(c) for c in comps]}")
    dropped = sorted(set(started.driver_id) - cohort)
    print(f"drivers dropped (not in main component): {dropped}")
    print("\ndnf_cause breakdown (all started rows in cohort):")
    print(df.dnf_cause.value_counts().to_string())
    print("\nrace entries per driver (cohort):")
    print(df.driver_id.value_counts().to_string())
    print(f"\nWrote {OUT_PARQUET.relative_to(ROOT)} and {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
