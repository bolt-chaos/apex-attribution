"""v2 phase 7a: walk the era back — teammate-graph connectivity vs start year.

The cross-era goal ("Senna in a modern Red Bull") needs the teammate chain to connect the
old era to today, on the data the skill model actually uses (qualifying entries with a usable
lap). This sweep, for decreasing start years (end fixed at 2025), reports:
  - drivers, team-years, # connected components, largest-component size, % of drivers in it
  - which tracked legends are PRESENT and IN the main component (esp. Senna)
so we can see how far back one connected component reaches and where the chain bridges eras.

Connectivity here is computed on the SAME quali-with-lap data the model would fit (qt fallback
included, so pre-2006 single-session quali counts). Pure graph analysis — no model fitting.

Usage: python v2/era_connectivity.py
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
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
END = 2025
START_YEARS = [2018, 2014, 2010, 2006, 2000, 1994, 1988, 1984, 1980]
# legends / hinge drivers to track across eras (f1db slugs; absent ones are skipped)
LEGENDS = ["ayrton-senna", "alain-prost", "nigel-mansell", "gerhard-berger", "jean-alesi",
           "michael-schumacher", "mika-hakkinen", "damon-hill", "rubens-barrichello",
           "fernando-alonso", "kimi-raikkonen", "lewis-hamilton", "max-verstappen"]


def load_quali(con) -> pd.DataFrame:
    q = """
    SELECT r.year, rd.driver_id, rd.constructor_id,
           rd.qualifying_q1_millis AS q1, rd.qualifying_q2_millis AS q2,
           rd.qualifying_q3_millis AS q3, rd.qualifying_time_millis AS qt
    FROM race_data rd JOIN race r ON r.id = rd.race_id
    WHERE rd.type = 'QUALIFYING_RESULT' AND r.year BETWEEN 1980 AND ?
    """
    df = pd.read_sql(q, con, params=(END,))
    df["best"] = df[["q1", "q2", "q3", "qt"]].min(axis=1)
    return df[df.best.notna()].copy()


def main() -> int:
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    allq = load_quali(con)
    con.close()

    rows = []
    L = ["=" * 78, "WALK THE ERA BACK — teammate-graph connectivity (quali-with-lap data)", "=" * 78,
         f"{'era':>11} {'drivers':>8} {'tm-yrs':>7} {'comps':>6} {'main':>5} {'in_main%':>9}  legends in main component"]
    senna_bridge = None
    for s in START_YEARS:
        d = allq[(allq.year >= s) & (allq.year <= END)]
        cohort, comps = largest_connected_cohort(d, min_shared=3)
        n_drivers = d.driver_id.nunique()
        n_ty = d.assign(ty=d.constructor_id + "@" + d.year.astype(str)).ty.nunique()
        pct_main = 100 * len(cohort) / n_drivers if n_drivers else 0
        present = [x for x in LEGENDS if x in set(d.driver_id)]
        in_main = [x.split("-")[-1] for x in present if x in cohort]
        if "ayrton-senna" in present and "ayrton-senna" in cohort and senna_bridge is None:
            senna_bridge = s
        rows.append(dict(start=s, drivers=n_drivers, team_years=n_ty,
                         components=len(comps), main=len(cohort), pct_main=pct_main))
        L.append(f"{s}-{END:>4} {n_drivers:>8} {n_ty:>7} {len(comps):>6} {len(cohort):>5} "
                 f"{pct_main:>8.0f}%  {', '.join(in_main)}")

    L.append("")
    if senna_bridge:
        L.append(f">> Senna IS in the main connected component once the era starts at {senna_bridge} "
                 f"(his career was 1984-1994).")
    else:
        L.append(">> Senna never joins the main connected component in this sweep.")
    # explicit Senna-era detail
    sd = allq[(allq.year >= 1984) & (allq.year <= END)]
    sc, scomps = largest_connected_cohort(sd, min_shared=3)
    L.append(f">> 1984-2025: {len(sc)}/{sd.driver_id.nunique()} drivers in main component "
             f"({len(scomps)} components total) — the full Senna->today span.")

    report = "\n".join(L)
    print(report)
    (OUT / "era_connectivity.txt").write_text(report + "\n")

    # --- figure: connectivity vs start year ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rdf = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(rdf.start, rdf.main, "o-", color="#1f3b73", label="drivers in main component")
    ax1.plot(rdf.start, rdf.drivers, "o--", color="#9bb0d6", label="total drivers")
    ax1.set_xlabel("era start year (end = 2025)"); ax1.set_ylabel("number of drivers")
    ax1.invert_xaxis()  # walking back = left-to-right
    ax2 = ax1.twinx()
    ax2.plot(rdf.start, rdf.pct_main, "s-", color="#c0504d", label="% in main component")
    ax2.set_ylabel("% of drivers in main component", color="#c0504d")
    ax2.set_ylim(0, 105)
    ax1.set_title("Walking the era back: teammate-graph connectivity\n"
                  "(one connected component = cross-era skill comparison is possible)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "era_connectivity.png", dpi=130, bbox_inches="tight")
    print("\nWrote outputs/era_connectivity.txt, figures/era_connectivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
