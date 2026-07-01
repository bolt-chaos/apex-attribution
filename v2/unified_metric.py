"""v2: unified expected-result metric — finish pace + mechanical risk + incident risk.

The SCM gives E[finish | the driver was classified]. But a season result also depends on NOT
retiring, and there are two ways to retire: the CAR fails (mechanical, charged to the constructor)
or the DRIVER errs (crash/spin, charged to the driver — see `fit_incident.py`). Per Option A we never
impute DNF positions into the finish model; we combine the three pieces ONLY here, at reporting:

    E_all = (1 - p_mech - p_inc) * E[finish|classified] + (p_mech + p_inc) * DNF_POS

    E[finish|classified]  from the SCM (driver skill in their car)   -> v2/attribution_v2.py
    p_mech                mechanical-DNF rate of the driver's constructor
    p_inc                 driver's own incident rate                 -> models/incident_rates_*.json
    DNF_POS               a retirement scores ~ back of the field

This turns "how fast" into "how much do you actually score", and exposes the **incident tax** — the
positions a driver loses per race to their OWN errors, p_inc * (DNF_POS - E[finish|classified]). A
fast driver in a good car has more to lose per incident, so incident-proneness costs the front of the
grid more than the back. Because the incident spread is modest (fit_incident.py shrinks it to
~5.4-7.0%), this is a real but small tiebreaker between similarly-fast drivers — reported honestly.

Usage: python v2/unified_metric.py [--scm-data data/f1_scm_v2_2018_2025_joint.parquet]
                                   [--results data/f1_results_2018_2025.parquet]
                                   [--incident-rates models/incident_rates_2018_2025.json]
                                   [--tag _2018_2025]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from attribution_v2 import build_scm, exp_finish, NODES

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "f1db.sqlite"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260701
DNF_POS = 18.0   # a retirement scores ~ back of a ~20-car field


def nice_names() -> dict:
    try:
        con = sqlite3.connect(DB); m = dict(con.execute("SELECT id, name FROM driver").fetchall())
        con.close(); return m
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scm-data", default=str(ROOT / "data" / "f1_scm_v2_2018_2025_joint.parquet"))
    ap.add_argument("--results", default=str(ROOT / "data" / "f1_results_2018_2025.parquet"))
    ap.add_argument("--incident-rates", default=str(ROOT / "models" / "incident_rates_2018_2025.json"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--min-starts", type=int, default=30)
    ap.add_argument("--tag", default="_2018_2025")
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    nm = nice_names()

    full = pd.read_parquet(args.scm_data)
    df = full[full.classified].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        df[c] = df[c].astype(float)
    df["circuit_type"] = df["circuit_type"].astype("object")
    scm = build_scm(df[NODES].dropna())

    # per-driver skill + their most-raced constructor's car pace (from the SCM data)
    skill_by_d = full.groupby("driver_id").driver_skill.mean()
    pace_by_c = full.groupby("constructor_id").car_pace.mean()
    main_car = full.groupby("driver_id").constructor_id.agg(lambda s: s.value_counts().index[0])

    # mechanical DNF rate per constructor (from the full started-rows results)
    res = pd.read_parquet(args.results)
    p_mech_by_c = res.assign(m=res.reliability_dnf.astype(int)).groupby("constructor_id").m.mean()
    p_mech_overall = float(res.reliability_dnf.mean())

    # per-driver incident rate (shrunk) from fit_incident.py
    inc = json.loads(Path(args.incident_rates).read_text())
    p_inc_overall = inc.get("_overall", 0.06)

    starts = res.groupby("driver_id").size()
    rows = []
    for d in skill_by_d.index:
        if starts.get(d, 0) < args.min_starts:
            continue
        car = main_car[d]
        e_fin = exp_finish(scm, skill_by_d[d], pace_by_c[car], args.n)
        p_mech = float(p_mech_by_c.get(car, p_mech_overall))
        p_inc = float(inc.get(d, p_inc_overall))
        e_all = (1 - p_mech - p_inc) * e_fin + (p_mech + p_inc) * DNF_POS
        incident_tax = p_inc * (DNF_POS - e_fin)        # positions/race lost to own errors
        mech_tax = p_mech * (DNF_POS - e_fin)
        rows.append(dict(driver=d, car=car, e_fin=e_fin, p_mech=p_mech, p_inc=p_inc,
                         e_all=e_all, incident_tax=incident_tax, mech_tax=mech_tax))
    R = pd.DataFrame(rows).set_index("driver").sort_values("e_all")

    L = ["=" * 76, "v2 UNIFIED EXPECTED-RESULT METRIC (finish + mechanical + incident risk)", "=" * 76,
         f"E_all = (1 - p_mech - p_inc)*E[finish|classified] + (p_mech + p_inc)*{DNF_POS:.0f}"
         "   (combined at reporting only, Option A)",
         f"drivers with >= {args.min_starts} starts, in their most-raced car; DNF scores P{DNF_POS:.0f}",
         "",
         f"  {'driver':20s} {'car':13s} {'E[fin|cl]':>9s} {'p_mech':>7s} {'p_inc':>6s} "
         f"{'E_all':>6s} {'inc.tax':>8s}"]
    for d, r in R.iterrows():
        L.append(f"  {nm.get(d, d):20s} {r.car:13s} {r.e_fin:9.1f} {r.p_mech:6.1%} {r.p_inc:5.1%} "
                 f"{r.e_all:6.1f} {r.incident_tax:+7.2f}")

    tax = R.sort_values("incident_tax", ascending=False)
    L.append("\n[INCIDENT TAX] positions/race a driver loses to their OWN errors "
             "= p_inc * (DNF - E[finish|classified])")
    L.append("  KEY: this is STAKES-dominated. Because shrunk incident rates barely vary (~5.4-7.0%),")
    L.append("  the tax tracks how HIGH a driver would finish, not how often they crash — a front-")
    L.append("  runner loses ~15 places per DNF, a backmarker only a few. So a clean front-runner's")
    L.append("  cleanliness is worth the most, and a crashy backmarker's proneness costs the least.")
    L.append("  most positions lost per race to own errors (front of grid — most at stake):")
    for d, r in tax.head(6).iterrows():
        L.append(f"    {nm.get(d, d):20s} {r.incident_tax:+.2f} pos/race  "
                 f"(p_inc {r.p_inc:.1%}, would finish P{r.e_fin:.1f})")
    L.append("  least at stake (back of grid — low tax despite SIMILAR crash rates, not 'cleaner'):")
    for d, r in tax.tail(4).iloc[::-1].iterrows():
        L.append(f"    {nm.get(d, d):20s} {r.incident_tax:+.2f} pos/race  "
                 f"(p_inc {r.p_inc:.1%}, would finish P{r.e_fin:.1f})")
    # cleanliness dividend isolates the DRIVER's proneness from stakes:
    # positions/race saved vs a field-average-rate driver in the SAME seat.
    R["dividend"] = (p_inc_overall - R.p_inc) * (DNF_POS - R.e_fin)
    div = R.sort_values("dividend", ascending=False)
    L.append("\n[CLEANLINESS DIVIDEND] positions/race saved vs a field-average-rate driver in the "
             "SAME seat")
    L.append("  = (p_inc_field - p_inc_driver) * (DNF - E[finish|classified]) — isolates proneness x stakes")
    for d, r in div.head(4).iterrows():
        L.append(f"    {nm.get(d, d):20s} {r.dividend:+.3f}  (cleaner than avg; p_inc {r.p_inc:.1%})")
    for d, r in div.tail(3).iloc[::-1].iterrows():
        L.append(f"    {nm.get(d, d):20s} {r.dividend:+.3f}  (crashier than avg; p_inc {r.p_inc:.1%})")
    L.append(f"\n  Dividends are tiny (|{div.dividend.abs().max():.2f}| pos/race at most): after shrinkage,")
    L.append("  incident-proneness is a real but MODEST tiebreaker between similarly-fast drivers.")
    L.append("  Where it matters most: a clean front-runner (Verstappen) protects a podium each race.")

    report = "\n".join(L)
    print(report)
    (OUT / f"v2_unified_metric_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: E[finish|classified] vs E_all (risk drags everyone back; slope shows the tax) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sc = ax.scatter(R.e_fin, R.e_all, c=R.p_inc * 100, cmap="Reds", s=45, edgecolor="#333", lw=0.4)
    lo = min(R.e_fin.min(), R.e_all.min()) - 0.5; hi = max(R.e_fin.max(), R.e_all.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="no DNF risk")
    for d, r in R.iterrows():
        ax.annotate(nm.get(d, d).split()[-1], (r.e_fin, r.e_all), fontsize=6.5,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("E[finish | classified]  (pure pace; better →)")
    ax.set_ylabel("E_all  (incl. mechanical + incident DNF risk; better →)")
    ax.set_title("Unified expected result: pace vs pace-after-DNF-risk\n"
                 "gap below the line = DNF tax; colour = driver incident rate")
    fig.colorbar(sc, ax=ax, label="driver incident rate (%)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"v2_unified_metric{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/v2_unified_metric_report{args.tag}.txt, "
          f"figures/v2_unified_metric{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
