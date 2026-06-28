"""Phase B (fun): over-/under-rated drivers — who is carried by their car, who is held back?

A driver's *reputation* comes from results: where they actually finish. But results blend the
driver and the car. This project's whole point is to pull those apart — so we can ask the fun
question the model uniquely answers: **is a driver flattered or robbed by their machinery?**

For each driver we compare two things:
  - ACTUAL average finish — their results-based reputation (car included), and
  - MIDFIELD-CAR finish — where the SCM says they'd finish on their own skill in an AVERAGE car
    (we intervene: do(car_pace = median), do(driver_skill = theirs), predict finish).

The difference is the **car effect** in finishing positions:
  car_effect = midfield_car_finish - actual_finish
    > 0  →  they finish BETTER than their skill alone warrants → the car flatters them → OVER-RATED
    < 0  →  they finish WORSE  than their skill alone warrants → a bad car holds them back → UNDER-RATED

Reusing the fitted race-outcome SCM from attribution_v2 (same DAG, same continuous skill/pace nodes),
so this is the *causal* car effect, not a raw correlation. Bonus: a best/worst car-pace ranking.

Usage: python v2/insights.py [--data data/f1_scm_v2_2018_2025_sess_rw.parquet]
                             [--n 3000] [--min-races 10] [--tag _2018_2025]
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# reuse the exact SCM construction + intervention helper from the attribution stage
from attribution_v2 import build_scm, exp_finish, NODES

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "f1db.sqlite"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260628


def nice_names() -> dict[str, str]:
    """driver_id -> 'First Last' from the f1db driver table (graceful if DB absent)."""
    try:
        con = sqlite3.connect(DB)
        m = dict(con.execute("SELECT id, name FROM driver").fetchall())
        con.close()
        return m
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "f1_scm_v2_2018_2025_sess_rw.parquet"))
    ap.add_argument("--n", type=int, default=3000, help="SCM samples per expected-finish estimate")
    ap.add_argument("--min-races", type=int, default=10, help="min classified races to rank a driver")
    ap.add_argument("--tag", default="_2018_2025")
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    names = nice_names()
    nm = lambda d: names.get(d, d)

    full = pd.read_parquet(args.data)
    df = full[full.classified].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        df[c] = df[c].astype(float)
    df["circuit_type"] = df["circuit_type"].astype("object")
    data = df[NODES].dropna()

    scm = build_scm(data)

    # "average car" = median car pace across team-years (matches attribution_v2's mid_pace)
    pace_by_ty = full.groupby("team_year").car_pace.first()
    mid_pace = float(np.median(sorted(pace_by_ty.values)))

    # per-driver summary over the classified era
    g = df.groupby("driver_id")
    summ = pd.DataFrame({
        "races": g.finish_pos.size(),
        "skill": g.driver_skill.mean(),         # career-mean skill (time-varying -> averaged)
        "car_pace": g.car_pace.mean(),          # how good their car was, on average
        "actual_finish": g.finish_pos.mean(),   # results-based reputation
    })
    summ = summ[summ.races >= args.min_races].copy()

    # midfield-car expected finish per driver (causal: hold car at median, set skill = theirs)
    summ["midfield_finish"] = [exp_finish(scm, s, mid_pace, args.n) for s in summ.skill]
    summ["car_effect"] = summ.midfield_finish - summ.actual_finish   # >0 over-rated, <0 under-rated
    summ = summ.sort_values("car_effect", ascending=False)

    L = ["=" * 72, "PHASE B (fun) — OVER-/UNDER-RATED DRIVERS (driver vs. car)", "=" * 72,
         f"data: {Path(args.data).name}   drivers with >= {args.min_races} classified races: {len(summ)}",
         f"'average car' = median car pace ({mid_pace:+.2f}%). car_effect = midfield-car finish - actual finish.",
         "  (+) results flatter them: the car is doing the work  -> OVER-RATED",
         "  (-) results hide them:    a bad car holds them back   -> UNDER-RATED",
         "  NB 'over-rated' = results overstate the driver's share, NOT that they lack skill —",
         "     see [3] for the car-removed pace ranking (e.g. Hamilton is flattered yet still elite).",
         ""]

    L.append("[1] MOST OVER-RATED BY RESULTS (the car is carrying them)")
    for d, r in summ.head(6).iterrows():
        L.append(f"    {nm(d):20s} finishes P{r.actual_finish:4.1f}, but only P{r.midfield_finish:4.1f} "
                 f"in a midfield car  -> car worth {r.car_effect:+.1f} positions")
    L.append("\n[2] MOST UNDER-RATED BY RESULTS (a bad car is hiding them)")
    for d, r in summ.tail(6).iloc[::-1].iterrows():
        L.append(f"    {nm(d):20s} finishes P{r.actual_finish:4.1f}, would be P{r.midfield_finish:4.1f} "
                 f"in a midfield car  -> car costs {abs(r.car_effect):.1f} positions")

    # pure-pace ranking vs results ranking — where the two disagree is where the car hides
    summ["skill_rank"] = summ.skill.rank().astype(int)               # 1 = fastest (most negative)
    summ["result_rank"] = summ.actual_finish.rank().astype(int)      # 1 = best results
    summ["rank_gap"] = summ.result_rank - summ.skill_rank            # >0 under-rated, <0 over-rated
    L.append("\n[3] PURE PACE vs RESULTS (rank by skill alone vs rank by where they finish)")
    L.append(f"    {'driver':20s} {'pace#':>6s} {'result#':>8s} {'gap':>5s}")
    for d, r in summ.sort_values("skill_rank").iterrows():
        flag = " under-rated" if r.rank_gap >= 3 else (" over-rated" if r.rank_gap <= -3 else "")
        L.append(f"    {nm(d):20s} {int(r.skill_rank):6d} {int(r.result_rank):8d} "
                 f"{int(r.rank_gap):+5d}{flag}")

    # bonus: best/worst cars
    L.append("\n[4] BEST & WORST CARS (mean car pace by team-year; lower = faster)")
    ty = pace_by_ty.sort_values()
    for t, v in ty.head(5).items():
        L.append(f"    BEST  {t:22s} {v:+.2f}%")
    for t, v in ty.tail(5).iloc[::-1].items():
        L.append(f"    WORST {t:22s} {v:+.2f}%")

    report = "\n".join(L)
    print(report)
    (OUT / f"insights_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: diverging bar of car_effect per driver ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ.sort_values("car_effect")
    colors = ["#c0504d" if v > 0 else "#4f81bd" for v in s.car_effect]  # red=over, blue=under
    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.32 * len(s))))
    ax.barh([nm(d) for d in s.index], s.car_effect, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("car effect (finishing positions)   ← bad car holds back | good car flatters →")
    ax.set_title("Driver vs. car: who is carried, who is held back?\n"
                 "blue = under-rated (better than results) | red = over-rated (worse than results)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"insights_over_under{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/insights_report{args.tag}.txt, figures/insights_over_under{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
