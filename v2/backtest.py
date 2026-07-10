"""Phase A: out-of-sample backtest — do the learned driver skills PREDICT the future?

The cleanest test for this model. Driver skills are fit on TRAIN seasons (2018-2023); we then
predict each TEAMMATE head-to-head in the HELD-OUT seasons (2024-2025) — who out-qualifies their
teammate, and by how much. Teammates share the car, so the car-pace term cancels and the qualifying
gap is PURELY a skill difference:

    actual_gap = pct_gap_A - pct_gap_B = (skill_A - skill_B) + noise      (car_pace cancels)
    predicted_gap = skill_A(train) - skill_B(train)                       (no future data used)

Reported: head-to-head accuracy (did we pick the right driver?), correlation of predicted vs actual
gap, mean absolute error, all vs a coin-flip baseline; plus a calibration check (do the model's
credible intervals cover reality at their stated rate?).

Usage: python v2/backtest.py [--train-idata models/v2_idata_2018_2023_sess_rw.pkl]
                             [--test-start 2024] [--test-end 2025]
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "f1db.sqlite"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
GAP_CAP = 10.0


def load_session_gaps(start: int, end: int) -> pd.DataFrame:
    """2024-2025 quali, session-relative pct_gap, ALL drivers (no cohort restriction)."""
    con = sqlite3.connect(DB)
    q = """SELECT r.year, r.round, rd.race_id, rd.driver_id, rd.constructor_id,
           rd.qualifying_q1_millis q1, rd.qualifying_q2_millis q2, rd.qualifying_q3_millis q3,
           rd.qualifying_time_millis qt
           FROM race_data rd JOIN race r ON r.id=rd.race_id
           WHERE rd.type='QUALIFYING_RESULT' AND r.year BETWEEN ? AND ?"""
    df = pd.read_sql(q, con, params=(start, end)); con.close()
    df = df[df[["q1", "q2", "q3", "qt"]].notna().any(axis=1)].copy()
    long = df.melt(id_vars=["race_id", "driver_id"], value_vars=["q1", "q2", "q3", "qt"],
                   var_name="session", value_name="millis").dropna(subset=["millis"])
    long["sess_pole"] = long.groupby(["race_id", "session"]).millis.transform("min")
    long["gap"] = long.millis / long.sess_pole - 1.0
    best = long.groupby(["race_id", "driver_id"]).gap.min().mul(100.0).rename("pct_gap")
    df = df.merge(best, on=["race_id", "driver_id"]).query("pct_gap <= @GAP_CAP")
    return df[["year", "round", "race_id", "driver_id", "constructor_id", "pct_gap"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-idata", default=str(ROOT / "models" / "v2_idata_2018_2023_sess_rw.pkl"))
    ap.add_argument("--test-start", type=int, default=2024)
    ap.add_argument("--test-end", type=int, default=2025)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    # --- train skills: each driver's LAST trained season (skill is ~stable) + posterior draws ---
    idata = pickle.load(open(args.train_idata, "rb"))
    post = idata.posterior
    skill = post["skill"].stack(s=("chain", "draw"))            # (driver, season, s)
    seasons = list(skill.coords["season"].values)
    last_skill_mean, last_skill_draws = {}, {}
    for d in skill.coords["driver"].values:
        last_skill_mean[str(d)] = float(skill.sel(driver=d, season=seasons[-1]).mean())
        last_skill_draws[str(d)] = skill.sel(driver=d, season=seasons[-1]).values  # (s,)
    sigma = float(post["sigma"].mean()) if "sigma" in post else 0.0
    nu = float(post["nu"].mean()) if "nu" in post else 5.0

    # --- held-out teammate head-to-heads ---
    test = load_session_gaps(args.test_start, args.test_end)
    known = set(last_skill_mean)
    rows = []
    for (rid, ctor), g in test.groupby(["race_id", "constructor_id"]):
        g = g[g.driver_id.isin(known)]
        if g.driver_id.nunique() < 2:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if a.driver_id > b.driver_id:           # canonical order so pairs merge
                    a, b = b, a
                rows.append(dict(race_id=rid, a=a.driver_id, b=b.driver_id,
                                 actual=a.pct_gap - b.pct_gap,
                                 pred=last_skill_mean[a.driver_id] - last_skill_mean[b.driver_id]))
    h2h = pd.DataFrame(rows)
    # drop ties / both-identical; keep meaningful gaps
    h2h = h2h[h2h.pred.abs() > 1e-6].copy()

    # --- metrics ---
    acc = float((np.sign(h2h.pred) == np.sign(h2h.actual)).mean())
    corr = float(h2h.pred.corr(h2h.actual))
    mae = float((h2h.pred - h2h.actual).abs().mean())
    # baseline: predict 0 gap (coin flip on sign) -> 50%; and MAE of "always predict 0"
    mae_naive = float(h2h.actual.abs().mean())
    # aggregate per ordered pair (season-long H2H) for a cleaner headline
    pair = h2h.assign(key=h2h.a + " vs " + h2h.b).groupby("key").agg(
        pred=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    pair_acc = float((np.sign(pair.pred) == np.sign(pair.actual)).mean())

    # --- calibration: predicted gap distribution (skill diff draws + StudentT noise) vs actual ---
    rng = np.random.default_rng(0)
    levels = [0.5, 0.8, 0.9]
    cover = {lv: 0 for lv in levels}
    n_cal = 0
    for _, r in h2h.iterrows():
        da, db = last_skill_draws.get(r.a), last_skill_draws.get(r.b)
        if da is None or db is None:
            continue
        m = len(da)
        # single-race teammate gap ~ (skillA-skillB) + (noiseA - noiseB); StudentT(nu) scaled by sigma
        nA = sigma * rng.standard_t(nu, size=m)
        nB = sigma * rng.standard_t(nu, size=m)
        pred_dist = (da - db) + (nA - nB)
        for lv in levels:
            lo, hi = np.quantile(pred_dist, [(1 - lv) / 2, 1 - (1 - lv) / 2])
            cover[lv] += int(lo <= r.actual <= hi)
        n_cal += 1
    cover = {lv: cover[lv] / n_cal for lv in levels}

    L = ["=" * 64, "PHASE A — OUT-OF-SAMPLE BACKTEST (teammate H2H)", "=" * 64,
         f"train: 2018-2023 skills -> predict held-out {args.test_start}-{args.test_end}",
         f"teammate H2H comparisons: {len(h2h)} (race-level), {len(pair)} season-long pairs",
         "",
         f"  HEAD-TO-HEAD ACCURACY (race-level): {acc:.1%}   (coin-flip baseline 50%)",
         f"  HEAD-TO-HEAD ACCURACY (season-long): {pair_acc:.1%}",
         f"  correlation (predicted vs actual gap): {corr:.2f}",
         f"  mean abs error: {mae:.2f}%   vs predict-zero baseline {mae_naive:.2f}%",
         "",
         "  CALIBRATION (does a X% interval contain reality X% of the time?):"]
    for lv in levels:
        L.append(f"    {int(lv*100)}% interval -> empirical coverage {cover[lv]:.0%}")
    L.append("\n  season-long teammate H2H (predicted vs actual mean gap, fastest-first):")
    for k, r in pair.sort_values("pred").head(12).iterrows():
        ok = "OK " if np.sign(r.pred) == np.sign(r.actual) else "MISS"
        L.append(f"    [{ok}] {k:34s} pred {r.pred:+.2f}%  actual {r.actual:+.2f}%  (n={int(r.n)})")

    report = "\n".join(L)
    print(report)
    (OUT / "backtest_report.txt").write_text(report + "\n")

    # --- figure: predicted vs actual (season-long pairs) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.scatter(pair.pred, pair.actual, s=28, alpha=0.8, color="#1f3b73")
    lim = max(pair.pred.abs().max(), pair.actual.abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="#c0504d", lw=1, label="perfect prediction")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("PREDICTED teammate gap (from 2018-2023 skills, %)")
    ax.set_ylabel("ACTUAL teammate gap (held-out 2024-2025, %)")
    ax.set_title(f"Out-of-sample backtest: do learned skills predict the future?\n"
                 f"H2H accuracy {pair_acc:.0%} (season-long), correlation {corr:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "backtest.png", dpi=130, bbox_inches="tight")
    print("\nWrote outputs/backtest_report.txt, figures/backtest.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
