"""Out-of-sample backtest for the RACE-PACE signal — does racecraft predict the future?

The race-pace analogue of `v2/backtest.py` (which validated the QUALI signal). We fit the joint
quali+race model on TRAIN seasons (2018-2023) and predict each TEAMMATE race-pace head-to-head in
the HELD-OUT seasons (2024-2025): who has the better race pace, and by how much? Teammates share the
car, so the car-pace term cancels and the race-pace gap is a pure racecraft difference:

    actual_gap = race_pct_gap_A - race_pct_gap_B = (racecraft_A - racecraft_B) + noise
    predicted_gap = racecraft_A(train) - racecraft_B(train)        (no future data used)

Same metrics as the quali backtest (head-to-head accuracy, correlation, MAE vs a predict-zero
baseline, StudentT-noise calibration). Race pace is noisier than qualifying, so we EXPECT lower
accuracy than the quali backtest's 67%/80% — the test is whether racecraft beats the 50% coin flip
out-of-sample at all.

Usage: python v2/backtest_race.py [--train-idata models/v2_idata_2018_2023_joint.pkl]
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
RACE_GAP_CAP = 8.0


def load_race_gaps(start: int, end: int) -> pd.DataFrame:
    """Held-out race pace as race_pct_gap (% deficit to winner), ALL drivers (no cohort restriction).

    Mirrors build_race_pace.py: leader = 0, lapped folded in via race_gap_laps*avg_lap, DNFs dropped,
    capped. Returns one row per (race, driver)."""
    con = sqlite3.connect(DB)
    q = """SELECT r.year, r.round, rd.race_id, rd.driver_id, rd.constructor_id,
           rd.position_number AS finish_pos, rd.race_laps,
           rd.race_time_millis, rd.race_gap_millis, rd.race_gap_laps
           FROM race_data rd JOIN race r ON r.id = rd.race_id
           WHERE rd.type = 'RACE_RESULT' AND r.year BETWEEN ? AND ?"""
    df = pd.read_sql(q, con, params=(start, end)); con.close()
    df = df[df.finish_pos.notna()].copy()
    winners = df[df.finish_pos == 1].set_index("race_id")
    df["winner_time"] = df.race_id.map(winners.race_time_millis)
    df["winner_laps"] = df.race_id.map(winners.race_laps)
    df = df[df.winner_time.notna() & (df.winner_time > 0)].copy()
    df["avg_lap"] = df.winner_time / df.winner_laps
    lapped = df.race_gap_millis.isna() & df.race_gap_laps.notna() & (df.finish_pos > 1)
    gap_ms = df.race_gap_millis.copy()
    gap_ms[df.finish_pos == 1] = 0.0
    gap_ms[lapped] = df.race_gap_laps[lapped] * df.avg_lap[lapped]
    df["gap_ms"] = gap_ms
    df = df[df.gap_ms.notna()].copy()
    df["race_pct_gap"] = df.gap_ms / df.winner_time * 100.0
    df = df[df.race_pct_gap <= RACE_GAP_CAP].copy()
    return df[["year", "round", "race_id", "driver_id", "constructor_id", "race_pct_gap"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-idata", default=str(ROOT / "models" / "v2_idata_2018_2023_joint.pkl"))
    ap.add_argument("--test-start", type=int, default=2024)
    ap.add_argument("--test-end", type=int, default=2025)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    # --- train racecraft: each driver's LAST trained season + posterior draws ---
    idata = pickle.load(open(args.train_idata, "rb"))
    post = idata.posterior
    rc = post["racecraft"].stack(s=("chain", "draw"))            # (driver, season, s)
    seasons = list(rc.coords["season"].values)
    last_mean, last_draws = {}, {}
    for d in rc.coords["driver"].values:
        last_mean[str(d)] = float(rc.sel(driver=d, season=seasons[-1]).mean())
        last_draws[str(d)] = rc.sel(driver=d, season=seasons[-1]).values     # (s,)
    sigma = float(post["sigma_r"].mean()) if "sigma_r" in post else 0.0
    nu = float(post["nu_r"].mean()) if "nu_r" in post else 5.0

    # --- held-out teammate race-pace head-to-heads ---
    test = load_race_gaps(args.test_start, args.test_end)
    known = set(last_mean)
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
                                 actual=a.race_pct_gap - b.race_pct_gap,
                                 pred=last_mean[a.driver_id] - last_mean[b.driver_id]))
    h2h = pd.DataFrame(rows)
    h2h = h2h[h2h.pred.abs() > 1e-6].copy()

    # --- metrics ---
    acc = float((np.sign(h2h.pred) == np.sign(h2h.actual)).mean())
    corr = float(h2h.pred.corr(h2h.actual))
    mae = float((h2h.pred - h2h.actual).abs().mean())
    mae_naive = float(h2h.actual.abs().mean())
    pair = h2h.assign(key=h2h.a + " vs " + h2h.b).groupby("key").agg(
        pred=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    pair_acc = float((np.sign(pair.pred) == np.sign(pair.actual)).mean())

    # --- calibration: racecraft-diff draws + StudentT race noise vs actual ---
    rng = np.random.default_rng(0)
    levels = [0.5, 0.8, 0.9]
    cover = {lv: 0 for lv in levels}
    n_cal = 0
    for _, r in h2h.iterrows():
        da, db = last_draws.get(r.a), last_draws.get(r.b)
        if da is None or db is None:
            continue
        m = len(da)
        nA = sigma * rng.standard_t(nu, size=m)
        nB = sigma * rng.standard_t(nu, size=m)
        pred_dist = (da - db) + (nA - nB)
        for lv in levels:
            lo, hi = np.quantile(pred_dist, [(1 - lv) / 2, 1 - (1 - lv) / 2])
            cover[lv] += int(lo <= r.actual <= hi)
        n_cal += 1
    cover = {lv: cover[lv] / n_cal for lv in levels}

    L = ["=" * 64, "RACE-PACE OUT-OF-SAMPLE BACKTEST (teammate racecraft H2H)", "=" * 64,
         f"train: 2018-2023 racecraft -> predict held-out {args.test_start}-{args.test_end}",
         f"teammate H2H comparisons: {len(h2h)} (race-level), {len(pair)} season-long pairs",
         "",
         f"  HEAD-TO-HEAD ACCURACY (race-level): {acc:.1%}   (coin-flip baseline 50%)",
         f"  HEAD-TO-HEAD ACCURACY (season-long): {pair_acc:.1%}",
         f"  correlation (predicted vs actual gap): {corr:.2f}",
         f"  mean abs error: {mae:.2f}%   vs predict-zero baseline {mae_naive:.2f}%",
         "",
         "  (race pace is noisier than qualifying; expect below the quali backtest's 67%/80%)",
         "",
         "  CALIBRATION (does a X% interval contain reality X% of the time?):"]
    for lv in levels:
        L.append(f"    {int(lv*100)}% interval -> empirical coverage {cover[lv]:.0%}")
    L.append("\n  season-long teammate racecraft H2H (predicted vs actual mean gap, fastest-first):")
    for k, r in pair.sort_values("pred").head(12).iterrows():
        ok = "OK " if np.sign(r.pred) == np.sign(r.actual) else "MISS"
        L.append(f"    [{ok}] {k:34s} pred {r.pred:+.2f}%  actual {r.actual:+.2f}%  (n={int(r.n)})")

    report = "\n".join(L)
    print(report)
    (OUT / "backtest_race_report.txt").write_text(report + "\n")

    # --- figure: predicted vs actual (season-long pairs) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.scatter(pair.pred, pair.actual, s=28, alpha=0.8, color="#6a3d9a")
    lim = max(pair.pred.abs().max(), pair.actual.abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="#c0504d", lw=1, label="perfect prediction")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("PREDICTED teammate race-pace gap (from 2018-2023 racecraft, %)")
    ax.set_ylabel("ACTUAL teammate race-pace gap (held-out 2024-2025, %)")
    ax.set_title(f"Out-of-sample: does RACECRAFT predict the future?\n"
                 f"H2H accuracy {pair_acc:.0%} (season-long), correlation {corr:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "backtest_race.png", dpi=130, bbox_inches="tight")
    print("\nWrote outputs/backtest_race_report.txt, figures/backtest_race.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
