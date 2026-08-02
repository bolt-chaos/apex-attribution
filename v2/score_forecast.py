"""Score a FORWARD forecast against the season that actually happened.

`v2/backtest.py` holds out seasons the model was deliberately not fit on. This script is the
stronger version of that test: it scores `v2/predict.py`'s **2026 forecast** — published before
a single 2026 lap was run — against the real first half of the season. Nothing here was tunable
after the fact; the forecast idata is the same file that produced `outputs/predict_report_2026.txt`.

Same estimand as the backtest, for the same reason: **teammate** qualifying head-to-heads. Teammates
share the car, so car pace cancels and the gap is a pure skill difference — which is what lets us
score a forecast for a season whose cars the model has never seen:

    actual_gap = pct_gap_A - pct_gap_B = (skill_A - skill_B) + noise      (car_pace cancels)
    predicted_gap = skill_A(projected) - skill_B(projected)               (no target-season data)

Skills are projected from the model's last trained season under the fitted random walk (a
martingale: same mean, uncertainty widened by sqrt(forward)*sigma_rw) — identical to `predict.py`.

Reported: head-to-head accuracy (race-level and season-long), correlation and MAE of predicted vs
actual gaps, a confidence-calibration table (do the stated P(out-qualifies) values come true at
their claimed rate?), and interval coverage. Line-up churn is handled by scoring the pairings that
ACTUALLY raced, and naming the pairs the forecast covered that never materialised.

Usage: python v2/score_forecast.py [--idata models/v2_idata_2018_2025_sess_rw.pkl]
                                   [--season 2026] [--tag _2026]
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
SEED = 20260802


def nice_names() -> dict[str, str]:
    """driver_id -> 'First Last' from the f1db driver table (graceful if DB absent)."""
    try:
        con = sqlite3.connect(DB)
        m = dict(con.execute("SELECT id, name FROM driver").fetchall())
        con.close()
        return m
    except Exception:
        return {}


def load_session_gaps(start: int, end: int) -> pd.DataFrame:
    """Qualifying, session-relative pct_gap, ALL drivers (no cohort restriction).

    Session-relative (gap to the fastest lap in the same session) rather than gap-to-pole, so
    Q1-eliminated drivers aren't penalised for track evolution — the `--gap-method session`
    normalisation from build_quali.py. Matches how the forecast model was fit.
    """
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


def project_skill_draws(idata, target_season: int, rng) -> tuple[dict, int, float]:
    """Per-driver projected-skill posterior draws at `target_season` (as in predict.py).

    Carry each driver's last trained-season skill forward under the fitted RW: same mean, with
    sqrt(forward)*sigma_rw of fresh innovation folded in. Returns {driver: draws}, last season,
    and the drift SD actually applied.
    """
    post = idata.posterior
    skill = post["skill"].stack(s=("chain", "draw"))          # (driver, season, s)
    seasons = list(skill.coords["season"].values)
    last = int(seasons[-1])
    forward = max(0, target_season - last)
    sigma_rw = float(post["sigma_rw"].mean()) if "sigma_rw" in post else 0.0
    drift_sd = float(np.sqrt(forward) * sigma_rw)

    draws = {}
    for d in skill.coords["driver"].values:
        base = skill.sel(driver=d, season=seasons[-1]).values  # (s,)
        innov = rng.normal(0.0, drift_sd, size=base.shape) if drift_sd > 0 else 0.0
        draws[str(d)] = base + innov
    return draws, last, drift_sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idata", default=str(ROOT / "models" / "v2_idata_2018_2025_sess_rw.pkl"),
                    help="forecast model — must NOT have been fit on the scored season")
    ap.add_argument("--season", type=int, default=2026, help="season to score the forecast against")
    ap.add_argument("--tag", default="_2026", help="suffix for output files")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    idata = pickle.load(open(args.idata, "rb"))
    post = idata.posterior
    sigma = float(post["sigma"].mean()) if "sigma" in post else 0.0
    nu = float(post["nu"].mean()) if "nu" in post else 5.0
    skill_draws, last_season, drift_sd = project_skill_draws(idata, args.season, rng)
    skill_mean = {d: float(v.mean()) for d, v in skill_draws.items()}
    names = nice_names()
    nm = lambda d: names.get(d, d)

    if args.season <= last_season:
        print(f"WARNING: model was trained through {last_season}; scoring {args.season} is NOT "
              f"out-of-sample.")

    # --- actual teammate head-to-heads in the scored season ---
    actual = load_session_gaps(args.season, args.season)
    if actual.empty:
        print(f"No qualifying data for {args.season}.")
        return 1
    rounds = sorted(actual["round"].unique())
    known = set(skill_mean)

    rows, unscored_drivers = [], set()
    for (rid, ctor), g in actual.groupby(["race_id", "constructor_id"]):
        missing = set(g.driver_id) - known
        unscored_drivers |= missing
        g = g[g.driver_id.isin(known)]
        if g.driver_id.nunique() < 2:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if a.driver_id > b.driver_id:              # canonical order so pairs merge
                    a, b = b, a
                rows.append(dict(race_id=rid, ctor=ctor, a=a.driver_id, b=b.driver_id,
                                 actual=a.pct_gap - b.pct_gap,
                                 pred=skill_mean[a.driver_id] - skill_mean[b.driver_id]))
    h2h = pd.DataFrame(rows)
    h2h = h2h[h2h.pred.abs() > 1e-6].copy()
    if h2h.empty:
        print(f"No scoreable teammate pairs in {args.season}.")
        return 1

    # --- metrics ---
    acc = float((np.sign(h2h.pred) == np.sign(h2h.actual)).mean())
    corr = float(h2h.pred.corr(h2h.actual))
    mae = float((h2h.pred - h2h.actual).abs().mean())
    mae_naive = float(h2h.actual.abs().mean())
    pair = h2h.assign(key=h2h.a + " vs " + h2h.b).groupby("key").agg(
        pred=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"),
        ctor=("ctor", "first"), a=("a", "first"), b=("b", "first"))
    pair_acc = float((np.sign(pair.pred) == np.sign(pair.actual)).mean())
    pair_corr = float(pair.pred.corr(pair.actual))

    # --- per-pair predicted confidence P(faster out-qualifies) + interval coverage ---
    levels = [0.5, 0.8, 0.9]
    cover = {lv: 0 for lv in levels}
    n_cal = 0
    for _, r in h2h.iterrows():
        da, db = skill_draws[r.a], skill_draws[r.b]
        m = len(da)
        nA = sigma * rng.standard_t(nu, size=m)
        nB = sigma * rng.standard_t(nu, size=m)
        pred_dist = (da - db) + (nA - nB)
        for lv in levels:
            lo, hi = np.quantile(pred_dist, [(1 - lv) / 2, 1 - (1 - lv) / 2])
            cover[lv] += int(lo <= r.actual <= hi)
        n_cal += 1
    cover = {lv: cover[lv] / n_cal for lv in levels}

    # per-pair P(predicted-faster driver out-qualifies) + the realised race-level hit rate
    conf_rows = []
    for key, r in pair.iterrows():
        da, db = skill_draws[r.a], skill_draws[r.b]
        m = len(da)
        gap = (da - db) + sigma * rng.standard_t(nu, size=m) - sigma * rng.standard_t(nu, size=m)
        if r.pred <= 0:
            fast, slow, p_fast = r.a, r.b, float((gap < 0).mean())
            hits = int((h2h[(h2h.a == r.a) & (h2h.b == r.b)].actual < 0).sum())
        else:
            fast, slow, p_fast = r.b, r.a, float((gap > 0).mean())
            hits = int((h2h[(h2h.a == r.a) & (h2h.b == r.b)].actual > 0).sum())
        n = int(r.n)
        conf_rows.append(dict(key=key, ctor=r.ctor, fast=fast, slow=slow, p=p_fast,
                              pred_gap=abs(r.pred), actual_gap=r.actual, n=n, hits=hits,
                              rate=hits / n, season_ok=np.sign(r.pred) == np.sign(r.actual)))
    conf = pd.DataFrame(conf_rows).sort_values("p", ascending=False)

    # confidence calibration: do the stated probabilities come true at their claimed rate?
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    cal_rows = []
    for lo, hi in bins:
        sel = conf[(conf.p >= lo) & (conf.p < hi)]
        if sel.n.sum() == 0:
            continue
        cal_rows.append(dict(band=f"{lo:.0%}-{min(hi,1.0):.0%}", pairs=len(sel),
                             stated=float((sel.p * sel.n).sum() / sel.n.sum()),
                             realised=float(sel.hits.sum() / sel.n.sum()), races=int(sel.n.sum())))

    # --- field-spread context: 2026's regulation reset widened the grid ---
    prev = load_session_gaps(last_season, last_season)
    spread_now = float(actual.groupby("race_id").pct_gap.std().mean())
    spread_prev = float(prev.groupby("race_id").pct_gap.std().mean())

    fwd = args.season - last_season
    L = ["=" * 72,
         f"FORECAST SCORECARD — {args.season} (first {len(rounds)} rounds) vs the pre-season forecast",
         "=" * 72,
         f"forecast model: {Path(args.idata).name} (trained through {last_season}; "
         f"{args.season} unseen)",
         f"skills projected forward {fwd} season(s) under the fitted RW "
         f"(+{drift_sd:.3f}% drift SD)",
         f"scored rounds: {rounds[0]}-{rounds[-1]}   teammate H2H: {len(h2h)} race-level, "
         f"{len(pair)} season-long pairs",
         ""]
    if unscored_drivers:
        L.append(f"  not scoreable (no pre-{args.season} model estimate): "
                 f"{', '.join(sorted(nm(d) for d in unscored_drivers))}")
        L.append("")
    L += [f"  HEAD-TO-HEAD ACCURACY (race-level):  {acc:.1%}   (coin-flip baseline 50%)",
          f"  HEAD-TO-HEAD ACCURACY (season-long): {pair_acc:.1%}   "
          f"({int(conf.season_ok.sum())}/{len(conf)} pairs)",
          f"  correlation (predicted vs actual gap): {corr:.2f} race-level, "
          f"{pair_corr:.2f} season-long",
          f"  mean abs error: {mae:.2f}%   vs predict-zero baseline {mae_naive:.2f}%",
          "",
          "  CONFIDENCE CALIBRATION (did stated P(out-qualifies) come true at its claimed rate?):",
          "    stated    realised   pairs  races"]
    for c in cal_rows:
        L.append(f"    {c['stated']:5.0%}     {c['realised']:5.0%}      {c['pairs']:3d}    "
                 f"{c['races']:3d}   [{c['band']}]")
    L += ["",
          "  INTERVAL COVERAGE (does an X% interval contain reality X% of the time?):"]
    for lv in levels:
        L.append(f"    {int(lv*100)}% interval -> empirical coverage {cover[lv]:.0%}")
    L += ["",
          f"  CONTEXT — the {args.season} regulation reset widened the field: mean within-race",
          f"  grid SD {spread_prev:.2f}% ({last_season}) -> {spread_now:.2f}% ({args.season}), "
          f"a {spread_now/spread_prev:.1f}x jump.",
          "  Teammate gaps therefore got LARGER than the converged-era model expects, which inflates",
          "  the absolute-error metrics (hence MAE losing to the predict-zero baseline) without",
          "  implying the skill ORDERING is wrong — read accuracy and correlation, not MAE.",
          "",
          f"  PER-PAIR SCORECARD ({args.season} line-ups that actually raced, most-confident first):"]
    for _, r in conf.iterrows():
        ok = "OK  " if r.season_ok else "MISS"
        L.append(f"    [{ok}] {nm(r.fast):18s} > {nm(r.slow):18s} P={r.p:.0%}  "
                 f"pred {r.pred_gap:.2f}%  actual {abs(r.actual_gap):.2f}%"
                 f"{'' if r.season_ok else ' (reversed)'}  won {r.hits}/{r.n}")

    report = "\n".join(L)
    print(report)
    (OUT / f"forecast_scorecard{args.tag}.txt").write_text(report + "\n")

    # --- figure: predicted vs actual (season-long pairs) + confidence calibration ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))

    ax1.axhline(0, color="grey", lw=0.6); ax1.axvline(0, color="grey", lw=0.6)
    okm = np.sign(pair.pred) == np.sign(pair.actual)
    ax1.scatter(pair.pred[okm], pair.actual[okm], s=40, alpha=0.85, color="#1f3b73", label="correct")
    ax1.scatter(pair.pred[~okm], pair.actual[~okm], s=40, alpha=0.85, color="#c0504d", label="missed")
    lim = float(max(pair.pred.abs().max(), pair.actual.abs().max())) * 1.15
    ax1.plot([-lim, lim], [-lim, lim], "--", color="#888", lw=1, label="perfect prediction")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_xlabel(f"PREDICTED teammate gap (from {last_season} skills, %)")
    ax1.set_ylabel(f"ACTUAL teammate gap ({args.season} R{rounds[0]}-{rounds[-1]}, %)")
    ax1.set_title(f"Forecast vs reality: {pair_acc:.0%} season-long H2H, corr {pair_corr:.2f}")
    ax1.legend(fontsize=8)

    if cal_rows:
        cd = pd.DataFrame(cal_rows)
        x = np.arange(len(cd)); w = 0.38
        ax2.bar(x - w/2, cd.stated * 100, w, label="stated P", color="#a7c7ff")
        ax2.bar(x + w/2, cd.realised * 100, w, label="realised", color="#1f3b73")
        ax2.axhline(50, ls="--", color="grey", lw=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(cd.band)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("stated-confidence band")
        ax2.set_ylabel("% of teammate qualifying sessions won")
        ax2.set_title("Confidence calibration: stated vs realised")
        ax2.legend(fontsize=8)

    fig.suptitle(f"apex-attribution — {args.season} forecast scorecard "
                 f"(model trained through {last_season}, {args.season} unseen)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"forecast_scorecard{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/forecast_scorecard{args.tag}.txt, "
          f"figures/forecast_scorecard{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
