"""Phase B: forecast — project the learned driver skills FORWARD to an upcoming season.

The backtest (`v2/backtest.py`) showed the learned skills predict held-out teammate
qualifying head-to-heads (67% race / 80% season, correlation 0.40). This script turns that
validated capability into a forward-looking forecast: given a season's driver line-ups, who
out-qualifies their teammate, and by how much?

Why teammate H2H (and not "who finishes P1"): teammates share the car, so car pace cancels and
the predicted gap is a PURE skill difference — no future car-performance data needed. Absolute
finishing position would require knowing the (still-unbuilt) target-season cars, so we don't
claim it.

Skills are carried forward from each driver's LAST trained season via the fitted random walk:
the RW is a martingale, so the expected skill is unchanged, but the uncertainty WIDENS by
sqrt(forward_seasons) * sigma_rw — an honest "we know less about next year" penalty.

Two products:
  1. a projected driver-skill power ranking (pure qualifying pace, fastest first), and
  2. per-team teammate H2H predictions: expected gap + P(driver A out-qualifies B), with the
     same posterior-draw + StudentT race-noise machinery the backtest calibrated.

Line-ups default to each constructor's pairing in the model's last trained season; pass
`--lineup pairs.json` ({"constructor": ["driver-a", "driver-b"], ...}) for a hypothetical grid.

Usage: python v2/predict.py [--idata models/v2_idata_2018_2025_sess_rw.pkl]
                            [--quali data/f1_quali_2018_2025_sess.parquet]
                            [--season 2026] [--lineup pairs.json] [--tag _2026]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

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


def project_skill_draws(idata, target_season: int, rng) -> tuple[dict, int]:
    """Per-driver projected-skill posterior draws at `target_season`.

    Carry each driver's last trained-season skill forward under the fitted RW: same mean, with
    sqrt(forward) * sigma_rw extra spread folded in as fresh innovations. Returns
    {driver_id: draws (s,)} and the last trained season.
    """
    post = idata.posterior
    skill = post["skill"].stack(s=("chain", "draw"))         # (driver, season, s)
    seasons = list(skill.coords["season"].values)
    last = int(seasons[-1])
    forward = max(0, target_season - last)
    sigma_rw = float(post["sigma_rw"].mean()) if "sigma_rw" in post else 0.0
    drift_sd = np.sqrt(forward) * sigma_rw                    # 0 if forecasting the last season

    draws = {}
    for d in skill.coords["driver"].values:
        base = skill.sel(driver=d, season=seasons[-1]).values  # (s,)
        innov = rng.normal(0.0, drift_sd, size=base.shape) if drift_sd > 0 else 0.0
        draws[str(d)] = base + innov
    return draws, last


def default_lineup(quali: Path) -> dict[str, list[str]]:
    """Each constructor's top-2 drivers (by entries) in the data's last season."""
    df = pd.read_parquet(quali)
    last = df[df.year == df.year.max()]
    out = {}
    for ctor, g in last.groupby("constructor_id"):
        out[str(ctor)] = list(g.driver_id.value_counts().index[:2])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idata", default=str(ROOT / "models" / "v2_idata_2018_2025_sess_rw.pkl"))
    ap.add_argument("--quali", default=str(ROOT / "data" / "f1_quali_2018_2025_sess.parquet"))
    ap.add_argument("--season", type=int, default=2026, help="target season to forecast")
    ap.add_argument("--lineup", default=None, help="JSON {constructor: [driverA, driverB]} override")
    ap.add_argument("--tag", default="_2026", help="suffix for output files")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    idata = pickle.load(open(args.idata, "rb"))
    post = idata.posterior
    sigma = float(post["sigma"].mean()) if "sigma" in post else 0.0
    nu = float(post["nu"].mean()) if "nu" in post else 5.0
    skill_draws, last_season = project_skill_draws(idata, args.season, rng)
    names = nice_names()
    nm = lambda d: names.get(d, d)

    lineup = json.loads(Path(args.lineup).read_text()) if args.lineup else default_lineup(Path(args.quali))
    on_grid = {d for pair in lineup.values() for d in pair}

    # --- 1. projected driver-skill ranking (pure quali pace, fastest = most negative) ---
    rank = []
    for d, dr in skill_draws.items():
        rank.append(dict(driver=d, mean=float(dr.mean()),
                         lo=float(np.quantile(dr, 0.05)), hi=float(np.quantile(dr, 0.95))))
    rank = pd.DataFrame(rank).sort_values("mean").reset_index(drop=True)

    fwd = args.season - last_season
    L = ["=" * 70, f"PHASE B — {args.season} FORECAST (skills projected from {last_season})", "=" * 70,
         f"model: {Path(args.idata).name}   forward {fwd} season(s) under the fitted RW",
         f"projected skill = last-season skill, uncertainty widened by sqrt({fwd})*sigma_rw",
         "",
         "[1] PROJECTED DRIVER-SKILL RANKING (full model cohort, % quali pace vs grid avg; "
         "lower = faster)",
         f"    ['*' = on the {args.season} grid below; un-starred = seen in 2018-{last_season} but "
         f"not in the line-up)"]
    for i, r in rank.iterrows():
        star = "*" if r.driver in on_grid else " "
        L.append(f"   {star}{i+1:2d}. {nm(r.driver):22s} {r['mean']:+.2f}%   "
                 f"90% CrI [{r.lo:+.2f}, {r.hi:+.2f}]")

    # --- 2. per-team teammate head-to-heads ---
    L.append(f"\n[2] TEAMMATE HEAD-TO-HEAD PREDICTIONS ({args.season} line-ups)")
    L.append("    (expected qualifying gap + P(faster driver out-qualifies teammate) over a race;")
    L.append("     teammates share the car, so this is a pure skill comparison)")
    h2h_rows = []
    for ctor in sorted(lineup):
        pair = lineup[ctor]
        known = [d for d in pair if d in skill_draws]
        missing = [d for d in pair if d not in skill_draws]
        if len(known) < 2:
            who = ", ".join(nm(d) for d in missing) or "—"
            L.append(f"    {ctor:14s}: no estimate ({who} not in {last_season} model cohort)")
            continue
        a, b = known[0], known[1]
        da, db = skill_draws[a], skill_draws[b]
        m = len(da)
        # single-race gap = (skillA - skillB) + sigma*(StudentT noise diff); lower = faster
        nA = sigma * rng.standard_t(nu, size=m)
        nB = sigma * rng.standard_t(nu, size=m)
        gap = (da - db) + (nA - nB)                      # gap<0 => A faster than B
        exp_gap = float((da - db).mean())                # season-long expected (no race noise)
        # orient so `fast` is the predicted-faster driver
        if exp_gap <= 0:
            fast, slow, p_fast = a, b, float((gap < 0).mean())
        else:
            fast, slow, p_fast = b, a, float((gap > 0).mean())
        h2h_rows.append(dict(ctor=ctor, fast=fast, slow=slow, gap=abs(exp_gap), p=p_fast))
        L.append(f"    {ctor:14s}: {nm(fast):16s} > {nm(slow):16s}  by {abs(exp_gap):.2f}%  "
                 f"(P={p_fast:.0%})")

    report = "\n".join(L)
    print(report)
    (OUT / f"predict_report{args.tag}.txt").write_text(report + "\n")

    # --- figures: skill forest (left) + teammate-H2H confidence (right) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

    yr = np.arange(len(rank))[::-1]                       # fastest at top
    grid_mask = rank.driver.isin(on_grid).values
    for m_, col, ec in [(grid_mask, "#1f3b73", "#9bb0d6"), (~grid_mask, "#b9b9b9", "#d8d8d8")]:
        if m_.any():
            sub = rank[m_]; ys = yr[m_]
            ax1.errorbar(sub["mean"], ys, xerr=[sub["mean"] - sub.lo, sub.hi - sub["mean"]],
                         fmt="o", color=col, ecolor=ec, capsize=2, ms=4)
    ax1.set_yticks(yr)
    ax1.set_yticklabels([f"{nm(d)}{' *' if d in on_grid else ''}" for d in rank.driver], fontsize=8)
    ax1.axvline(0, ls="--", color="grey", lw=0.8)
    ax1.set_xlabel("projected skill (% quali pace vs grid avg; left = faster)")
    ax1.set_title(f"{args.season} projected driver skill (90% CrI; * = on grid)")

    if h2h_rows:
        hd = pd.DataFrame(h2h_rows).sort_values("p")
        ax2.barh(range(len(hd)), hd.p * 100, color="#a7c7ff")
        ax2.set_yticks(range(len(hd)))
        ax2.set_yticklabels([f"{nm(r.fast)} > {nm(r.slow)}" for _, r in hd.iterrows()], fontsize=8)
        ax2.axvline(50, ls="--", color="grey", lw=0.8, label="coin-flip")
        ax2.set_xlim(40, 100); ax2.set_xlabel("P(predicted-faster driver out-qualifies teammate), %")
        ax2.set_title(f"{args.season} teammate H2H confidence")
        ax2.legend(fontsize=8)
    fig.suptitle(f"apex-attribution — {args.season} forecast from {last_season} skills "
                 f"(forward {fwd} season under the fitted RW)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"predict{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/predict_report{args.tag}.txt, figures/predict{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
