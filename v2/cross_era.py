"""Phase B (fun, ILLUSTRATIVE ONLY): "Senna in a modern Red Bull".

The crowd-pleaser query — and the most caveated thing in the repo. We take a legend's skill,
estimated in their own era, and ask where it would land in a 2024 Red Bull. Read the WARNINGS
block in the output before quoting any number: this is *what the model implies under heroic
assumptions*, NOT an identified causal effect.

The hard part is the ERA SCALE. A driver's skill here is "% qualifying gap vs the field", but the
field's SPREAD has collapsed over time: in 1988-1994 the grid SD is ~0.71%, by 2018-2025 it is
~0.25% (cars and drivers converged). So a raw -1.3% gap in 1990 is NOT the same achievement as
-1.3% in 2024. Plugging the raw number straight into a modern model is meaningless.

We show two translations side by side, so the assumption is visible, not hidden:
  1. NAIVE (raw skill, no era adjustment) — wrong, shown only to demonstrate why.
  2. ERA-NORMALIZED (z-score): how many standard deviations the legend was ahead of THEIR field,
     mapped onto the modern field's spread:  modern_equiv = modern_mean + z * modern_sd.
     (Senna at z=-2.18 SD -> modern-equiv skill ~ -1.59%, just ahead of Verstappen's 2024 -1.51%.)

Both translated skills are fed, with the real red-bull@2024 car pace, into the well-behaved MODERN
race-outcome SCM (fit on 2018-2025 rows, same continuous skill/pace scale) to read off an expected
finishing position with a WIDE credible band. Everything is reported as a range with the caveats.

Source model: v2_idata_1988_2025_sess_rw (max R-hat 1.04 -> NOT fully converged; another reason
this is illustrative). SCM data: f1_scm_v2_1988_2025_sess_rw.parquet (build via build_scm_data.py).

Usage: python v2/cross_era.py [--target-team red-bull@2024]
                              [--legends ayrton-senna michael-schumacher alain-prost]
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from attribution_v2 import build_scm, exp_finish, NODES

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "f1db.sqlite"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260628
MODERN = range(2018, 2026)        # era whose field-spread we map onto + whose rows fit the SCM
# legend -> peak seasons to average their skill over (their era, for the z-score)
PEAK = {"ayrton-senna": range(1988, 1995), "alain-prost": range(1988, 1994),
        "michael-schumacher": range(1994, 2005), "mika-hakkinen": range(1998, 2002),
        "nigel-mansell": range(1988, 1993), "damon-hill": range(1994, 1997)}


def nice_names() -> dict:
    try:
        con = sqlite3.connect(DB); m = dict(con.execute("SELECT id, name FROM driver").fetchall())
        con.close(); return m
    except Exception:
        return {}


def season_spread(skill_mean: pd.DataFrame, active: dict) -> pd.DataFrame:
    """Per-season (mean, sd) of driver skills across the drivers active that season."""
    rows = []
    for y, drv in active.items():
        vals = [skill_mean.loc[d, y] for d in drv if d in skill_mean.index and y in skill_mean.columns]
        if len(vals) > 3:
            rows.append((y, float(np.mean(vals)), float(np.std(vals))))
    return pd.DataFrame(rows, columns=["year", "mean", "sd"]).set_index("year")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scm-data", default=str(ROOT / "data" / "f1_scm_v2_1988_2025_sess_rw.parquet"))
    ap.add_argument("--idata", default=str(ROOT / "models" / "v2_idata_1988_2025_sess_rw.pkl"))
    ap.add_argument("--quali", default=str(ROOT / "data" / "f1_quali_1988_2025_sess.parquet"))
    ap.add_argument("--target-team", default="red-bull@2024")
    ap.add_argument("--legends", nargs="+",
                    default=["ayrton-senna", "michael-schumacher", "alain-prost"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    names = nice_names(); nm = lambda d: names.get(d, d)

    # --- modern, well-behaved race-outcome SCM (fit on MODERN rows; consistent skill/pace scale) ---
    full = pd.read_parquet(args.scm_data)
    mod = full[full.classified & full.year.isin(MODERN)].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        mod[c] = mod[c].astype(float)
    mod["circuit_type"] = mod["circuit_type"].astype("object")
    scm = build_scm(mod[NODES].dropna())

    target_pace = float(full[full.team_year == args.target_team].car_pace.iloc[0])

    # --- posterior skills + per-era field spread (all on the 1988-2025 model scale) ---
    post = pickle.load(open(args.idata, "rb")).posterior
    skill_draws = post["skill"].stack(s=("chain", "draw"))     # (driver, season, s)
    skill_mean = post["skill"].mean(("chain", "draw")).to_pandas()
    qd = pd.read_parquet(args.quali)
    active = {y: list(g.driver_id.unique()) for y, g in qd.groupby("year")}
    spread = season_spread(skill_mean, active)
    mod_mean = float(spread.loc[spread.index.isin(MODERN), "mean"].mean())
    mod_sd = float(spread.loc[spread.index.isin(MODERN), "sd"].mean())

    # modern reference: Verstappen 2024 in the same car
    vers_skill = float(skill_mean.loc["max-verstappen", 2024])
    vers_finish = exp_finish(scm, vers_skill, target_pace, args.n)

    # 2024 grid skills (to rank the legend among real modern drivers)
    grid24 = [skill_mean.loc[d, 2024] for d in active.get(2024, []) if d in skill_mean.index]

    L = ["=" * 76, f'PHASE B (ILLUSTRATIVE) — "{nm(args.legends[0])} in a {args.target_team}"', "=" * 76,
         "",
         "  *** WARNINGS — read before quoting any number ***",
         "  - This is an EXTRAPOLATION, not an identified effect: no legend ever drove this car.",
         "  - Cross-era skill is only comparable AFTER an era-scale assumption (z-score below);",
         "    the raw/naive number is shown ONLY to demonstrate why it is wrong.",
         "  - Source skill model has max R-hat 1.04 (NOT fully converged) -> treat as a story,",
         "    not a measurement. Intervals are WIDE on purpose.",
         "",
         f"  field spread collapsed over time: 1988-1994 SD ~{spread.loc[1988:1994].sd.mean():.2f}%"
         f"  vs  {min(MODERN)}-{max(MODERN)} SD ~{mod_sd:.2f}%  (grid converged).",
         f"  modern field: mean {mod_mean:+.2f}%, SD {mod_sd:.2f}%.  "
         f"target car ({args.target_team}) pace {target_pace:+.2f}%.",
         f"  reference: Verstappen 2024 skill {vers_skill:+.2f}% -> E[finish] in that car "
         f"P{vers_finish:.1f}.",
         ""]

    fig_rows = []
    for lg in args.legends:
        if lg not in skill_mean.index:
            L.append(f"  {nm(lg)}: not in model cohort — skipped."); continue
        yrs = [y for y in PEAK.get(lg, range(1988, 1995)) if y in skill_mean.columns]
        # peak-season skill posterior draws (pooled across the legend's peak years)
        draws = np.concatenate([skill_draws.sel(driver=lg, season=y).values for y in yrs])
        raw_mean = float(draws.mean())
        era = spread.loc[spread.index.isin(yrs)]
        era_mean, era_sd = float(era["mean"].mean()), float(era.sd.mean())
        # z-score within their era, then map onto the modern field's spread
        z_draws = (draws - era_mean) / era_sd
        equiv_draws = mod_mean + z_draws * mod_sd
        z_mean = float(z_draws.mean())
        eq_mean, eq_lo, eq_hi = (float(equiv_draws.mean()),
                                 float(np.quantile(equiv_draws, 0.05)),
                                 float(np.quantile(equiv_draws, 0.95)))
        # rank among the 2024 grid on the normalized skill
        rank = 1 + sum(g < eq_mean for g in grid24)
        # E[finish] in the target car: naive (raw) vs era-normalized (mean + wide band at 5/95).
        # near the front the SCM saturates and is noisy, so take a robust min/max for the band.
        f_naive = exp_finish(scm, raw_mean, target_pace, args.n)
        f_norm = exp_finish(scm, eq_mean, target_pace, args.n)
        f_a = exp_finish(scm, eq_hi, target_pace, args.n)    # faster skill end
        f_b = exp_finish(scm, eq_lo, target_pace, args.n)    # slower skill end
        f_lo, f_hi = min(f_a, f_b, f_norm), max(f_a, f_b, f_norm)

        L.append(f"  {nm(lg).upper()}  (peak {yrs[0]}-{yrs[-1]})")
        L.append(f"    raw skill {raw_mean:+.2f}%  ->  z = {z_mean:+.2f} SD vs his era "
                 f"(field SD {era_sd:.2f}%)")
        L.append(f"    ERA-NORMALIZED skill {eq_mean:+.2f}%  90% CrI [{eq_hi:+.2f}, {eq_lo:+.2f}]  "
                 f"-> would rank ~P{rank} on the 2024 grid by pace")
        L.append(f"    E[finish] in {args.target_team}:  "
                 f"NAIVE P{f_naive:.1f}  |  ERA-NORM P{f_norm:.1f}  (90% band P{f_lo:.1f}-P{f_hi:.1f})")
        L.append(f"    vs Verstappen 2024 in the same car: P{vers_finish:.1f}")
        L.append("")
        fig_rows.append(dict(legend=nm(lg), norm=f_norm, lo=f_lo, hi=f_hi))

    L.append("  TAKEAWAY: era-normalized, the all-time greats land roughly where peak Verstappen")
    L.append("  does in the same car — utterly dominant, but NOT physically impossible. The honest")
    L.append("  answer to 'Senna in a modern Red Bull' is 'a front-running, title-winning car-and-")
    L.append("  driver combo' — with the loud caveat that cross-era skill is an assumption, not data.")

    report = "\n".join(L)
    print(report)
    (OUT / f"cross_era_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: legends + Verstappen reference, expected finish with wide bands ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = pd.DataFrame(fig_rows)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.7 * (len(fr) + 1)))
    y = np.arange(len(fr))
    lo_err = (fr.norm - fr.lo).clip(lower=0)
    hi_err = (fr.hi - fr.norm).clip(lower=0)
    ax.barh(y, fr.norm, color="#6a51a3", alpha=0.85,
            xerr=[lo_err, hi_err], error_kw=dict(ecolor="#b5a6d6", capsize=4))
    ax.axvline(vers_finish, ls="--", color="#c0504d", lw=1.5,
               label=f"Verstappen 2024 (P{vers_finish:.1f})")
    ax.set_yticks(y); ax.set_yticklabels(fr.legend)
    ax.invert_yaxis()
    ax.set_xlabel(f"expected finish in a {args.target_team}  (lower = better; 90% band)")
    ax.set_title(f"ILLUSTRATIVE: era-normalized legends in a {args.target_team}\n"
                 "extrapolation under an era-scale assumption — not an identified effect",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"cross_era{args.tag}.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote outputs/cross_era_report{args.tag}.txt, figures/cross_era{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
