"""v2 phase 5: time-varying driver skill (per-season Gaussian random walk).

The constant-skill assumption is the weakest link over the 8-year 2018-2025 span (rookies
improve, veterans decline). Here skill is allowed to drift SLOWLY across seasons via a
per-driver Gaussian random walk:

    pct_gap[i,r] ~ StudentT(nu, mu, sigma)
    mu          = skill[driver, season] + pace[constructor-season]
    skill[d, s] = skill_base[d] + drift[d, s]          # drift[d, 2018] = 0 (anchor)
    drift[d, s] = drift[d, s-1] + sigma_rw * z[d, s]    # random walk, small sigma_rw = slow
    skill_base  ~ ZeroSumNormal(sigma_skill)           # career level, anchors the scale
    pace        ~ Normal(mu_pace, sigma_pace)          # car pace per team-year

sigma_rw is learned (HalfNormal prior): the data decides how much skill actually drifts.
Within each season teammates identify relative skill; the random walk links a driver's
seasons. Skill lives on the full driver x season grid; only seasons a driver raced enter
the likelihood (pre-debut cells are unconstrained prior draws — see the anchoring caveat).

Usage: python v2/fit_skill_rw.py [--data data/f1_quali_2018_2025.parquet] [--tag _2018_2025_rw]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260623
NICE = {"max-verstappen": "Verstappen", "lewis-hamilton": "Hamilton", "fernando-alonso": "Alonso",
        "charles-leclerc": "Leclerc", "lando-norris": "Norris", "oscar-piastri": "Piastri",
        "sebastian-vettel": "Vettel", "kimi-raikkonen": "Raikkonen", "george-russell": "Russell"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "f1_quali_2018_2025.parquet"))
    ap.add_argument("--tag", default="_2018_2025_rw")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    IDATA = ROOT / "models" / f"v2_idata{args.tag}.pkl"

    df = pd.read_parquet(args.data)
    drv = df.driver_id.astype("category")
    seasons = sorted(df.year.unique())
    s_index = {y: i for i, y in enumerate(seasons)}
    ty = df.team_year.astype("category")
    di = drv.cat.codes.values
    si = df.year.map(s_index).values
    ti = ty.cat.codes.values
    D, S = len(drv.cat.categories), len(seasons)
    coords = {"driver": list(drv.cat.categories), "season": seasons,
              "team_year": list(ty.cat.categories)}
    y = df.pct_gap.values.astype(float)

    with pm.Model(coords=coords) as model:
        sigma_skill = pm.HalfNormal("sigma_skill", 1.0)
        sigma_pace = pm.HalfNormal("sigma_pace", 2.0)
        sigma_rw = pm.HalfNormal("sigma_rw", 0.15)        # season-to-season drift (small=slow)
        mu_pace = pm.Normal("mu_pace", 1.5, 1.0)

        skill_base = pm.ZeroSumNormal("skill_base", sigma=sigma_skill, dims="driver")
        z = pm.Normal("z", 0.0, 1.0, dims=("driver", "season"))   # non-centered RW innovations
        steps = sigma_rw * z[:, 1:]                                # (D, S-1)
        drift = pt.concatenate([pt.zeros((D, 1)), pt.cumsum(steps, axis=1)], axis=1)  # drift[:,0]=0
        skill = pm.Deterministic("skill", skill_base[:, None] + drift, dims=("driver", "season"))
        pace = pm.Normal("pace", mu_pace, sigma_pace, dims="team_year")

        nu = pm.Gamma("nu", 2.0, 0.1)
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = skill[di, si] + pace[ti]
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=4,
                          target_accept=0.95, random_seed=SEED, progressbar=False)

    with open(IDATA, "wb") as f:
        pickle.dump(idata, f)

    post = idata.posterior
    skill_da = post["skill"].mean(("chain", "draw"))                 # (driver, season)
    skill_df = skill_da.to_pandas()                                  # rows=driver, cols=season
    pace_m = post["pace"].mean(("chain", "draw")).to_series()

    # active (driver, season) presence -> career mean over seasons actually raced
    active = {d: sorted(g.year.unique()) for d, g in df.groupby("driver_id")}
    career = pd.Series({d: skill_df.loc[d, active[d]].mean() for d in skill_df.index})
    drift_span = pd.Series({d: skill_df.loc[d, active[d]].max() - skill_df.loc[d, active[d]].min()
                            for d in skill_df.index})

    rh = float(az.summary(idata, var_names=["skill_base", "pace", "sigma_rw"])["r_hat"].max())
    L = ["=" * 66, "v2 TIME-VARYING SKILL (per-season random walk) — RESULTS", "=" * 66,
         f"max R-hat = {rh:.3f}   sigma_rw = {float(post['sigma_rw'].mean()):.3f}% / season "
         f"(90% HDI [{float(post['sigma_rw'].quantile(0.05)):.3f}, "
         f"{float(post['sigma_rw'].quantile(0.95)):.3f}])"]
    L.append("\n[CAREER-MEAN DRIVER SKILL] % vs grid avg (negative = faster), best first")
    for d, v in career.sort_values().items():
        L.append(f"    {d:20s} {v:+.2f}%   (drift across career: {drift_span[d]:.2f}%)")
    L.append("\n[BIGGEST SKILL DRIFT across career] (improvement or decline the RW found)")
    for d, v in drift_span.sort_values(ascending=False).head(8).items():
        act = active[d]
        traj = " ".join(f"{yr}:{skill_df.loc[d, yr]:+.2f}" for yr in act)
        L.append(f"    {d:18s} span {v:.2f}%   {traj}")

    report = "\n".join(L)
    print(report)
    (OUT / f"v2_skill_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: skill trajectories for notable drivers ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    show = [d for d in ["max-verstappen", "lewis-hamilton", "fernando-alonso", "charles-leclerc",
                        "lando-norris", "oscar-piastri", "sebastian-vettel"] if d in active]
    for d in show:
        yrs = active[d]
        ax.plot(yrs, [skill_df.loc[d, yr] for yr in yrs], marker="o", label=NICE.get(d, d))
    ax.axhline(0, ls="--", color="grey", lw=0.8)
    ax.invert_yaxis()  # faster (more negative) at top
    ax.set_xlabel("season"); ax.set_ylabel("skill (% quali pace vs grid avg; up = faster)")
    ax.set_title(f"v2 time-varying driver skill (per-season random walk)\n"
                 f"sigma_rw = {float(post['sigma_rw'].mean()):.2f}%/season")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_skill_trajectories{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/v2_skill_report{args.tag}.txt, "
          f"figures/v2_skill_trajectories{args.tag}.png, {IDATA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
