"""v2: JOINT quali + race skill model — two correlated driver abilities.

`fit_skill_rw.py` identifies one driver skill from qualifying pace. This generalises it: each
driver has TWO latent abilities, jointly modelled and correlated:

    quali_skill  — one-lap pace (Saturday), from build_quali.py's `pct_gap`
    racecraft    — race pace    (Sunday),   from build_race_pace.py's `race_pct_gap`

Both are identified the same way the single skill already is — teammates share the car, so within a
constructor-season the qualifying gap pins quali_skill differences and the race gap pins racecraft
differences; the car term cancels for both. The two latents are linked only by a SOFT prior
(an LKJ correlation), not by an identification crutch — each likelihood stands on its own data.

    [quali_skill_base, racecraft_base][d] ~ MVN(0, Σ)        # Σ via LKJCholeskyCov (corr = rho)
    each latent then drifts per-season via its own Gaussian random walk (race drifts more)
    pct_gap[i]      ~ StudentT(nu_q, quali_skill[d,s] + pace_q[ty], sigma_q)
    race_pct_gap[j] ~ StudentT(nu_r, racecraft[d,s]   + pace_r[ty], sigma_r)   # own, larger sigma

The payoff is `rho_quali_race` (how aligned are Saturday and Sunday pace?) and the per-driver
**qualifying-merchant delta** `racecraft - quali_skill`: who races better than they qualify
(Alonso/Perez/Hamilton expected) vs the quali specialists who fade.

Race pace is noisier (strategy, safety cars, traffic, tyres); it gets its own larger StudentT
sigma, so it is correctly down-weighted relative to qualifying — a complement, not a replacement.

Usage: python v2/fit_skill_joint.py [--quali data/f1_quali_2018_2025_sess.parquet]
                                    [--race data/f1_race_pace_2018_2025.parquet]
                                    [--tag _2018_2025_joint]
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "f1db.sqlite"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260630


def nice_names() -> dict:
    try:
        con = sqlite3.connect(DB); m = dict(con.execute("SELECT id, name FROM driver").fetchall())
        con.close(); return m
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quali", default=str(ROOT / "data" / "f1_quali_2018_2025_sess.parquet"))
    ap.add_argument("--race", default=str(ROOT / "data" / "f1_race_pace_2018_2025.parquet"))
    ap.add_argument("--tag", default="_2018_2025_joint")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    IDATA = ROOT / "models" / f"v2_idata{args.tag}.pkl"
    nm = nice_names()

    q = pd.read_parquet(args.quali)
    r = pd.read_parquet(args.race)

    # shared coordinate sets (union over both frames); each likelihood indexes into them
    drivers = sorted(set(q.driver_id) | set(r.driver_id))
    seasons = sorted(set(q.year) | set(r.year))
    team_years = sorted(set(q.team_year) | set(r.team_year))
    d_ix = {d: i for i, d in enumerate(drivers)}
    s_ix = {y: i for i, y in enumerate(seasons)}
    t_ix = {t: i for i, t in enumerate(team_years)}
    D, S, T = len(drivers), len(seasons), len(team_years)

    dq = q.driver_id.map(d_ix).values; sq = q.year.map(s_ix).values; tq = q.team_year.map(t_ix).values
    dr = r.driver_id.map(d_ix).values; sr = r.year.map(s_ix).values; tr = r.team_year.map(t_ix).values
    yq = q.pct_gap.values.astype(float)
    yr = r.race_pct_gap.values.astype(float)

    coords = {"driver": drivers, "season": seasons, "team_year": team_years,
              "latent": ["quali", "race"]}
    with pm.Model(coords=coords) as model:
        # --- car: a quali pace AND a race pace per team-year (a car can be better on Sat than Sun) ---
        mu_pace_q = pm.Normal("mu_pace_q", 1.5, 1.0)
        mu_pace_r = pm.Normal("mu_pace_r", 1.0, 1.0)
        sigma_pace_q = pm.HalfNormal("sigma_pace_q", 2.0)
        sigma_pace_r = pm.HalfNormal("sigma_pace_r", 2.0)
        pace_q = pm.Normal("pace_q", mu_pace_q, sigma_pace_q, dims="team_year")
        pace_r = pm.Normal("pace_r", mu_pace_r, sigma_pace_r, dims="team_year")

        # --- per-driver career-level 2-vector, correlated across the two abilities ---
        sd_dist = pm.HalfNormal.dist(1.0, shape=2)
        chol, corr, _ = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)
        z_base = pm.Normal("z_base", 0.0, 1.0, dims=("driver", "latent"))   # non-centered
        base_raw = pt.dot(z_base, chol.T)                                   # (D, 2), correlated
        base = base_raw - base_raw.mean(axis=0, keepdims=True)              # sum-to-zero per latent
        rho = pm.Deterministic("rho_quali_race", corr[0, 1])

        # --- per-season random walk on each latent (race drifts more) ---
        sigma_rw_q = pm.HalfNormal("sigma_rw_q", 0.15)
        sigma_rw_r = pm.HalfNormal("sigma_rw_r", 0.25)
        zq_rw = pm.Normal("zq_rw", 0.0, 1.0, dims=("driver", "season"))
        zr_rw = pm.Normal("zr_rw", 0.0, 1.0, dims=("driver", "season"))
        drift_q = pt.concatenate([pt.zeros((D, 1)), pt.cumsum(sigma_rw_q * zq_rw[:, 1:], axis=1)], axis=1)
        drift_r = pt.concatenate([pt.zeros((D, 1)), pt.cumsum(sigma_rw_r * zr_rw[:, 1:], axis=1)], axis=1)
        quali_skill = pm.Deterministic("quali_skill", base[:, 0][:, None] + drift_q,
                                       dims=("driver", "season"))
        racecraft = pm.Deterministic("racecraft", base[:, 1][:, None] + drift_r,
                                     dims=("driver", "season"))

        # --- two StudentT likelihoods sharing the latents; race gets its own (larger) sigma ---
        nu_q = pm.Gamma("nu_q", 2.0, 0.1); sigma_q = pm.HalfNormal("sigma_q", 1.0)
        nu_r = pm.Gamma("nu_r", 2.0, 0.1); sigma_r = pm.HalfNormal("sigma_r", 1.5)
        pm.StudentT("y_quali", nu=nu_q, mu=quali_skill[dq, sq] + pace_q[tq], sigma=sigma_q, observed=yq)
        pm.StudentT("y_race", nu=nu_r, mu=racecraft[dr, sr] + pace_r[tr], sigma=sigma_r, observed=yr)

        idata = pm.sample(draws=args.draws, tune=args.tune, chains=4,
                          target_accept=0.95, random_seed=SEED, progressbar=False)

    with open(IDATA, "wb") as f:
        pickle.dump(idata, f)

    post = idata.posterior
    qs = post["quali_skill"].mean(("chain", "draw")).to_pandas()   # driver x season
    rc = post["racecraft"].mean(("chain", "draw")).to_pandas()
    active_q = {d: sorted(g.year.unique()) for d, g in q.groupby("driver_id")}
    active_r = {d: sorted(g.year.unique()) for d, g in r.groupby("driver_id")}
    career_q = pd.Series({d: qs.loc[d, active_q[d]].mean() for d in active_q})
    career_r = pd.Series({d: rc.loc[d, active_r[d]].mean() for d in active_r})
    both = sorted(set(active_q) & set(active_r))
    merchant = (career_r[both] - career_q[both]).sort_values()   # neg = better on Sunday

    rho_mean = float(post["rho_quali_race"].mean())
    rho_lo, rho_hi = (float(post["rho_quali_race"].quantile(0.05)),
                      float(post["rho_quali_race"].quantile(0.95)))
    rh = float(az.summary(idata, var_names=["pace_q", "pace_r", "sigma_rw_q", "sigma_rw_r",
                                            "rho_quali_race"])["r_hat"].max())
    sg_q, sg_r = float(post["sigma_q"].mean()), float(post["sigma_r"].mean())

    name = lambda d: nm.get(d, d)
    L = ["=" * 70, "v2 JOINT QUALI+RACE SKILL (two correlated latents) — RESULTS", "=" * 70,
         f"max R-hat = {rh:.3f}   sigma_q (quali noise) = {sg_q:.2f}%   "
         f"sigma_r (race noise) = {sg_r:.2f}%  ({'race noisier OK' if sg_r > sg_q else 'CHECK: race not noisier'})",
         f"rho(quali_skill, racecraft) = {rho_mean:+.2f}  (90% CrI [{rho_lo:+.2f}, {rho_hi:+.2f}])",
         "\n[CAREER QUALI_SKILL] % quali pace vs grid avg (neg = faster), best first"]
    for d, v in career_q[both].sort_values().items():
        L.append(f"    {name(d):20s} {v:+.2f}%")
    L.append("\n[CAREER RACECRAFT] % race pace vs grid avg (neg = faster), best first")
    for d, v in career_r[both].sort_values().items():
        L.append(f"    {name(d):20s} {v:+.2f}%")
    L.append("\n[QUALIFYING-MERCHANT DELTA] racecraft - quali_skill")
    L.append("  (most NEGATIVE = races much better than they qualify; POSITIVE = quali specialist)")
    L.append("  CAVEAT: deltas are small vs the signal's noise and are lapped-car SENSITIVE "
             "(run --drop-lapped).")
    L.append("  Trust the mid-grid signal (e.g. Perez races > qualifies; Tsunoda/Bottas quali "
             "specialists); the")
    L.append("  backmarker tail is partly race-gap COMPRESSION (lapped cars can't lose more), not "
             "pure racecraft.")
    for d, v in merchant.items():
        tag = " <- races > qualifies" if v < -0.05 else (" <- quali specialist" if v > 0.05 else "")
        L.append(f"    {name(d):20s} {v:+.2f}%{tag}")

    report = "\n".join(L)
    print(report)
    (OUT / f"v2_skill_joint_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: career quali_skill vs racecraft scatter ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7.5))
    x = career_q[both].values; y = career_r[both].values
    ax.scatter(x, y, s=26, color="#1f3b73", alpha=0.8)
    lo = min(x.min(), y.min()) - 0.1; hi = max(x.max(), y.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "--", color="#c0504d", lw=1, label="races as well as qualifies")
    for d in both:
        ax.annotate(name(d).split()[-1], (career_q[d], career_r[d]), fontsize=6.5,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.invert_xaxis(); ax.invert_yaxis()  # faster up/right
    ax.set_xlabel("career QUALI_SKILL (% pace vs grid avg; faster →)")
    ax.set_ylabel("career RACECRAFT (% pace vs grid avg; faster →)")
    ax.set_title(f"Saturday vs Sunday: one-lap pace vs racecraft\n"
                 f"rho = {rho_mean:+.2f}  (above the line = races better than qualifies)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_skill_joint_quali_vs_race{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/v2_skill_joint_report{args.tag}.txt, "
          f"figures/v2_skill_joint_quali_vs_race{args.tag}.png, {IDATA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
