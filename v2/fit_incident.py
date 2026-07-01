"""v2: driver incident-proneness — a per-driver driver-error-DNF risk term.

The skill/pace models measure how FAST a driver is. But throwing away a result by crashing is
also part of a driver's contribution: not binning the car is a skill. The DNF taxonomy already
splits retirements into `mechanical` (charged to the car's reliability, never the driver) and
`driver_error` (collision / accident / spun off — a real driver outcome). Here we model the
per-driver driver-error-DNF risk, the missing "incident-proneness" term.

Raw per-driver rates are noisy (driver-error DNFs are rare, ~6% of starts) and confounded by
circuit hazard (street circuits crash more). So we fit a hierarchical Bayesian logistic model
with PARTIAL POOLING — rare/short-career drivers shrink toward the field, and circuit hazard is
controlled:

    driver_error[i] ~ Bernoulli(p_i)
    logit(p_i)      = alpha + incident[driver_i] + hazard[circuit_type_i]
    incident[d]     ~ Normal(0, sigma_driver)          # per-driver proneness, shrunk
    hazard[c]       ~ Normal(0, sigma_circuit)         # street > road > race
    alpha           ~ Normal(logit(base_rate), 1.5)

Reports each driver's shrunk per-race incident probability (at the average circuit mix), and
saves `models/incident_rates_{tag}.json` for the unified metric (expected result including risk).

CAVEAT: driver-error DNFs are mostly the driver by construction, but a twitchy/hard-to-drive car
can induce errors; this model attributes incident-proneness to the DRIVER (controlling only for
circuit), so a small car-induced component may be folded in. Documented, not hidden.

Usage: python v2/fit_incident.py [--data data/f1_results_2018_2025.parquet] [--tag _2018_2025]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "f1db.sqlite"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260701


def nice_names() -> dict:
    try:
        con = sqlite3.connect(DB); m = dict(con.execute("SELECT id, name FROM driver").fetchall())
        con.close(); return m
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "f1_results_2018_2025.parquet"))
    ap.add_argument("--tag", default="_2018_2025")
    ap.add_argument("--min-starts", type=int, default=20, help="min starts to list a driver")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    IDATA = ROOT / "models" / f"v2_incident_idata{args.tag}.pkl"
    RATES = ROOT / "models" / f"incident_rates{args.tag}.json"
    nm = nice_names()

    df = pd.read_parquet(args.data).copy()
    df["incident"] = df.dnf_cause.eq("driver_error").astype(int)
    drv = df.driver_id.astype("category")
    ct = df.circuit_type.astype("category")
    di = drv.cat.codes.values
    ci = ct.cat.codes.values
    drivers = list(drv.cat.categories)
    circuits = list(ct.cat.categories)
    D, C = len(drivers), len(circuits)
    y = df.incident.values.astype(float)
    base_rate = float(y.mean())
    base_logit = float(np.log(base_rate / (1 - base_rate)))

    coords = {"driver": drivers, "circuit_type": circuits}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", base_logit, 1.5)
        sigma_driver = pm.HalfNormal("sigma_driver", 1.0)
        sigma_circuit = pm.HalfNormal("sigma_circuit", 0.5)
        z_d = pm.Normal("z_d", 0.0, 1.0, dims="driver")               # non-centered
        z_c = pm.Normal("z_c", 0.0, 1.0, dims="circuit_type")
        incident = pm.Deterministic("incident", sigma_driver * z_d, dims="driver")
        hazard = pm.Deterministic("hazard", sigma_circuit * z_c, dims="circuit_type")
        logit_p = alpha + incident[di] + hazard[ci]
        pm.Bernoulli("y", logit_p=logit_p, observed=y)
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=4,
                          target_accept=0.95, random_seed=SEED, progressbar=False)

    with open(IDATA, "wb") as f:
        pickle.dump(idata, f)

    post = idata.posterior
    # per-driver per-race incident probability at the AVERAGE circuit mix (marginalize hazard by
    # its observed circuit-type frequencies), with credible intervals
    haz = post["hazard"].stack(s=("chain", "draw"))            # (circuit_type, s)
    ct_freq = df.circuit_type.value_counts(normalize=True).reindex(circuits).fillna(0).values
    mean_haz = (haz.values * ct_freq[:, None]).sum(0)          # (s,) circuit-freq-weighted hazard
    inc = post["incident"].stack(s=("chain", "draw"))          # (driver, s)
    a = post["alpha"].stack(s=("chain", "draw")).values        # (s,)
    starts = df.groupby("driver_id").size()
    rows = []
    for d in drivers:
        lp = a + inc.sel(driver=d).values + mean_haz           # (s,) log-odds at avg circuit
        p = 1.0 / (1.0 + np.exp(-lp))
        rows.append(dict(driver=d, p=float(p.mean()), lo=float(np.quantile(p, 0.05)),
                         hi=float(np.quantile(p, 0.95)), starts=int(starts.get(d, 0)),
                         raw=float(df[df.driver_id == d].incident.mean())))
    R = pd.DataFrame(rows).set_index("driver")
    listed = R[R.starts >= args.min_starts].sort_values("p", ascending=False)

    # save per-driver incident rates for the unified metric (all drivers, posterior mean)
    RATES.write_text(json.dumps(
        {"_overall": base_rate, **{d: R.loc[d, "p"] for d in R.index}}, indent=0))

    rh = float(az.summary(idata, var_names=["incident", "hazard", "alpha"])["r_hat"].max())
    haz_mean = post["hazard"].mean(("chain", "draw")).to_series()
    L = ["=" * 66, "v2 DRIVER INCIDENT-PRONENESS (driver-error-DNF risk) — RESULTS", "=" * 66,
         f"max R-hat = {rh:.3f}   base driver-error-DNF rate = {base_rate:.1%} per start",
         f"circuit hazard (log-odds vs avg): " +
         ", ".join(f"{c} {haz_mean[c]:+.2f}" for c in circuits),
         f"sigma_driver = {float(post['sigma_driver'].mean()):.2f} (spread of driver proneness)",
         "",
         f"[MOST INCIDENT-PRONE] shrunk per-race driver-error-DNF prob (>= {args.min_starts} starts)"]
    for d, r in listed.head(8).iterrows():
        L.append(f"    {nm.get(d, d):20s} {r.p:.1%}  90% CrI [{r.lo:.1%}, {r.hi:.1%}]  "
                 f"(raw {r.raw:.1%}, {int(r.starts)} starts)")
    L.append("\n[CLEANEST] least incident-prone")
    for d, r in listed.tail(8).iloc[::-1].iterrows():
        L.append(f"    {nm.get(d, d):20s} {r.p:.1%}  90% CrI [{r.lo:.1%}, {r.hi:.1%}]  "
                 f"(raw {r.raw:.1%}, {int(r.starts)} starts)")
    L.append("\n  (shrinkage pulls rare/short-career rates toward the field; note e.g. how far the")
    L.append("   raw rate moves for low-start drivers. Circuit hazard is controlled.)")
    L.append("  CAVEAT: attributed to the DRIVER (controlling for circuit only) — a small")
    L.append("  car-induced-error component may be folded in; see docstring.")

    report = "\n".join(L)
    print(report)
    (OUT / f"v2_incident_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: forest plot of shrunk incident probability ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = listed.sort_values("p")
    fig, ax = plt.subplots(figsize=(8, max(5, 0.32 * len(s))))
    yy = np.arange(len(s))
    ax.errorbar(s.p * 100, yy, xerr=[(s.p - s.lo) * 100, (s.hi - s.p) * 100],
                fmt="o", color="#7d2b2b", ecolor="#d6a6a6", capsize=2, ms=4)
    ax.scatter(s.raw * 100, yy, marker="|", s=120, color="#888", label="raw rate")
    ax.axvline(base_rate * 100, ls="--", color="grey", lw=0.8, label=f"field avg {base_rate:.1%}")
    ax.set_yticks(yy); ax.set_yticklabels([nm.get(d, d) for d in s.index], fontsize=8)
    ax.set_xlabel("driver-error-DNF probability per start (%, shrunk; bars = 90% CrI)")
    ax.set_title("Driver incident-proneness (partial-pooled, circuit hazard controlled)\n"
                 "how often each driver throws a result away by crashing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_incident_proneness{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/v2_incident_report{args.tag}.txt, "
          f"figures/v2_incident_proneness{args.tag}.png, {IDATA.name}, {RATES.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
