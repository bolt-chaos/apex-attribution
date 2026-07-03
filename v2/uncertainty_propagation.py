"""v2 phase 3: propagate posterior skill/pace uncertainty into the attribution.

v2 phase 2 fed posterior MEANS into the SCM — treating a noisy backmarker's skill as if it
were precise. That likely inflates the driver share: where a driver is poorly identified, the
posterior trades skill off against car pace (they are anti-correlated for entangled drivers),
and using only the mean throws that structure away.

Here we draw K JOINT samples from the posterior (skill and pace from the SAME draw, so the
anti-correlation is respected), re-fit the finish SCM and re-run ICC for each, and report the
DISTRIBUTION of the car-vs-driver split — a point estimate WITH a credible interval, plus
P(car > driver) — instead of one overconfident number.

Works for both the constant-skill model (skill indexed by driver) and the time-varying model
(skill indexed by driver x season; mapped onto race rows by (driver, year), mirroring
build_scm_data.py). The point-estimate baseline is computed in-script (ICC on the posterior
mean), so it is correct for whatever idata/era is passed.

Usage: python v2/uncertainty_propagation.py [--idata ...] [--results ...] [--tag ...]
       [--draws 30] [--icc-rand 100] [--icc-base 300]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import dowhy.gcm as gcm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260623

NODES = ["circuit_type", "driver_skill", "car_pace", "grid", "finish_pos"]
EDGES = [("circuit_type", "grid"), ("driver_skill", "grid"), ("car_pace", "grid"),
         ("circuit_type", "finish_pos"), ("driver_skill", "finish_pos"),
         ("car_pace", "finish_pos"), ("grid", "finish_pos")]


def _icc_shares(scm, rand, base):
    icc = gcm.intrinsic_causal_influence(
        scm, "finish_pos", num_samples_randomization=rand, num_samples_baseline=base)
    tot = sum(abs(v) for v in icc.values())
    return {n: 100 * abs(icc.get(n, 0)) / tot for n in
            ["car_pace", "driver_skill", "grid", "finish_pos", "circuit_type"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--icc-rand", type=int, default=100)
    ap.add_argument("--icc-base", type=int, default=300)
    ap.add_argument("--idata", default=str(ROOT / "models" / "v2_idata.pkl"))
    ap.add_argument("--results", default=str(ROOT / "data" / "f1_results.parquet"))
    ap.add_argument("--tag", default="", help="suffix for outputs, e.g. _2018_2025_rw")
    ap.add_argument("--var-skill", default="skill",
                    help="posterior var for driver_skill (joint model: racecraft or quali_skill)")
    ap.add_argument("--var-pace", default="pace",
                    help="posterior var for car_pace (joint model: pace_r or pace_q)")
    ap.add_argument("--hiring-edge", default=False, action=argparse.BooleanOptionalAction,
                    help="add driver_skill->car_pace. NOTE this is an ICC tool and ICC is now a "
                         "DEMOTED descriptive metric — the edge collapses the car's ICC share "
                         "(see attribution_v2's specification check). Default off (keeps the "
                         "descriptive independent-roots credible intervals).")
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    idata = pickle.load(open(Path(args.idata).resolve(), "rb"))
    post = idata.posterior
    skill_da = post[args.var_skill].stack(s=("chain", "draw"))   # (driver[, season], s)
    pace_da = post[args.var_pace].stack(s=("chain", "draw"))      # (team_year, s)
    time_varying = "season" in skill_da.dims               # driver x season skill?
    S = skill_da.sizes["s"]
    draw_idx = np.random.choice(S, size=min(args.draws, S), replace=False)

    base = pd.read_parquet(Path(args.results).resolve())
    base = base[base.classified].copy()
    base["team_year"] = base.constructor_id + "@" + base.year.astype(str)
    base["circuit_type"] = base["circuit_type"].astype("object")
    base["grid"] = base["grid"].astype(float)
    base["finish_pos"] = base["finish_pos"].astype(float)

    def data_for(k):
        """Build the per-draw dataframe. k=int draw index, or None for the posterior mean.
        Skill and pace come from the SAME draw (joint), preserving their correlation."""
        sk = (skill_da.isel(s=k) if k is not None else skill_da.mean("s")).to_pandas()
        pc = (pace_da.isel(s=k) if k is not None else pace_da.mean("s")).to_series()
        d = base.copy()
        if time_varying:   # sk is a DataFrame (driver x year) -> map by (driver, year)
            d["driver_skill"] = [sk.loc[dr, yr] if (dr in sk.index and yr in sk.columns)
                                 else np.nan for dr, yr in zip(d.driver_id, d.year)]
        else:              # sk is a Series indexed by driver
            d["driver_skill"] = d.driver_id.map(sk)
        d["driver_skill"] = d["driver_skill"].astype(float)
        d["car_pace"] = d.team_year.map(pc).astype(float)
        return d[NODES].dropna()

    # assign mechanisms ONCE on the posterior-mean data; re-fit per draw (fast)
    edges = EDGES + [("driver_skill", "car_pace")] if args.hiring_edge else EDGES
    mean_data = data_for(None)
    scm = gcm.InvertibleStructuralCausalModel(nx.DiGraph(edges))
    gcm.auto.assign_causal_mechanisms(scm, mean_data, quality=gcm.auto.AssignmentQuality.GOOD)

    # point-estimate baseline: ICC on the posterior-mean data (correct for any model/era)
    gcm.fit(scm, mean_data)
    pt = _icc_shares(scm, args.icc_rand, args.icc_base)
    PT = {"car": pt["car_pace"], "driver": pt["driver_skill"]}

    rows = []
    for i, k in enumerate(draw_idx):
        gcm.fit(scm, data_for(int(k)))
        rows.append(_icc_shares(scm, args.icc_rand, args.icc_base))
        print(f"  draw {i+1}/{len(draw_idx)}: car {rows[-1]['car_pace']:.1f}%  "
              f"driver {rows[-1]['driver_skill']:.1f}%")

    df = pd.DataFrame(rows)
    car, drv = df.car_pace, df.driver_skill
    ratio = drv / car
    p_car_leads = float((car > drv).mean())   # is "car leads" real?

    def ci(x):
        return np.percentile(x, 5), np.median(x), np.percentile(x, 95)

    graph = "hiring edge (driver_skill->car_pace)" if args.hiring_edge else "independent roots"
    L = ["=" * 68, "v2 UNCERTAINTY-PROPAGATED ATTRIBUTION (ICC over posterior draws)", "=" * 68,
         f"draws: {len(draw_idx)}  (ICC rand={args.icc_rand}, base={args.icc_base})  graph: {graph}",
         "CAVEAT: ICC is a DEMOTED descriptive metric. Beyond this sampling CrI, the ICC split also",
         "swings ~25pp depending on whether the skill<->car_pace confounding is in the graph (run",
         "with --hiring-edge; see attribution_v2's specification check) — a graph uncertainty that",
         "DWARFS the sampling one below. The graph-robust car-vs-driver answer is the interventional",
         "/ necessity result in attribution_v2, not this variance share.", ""]
    for name, x in [("car_pace", car), ("driver_skill", drv), ("grid", df.grid),
                    ("finish_pos resid", df.finish_pos)]:
        lo, md, hi = ci(x)
        L.append(f"    {name:18s} median {md:5.1f}%   90% CrI [{lo:5.1f}, {hi:5.1f}]")
    rlo, rmd, rhi = ci(ratio)
    L.append(f"\n    driver:car ratio   median {rmd:4.1f}x   90% CrI [{rlo:.1f}, {rhi:.1f}]")
    L.append(f"\n    P(car > driver) across draws = {p_car_leads:.0%}   <- is 'car leads' real?")
    L.append(f"\n    vs POINT estimate (ICC on posterior mean): "
             f"car {PT['car']:.1f}%, driver {PT['driver']:.1f}% (ratio {PT['driver']/PT['car']:.1f}x)")
    car_lo, car_md, car_hi = ci(car)
    drv_lo, drv_md, drv_hi = ci(drv)
    L.append("\n[INTERPRETATION]")
    L.append(f"    Car share:    point {PT['car']:.1f}% -> posterior median {car_md:.1f}% "
             f"(90% CrI [{car_lo:.1f}, {car_hi:.1f}]).")
    L.append(f"    Driver share: point {PT['driver']:.1f}% -> posterior median {drv_md:.1f}% "
             f"(90% CrI [{drv_lo:.1f}, {drv_hi:.1f}]).")
    L.append(f"    Driver:car ratio median {rmd:.1f}x.   P(car > driver) = {p_car_leads:.0%}.")
    if p_car_leads >= 0.95:
        L.append("    'CAR LEADS' IS ROBUST: car > driver in >=95% of posterior draws, even after")
        L.append("    propagating the full skill/pace uncertainty. The headline holds.")
    elif p_car_leads >= 0.5:
        L.append("    'Car leads' is the MODAL outcome but NOT decisive: the car and driver CrIs")
        L.append("    overlap, so the split is not sharply pinned -- report it as a range, and")
        L.append("    treat any close car-vs-driver claim with caution.")
    else:
        L.append("    The posterior is DRIVER-HEAVY: car leads in a minority of draws, so once")
        L.append("    SCM-stage uncertainty is propagated 'car leads' is not actually supported.")

    report = "\n".join(L)
    print("\n" + report)
    (OUT / f"v2_uncertainty_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: distribution of car vs driver share ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.violinplot([car.values, drv.values], positions=[1, 2], showmedians=True)
    ax.axhline(PT["car"], ls="--", color="#2a7", lw=1, label=f"point: car {PT['car']:.1f}%")
    ax.axhline(PT["driver"], ls="--", color="#37a", lw=1, label=f"point: driver {PT['driver']:.1f}%")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["car_pace", "driver_skill"])
    ax.set_ylabel("ICC share of finish_pos variance (%)")
    ax.set_title(f"Attribution with posterior uncertainty propagated  "
                 f"(P(car>driver)={p_car_leads:.0%})\n"
                 "violins = distribution over posterior draws; dashed = point est (ICC on mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_uncertainty{args.tag}.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote outputs/v2_uncertainty_report{args.tag}.txt, figures/v2_uncertainty{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
