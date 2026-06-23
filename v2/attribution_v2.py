"""v2 phase 2b: race-outcome SCM with CONTINUOUS skill/pace nodes — the payoff.

DAG (parallels v1, but driver_skill / car_pace replace the nested categoricals):

    circuit_type ─┐
    driver_skill ─┼─→ grid ──────────┐
    car_pace ─────┘                  ├─→ finish_pos
    driver_skill ────────────────────┤
    car_pace ────────────────────────┤
    circuit_type ────────────────────┘

Because skill and pace are now decoupled (corr ~0.49, vs v1's near-perfect nesting) the SCM
CAN separate them. We re-run the SAME three lenses as v1 (ICC, interventional spreads,
counterfactual swap) for a direct contrast. RESULT: a large improvement (car effect ~4x v1's,
sensible counterfactual swaps) but car-dominance is still NOT reproduced — the race attribution
inherits the quali-stage split, where residual entanglement still overstates driver skill. See
the honest verdict in the generated report. The fix is better latent identification, not the SCM.

Usage: python v2/attribution_v2.py [--icc-rand 150] [--icc-base 500] [--n 2000]
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
DATA = ROOT / "data" / "f1_scm_v2.parquet"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260623
NICE = {"max-verstappen": "Verstappen", "fernando-alonso": "Alonso", "lewis-hamilton": "Hamilton",
        "lando-norris": "Norris", "george-russell": "Russell", "alexander-albon": "Albon",
        "yuki-tsunoda": "Tsunoda", "lance-stroll": "Stroll", "logan-sargeant": "Sargeant"}

NODES = ["circuit_type", "driver_skill", "car_pace", "grid", "finish_pos"]
EDGES = [("circuit_type", "grid"), ("driver_skill", "grid"), ("car_pace", "grid"),
         ("circuit_type", "finish_pos"), ("driver_skill", "finish_pos"),
         ("car_pace", "finish_pos"), ("grid", "finish_pos")]


def build_scm(data: pd.DataFrame):
    g = nx.DiGraph(EDGES)
    scm = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(scm, data, quality=gcm.auto.AssignmentQuality.GOOD)
    gcm.fit(scm, data)
    return scm


def exp_finish(scm, skill, pace, n=2000):
    iv = {"driver_skill": lambda x, s=skill: s, "car_pace": lambda x, p=pace: p}
    return float(gcm.interventional_samples(scm, iv, num_samples_to_draw=n)["finish_pos"].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc-rand", type=int, default=150)
    ap.add_argument("--icc-base", type=int, default=500)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--tag", default="", help="suffix for outputs, e.g. _2018_2025")
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    full = pd.read_parquet(args.data)
    df = full[full.classified].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        df[c] = df[c].astype(float)
    df["circuit_type"] = df["circuit_type"].astype("object")
    data = df[NODES].dropna()

    L = ["=" * 68, "v2 RACE-OUTCOME ATTRIBUTION (continuous skill/pace nodes)", "=" * 68,
         f"classified rows: {len(data)}"]

    scm = build_scm(data)

    # --- 1. ICC ---
    L.append("\n[1] intrinsic_causal_influence on finish_pos (variance shares)")
    icc = gcm.intrinsic_causal_influence(scm, "finish_pos",
                                         num_samples_randomization=args.icc_rand,
                                         num_samples_baseline=args.icc_base)
    tot = sum(abs(v) for v in icc.values())
    for k, v in sorted(icc.items(), key=lambda kv: -abs(kv[1])):
        L.append(f"    {k:14s} {v:8.3f}   {100*abs(v)/tot:5.1f}%")
    car, drv = abs(icc.get("car_pace", 0)) / tot, abs(icc.get("driver_skill", 0)) / tot
    L.append(f"    SANITY CHECK (car should dominate): car {100*car:.1f}% vs "
             f"driver {100*drv:.1f}%  -> {'PASS' if car > drv else 'FAIL'}")

    # --- 2. interventional spreads (FAIR: each swept over its full observed range) ---
    skill_by_d = full.groupby("driver_id").driver_skill.first()
    pace_by_c = full.groupby("constructor_id").car_pace.mean()        # for the figure (11 bars)
    pace_by_ty = full.groupby("team_year").car_pace.first()           # full range (34 team-years)
    skill_vals = sorted(skill_by_d.values)
    pace_vals = sorted(pace_by_ty.values)
    mid_skill, mid_pace = float(np.median(skill_vals)), float(np.median(pace_vals))

    # car effect: hold driver at median skill, sweep car pace over its full team-year range
    car_curve = [exp_finish(scm, mid_skill, p, args.n) for p in pace_vals]
    # driver effect: hold car at median pace, sweep driver skill over its full range
    drv_curve = [exp_finish(scm, s, mid_pace, args.n) for s in skill_vals]
    car_spread = max(car_curve) - min(car_curve)
    drv_spread = max(drv_curve) - min(drv_curve)
    L.append("\n[2] interventional car-effect vs driver-effect (each swept over full range)")
    L.append(f"    CAR effect    (median driver, sweep car {min(pace_vals):.2f}..{max(pace_vals):.2f}%):  {car_spread:5.2f} positions")
    L.append(f"    DRIVER effect (median car, sweep skill {min(skill_vals):+.2f}..{max(skill_vals):+.2f}%): {drv_spread:5.2f} positions")
    L.append(f"    v1 was car 2.3 << driver 8.4. v2: car {car_spread:.1f} vs driver {drv_spread:.1f} "
             f"({'car ahead' if car_spread>drv_spread else 'driver still ahead'}).")

    # figure data (illustrative, per-constructor)
    ref_drivers = full.driver_id.value_counts().head(12).index
    vers_by_car = {c: exp_finish(scm, skill_by_d["max-verstappen"], pace_by_c[c], args.n) for c in sorted(pace_by_c.index)}
    drivers_in_rb = {d: exp_finish(scm, skill_by_d[d], pace_by_c["red-bull"], args.n) for d in ref_drivers}

    # --- 3. counterfactual swaps ---
    L.append("\n[3] counterfactual 'put driver in another car' (now identified)")
    demos = [("fernando-alonso", "aston-martin", "red-bull"),
             ("alexander-albon", "williams", "red-bull"),
             ("logan-sargeant", "williams", "ferrari")]
    for drv_id, from_c, to_c in demos:
        sub = df[(df.driver_id == drv_id) & (df.constructor_id == from_c)]
        if sub.empty:
            continue
        obs = sub.iloc[[0]][NODES].copy()
        for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
            obs[c] = obs[c].astype(float)
        obs["circuit_type"] = obs["circuit_type"].astype("object")
        cf = gcm.counterfactual_samples(scm, {"car_pace": lambda x, p=pace_by_c[to_c]: p},
                                        observed_data=obs)
        L.append(f"    {NICE.get(drv_id, drv_id):10s}: P{int(obs.finish_pos.iloc[0])} in {from_c} "
                 f"-> P{int(round(cf.finish_pos.iloc[0]))} if in {to_c} (same driver+luck)")

    # --- OLS ground-truth: standardized betas (decisive, gcm-independent) ---
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    def sbeta(cols, target="finish_pos"):
        X = StandardScaler().fit_transform(data[cols].astype(float))
        y = (data[target] - data[target].mean()) / data[target].std()
        m = sm.OLS(y.values, sm.add_constant(X)).fit()
        return {c: round(float(b), 2) for c, b in zip(cols, m.params[1:])}
    L.append("\n[4] OLS standardized betas (ground truth; gcm-independent)")
    L.append(f"    finish ~ skill+pace      : {sbeta(['driver_skill','car_pace'])}")
    L.append(f"    finish ~ skill+pace+grid : {sbeta(['driver_skill','car_pace','grid'])}")
    L.append(f"    grid   ~ skill+pace      : {sbeta(['driver_skill','car_pace'],'grid')}")

    car_leads = car > drv
    head = ("CAR now leads the attribution — car-dominance reproduced."
            if car_leads else "driver still leads — car-dominance NOT fully reproduced.")
    L_tail = ("\n" + "=" * 68 + "\nVERDICT (honest)\n" + "=" * 68 + "\n"
              f"{head}\n"
              f"  - v1 (categorical):        ICC car 1.3% vs driver 45%; intervention car 2.3 << 8.4.\n"
              f"  - this run (skill/pace):   ICC car {100*car:.0f}% vs driver {100*drv:.0f}%; "
              f"intervention car {car_spread:.1f} vs driver {drv_spread:.1f}.\n"
              f"  - OLS betas (gcm-independent): finish ~ skill {sbeta(['driver_skill','car_pace'])['driver_skill']}"
              f", pace {sbeta(['driver_skill','car_pace'])['car_pace']}.\n"
              "The car/driver split tracks teammate-graph connectivity: the wider (2018-2025) era is\n"
              "one connected component, which de-confounds car pace from driver skill; the narrow\n"
              "(2022-2025) era fragments into 3 components and inflates the driver share.")
    report = "\n".join(L)
    print(report + L_tail)
    (OUT / f"v2_attribution_report{args.tag}.txt").write_text(report + L_tail + "\n")

    # --- figure: direct parallel to v1's diagnostic ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    s1 = pd.Series(vers_by_car).sort_values()
    a1.barh([c for c in s1.index], s1.values, color="#a7d8a0")
    a1.set_title("CAR effect: Verstappen in each car")
    a1.set_xlabel("E[finish]"); a1.invert_yaxis()
    s2 = pd.Series(drivers_in_rb).sort_values()
    a2.barh([NICE.get(d, d) for d in s2.index], s2.values, color="#a7c7ff")
    a2.set_title("DRIVER effect: each driver in the Red Bull")
    a2.set_xlabel("E[finish]"); a2.invert_yaxis()
    lead = "CAR leads" if car > drv else "driver leads"
    fig.suptitle(f"v2 skill & pace separated — {lead} by ICC: "
                 f"car {100*car:.0f}% vs driver {100*drv:.0f}% of finish variance", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_attribution_diagnostic{args.tag}.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote outputs/v2_attribution_report{args.tag}.txt, "
          f"figures/v2_attribution_diagnostic{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
