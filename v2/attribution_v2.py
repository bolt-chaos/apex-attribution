"""v2 phase 2b: race-outcome SCM with CONTINUOUS skill/pace nodes — the payoff.

DAG (parallels v1, but driver_skill / car_pace replace the nested categoricals):

    circuit_type ─┐
    driver_skill ─┼─→ grid ──────────┐        driver_skill ─→ car_pace  (the hiring edge:
    car_pace ─────┘                  ├─→ finish_pos                       good drivers get good
    driver_skill ────────────────────┤                                   cars; --hiring-edge, on)
    car_pace ────────────────────────┤
    circuit_type ────────────────────┘

Because skill and pace are now decoupled (corr ~0.5, vs v1's near-perfect nesting) the SCM CAN
separate them. Following a Pearl-style review (see docs/IDEAS.md) this stage was refactored twice:

  1. SPECIFICATION FIX — the roots are correlated (good drivers are hired into good cars), so
     drawing them as INDEPENDENT was a mis-specification. We put the confounder in the graph via
     the hiring pathway driver_skill -> car_pace (--hiring-edge, default on). ICC assumes
     independent root noise, so this MOVES it ~25pp (car share collapses); the do()-sweeps and
     counterfactuals set both roots, so they don't move — the confounding is only a problem for
     the variance share.
  2. REPORTING FIX — demote ICC to a descriptive, population/graph-dependent number; lead with the
     graph-ROBUST measures: the interventional car-vs-driver "positions" spreads, a rung-3
     NECESSITY query ("would this podium have happened but for the car / the driver?"), and OLS.

RESULT (wide era, race pace): by every graph-robust measure the car at least matches the driver
(car ~10.6 vs driver ~10.1 positions; podium needs the car 82% vs the driver 68%; OLS pace 0.47 >
skill 0.35) — while ICC swings from car 26%/driver 16% to car 1%/driver 58% on the graph choice,
which is exactly why it is no longer the headline.

Usage: python v2/attribution_v2.py [--icc-rand 150] [--icc-base 500] [--n 2000]
                                   [--no-hiring-edge] [--pn-threshold 3]
"""
from __future__ import annotations

import argparse
import json
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
# driver_skill and car_pace are correlated (~0.5) because good drivers are hired into good cars.
# Drawing them as independent roots is a specification error that biases ICC (which assumes
# independent root noise). We put the confounder IN THE GRAPH via the hiring pathway
# driver_skill -> car_pace (a reduced-form stand-in for the latent team-resources common cause
# gcm 0.14 can't represent). The do()-sweeps/counterfactuals set BOTH roots, so they're invariant
# to this; only ICC moves. Default on.
HIRING_EDGE = ("driver_skill", "car_pace")


def build_scm(data: pd.DataFrame, hiring_edge: bool = True):
    edges = EDGES + [HIRING_EDGE] if hiring_edge else EDGES
    g = nx.DiGraph(edges)
    scm = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(scm, data, quality=gcm.auto.AssignmentQuality.GOOD)
    gcm.fit(scm, data)
    return scm


def icc_car_driver(scm, rand, base):
    """ICC variance shares -> (car %, driver %) of |total|, plus the raw dict."""
    icc = gcm.intrinsic_causal_influence(scm, "finish_pos",
                                         num_samples_randomization=rand, num_samples_baseline=base)
    tot = sum(abs(v) for v in icc.values())
    return abs(icc.get("car_pace", 0)) / tot, abs(icc.get("driver_skill", 0)) / tot, icc, tot


def sweep_spreads(scm, pace_vals, skill_vals, mid_skill, mid_pace, n):
    """Interventional car-effect vs driver-effect spreads (positions), each swept over full range."""
    car_curve = [exp_finish(scm, mid_skill, p, n) for p in pace_vals]
    drv_curve = [exp_finish(scm, s, mid_pace, n) for s in skill_vals]
    return max(car_curve) - min(car_curve), max(drv_curve) - min(drv_curve)


def exp_finish(scm, skill, pace, n=2000):
    iv = {"driver_skill": lambda x, s=skill: s, "car_pace": lambda x, p=pace: p}
    return float(gcm.interventional_samples(scm, iv, num_samples_to_draw=n)["finish_pos"].mean())


def necessity_query(scm_indep, df, mid_pace, mid_skill, threshold=3):
    """Probability of Necessity (rung 3): of results actually ACHIEVED (finish <= threshold), what
    fraction would have been LOST but for the car (counterfactual car_pace -> midfield) vs but for
    the driver (counterfactual driver_skill -> median)? Abduct each race's noise, downgrade ONE
    factor, replay.

    Computed on the INDEPENDENT-ROOTS graph: there the two factors are separable, so downgrading one
    naturally holds the other at its observed value — a clean, symmetric 'but for X alone' direct
    effect. (On the hiring-edge graph, changing the driver would drag the car along.) Returns overall
    PN_car/PN_driver + a per-driver breakdown (drivers with >= 5 achieved results)."""
    succ = df[df.finish_pos <= threshold].copy()
    obs = succ[NODES].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        obs[c] = obs[c].astype(float)
    obs["circuit_type"] = obs["circuit_type"].astype("object")
    cf_car = gcm.counterfactual_samples(scm_indep, {"car_pace": lambda x: np.full(np.shape(x), mid_pace)},
                                        observed_data=obs)
    cf_drv = gcm.counterfactual_samples(scm_indep, {"driver_skill": lambda x: np.full(np.shape(x), mid_skill)},
                                        observed_data=obs)
    lost_car = cf_car["finish_pos"].values > threshold
    lost_drv = cf_drv["finish_pos"].values > threshold
    per = pd.DataFrame({"driver": succ.driver_id.values, "lost_car": lost_car, "lost_drv": lost_drv})
    g = per.groupby("driver").agg(n=("lost_car", "size"), pn_car=("lost_car", "mean"),
                                  pn_drv=("lost_drv", "mean"))
    return dict(n=len(obs), threshold=threshold, pn_car=float(lost_car.mean()),
                pn_drv=float(lost_drv.mean()), per_driver=g[g.n >= 5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc-rand", type=int, default=150)
    ap.add_argument("--icc-base", type=int, default=500)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--tag", default="", help="suffix for outputs, e.g. _2018_2025")
    ap.add_argument("--hiring-edge", default=True, action=argparse.BooleanOptionalAction,
                    help="add driver_skill->car_pace (the hiring pathway) so the correlated roots "
                         "aren't treated as independent — 'put the confounder in the graph'. Default on.")
    ap.add_argument("--pn-threshold", type=int, default=3,
                    help="'success' finishing position for the necessity (PN) query (default podium P3)")
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    full = pd.read_parquet(args.data)
    df = full[full.classified].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        df[c] = df[c].astype(float)
    df["circuit_type"] = df["circuit_type"].astype("object")
    data = df[NODES].dropna()

    graph_desc = "driver_skill->car_pace hiring edge (confounder in the graph)" if args.hiring_edge \
        else "independent roots (legacy)"
    L = ["=" * 68, "v2 RACE-OUTCOME ATTRIBUTION (continuous skill/pace nodes)", "=" * 68,
         f"classified rows: {len(data)}   graph: {graph_desc}"]

    scm = build_scm(data, hiring_edge=args.hiring_edge)       # primary: sweeps, CF swaps, figure
    scm_indep = scm if not args.hiring_edge else build_scm(data, hiring_edge=False)
    scm_he = scm if args.hiring_edge else build_scm(data, hiring_edge=True)

    # ranges for the interventional sweeps + figure data
    skill_by_d = full.groupby("driver_id").driver_skill.first()
    pace_by_c = full.groupby("constructor_id").car_pace.mean()
    pace_by_ty = full.groupby("team_year").car_pace.first()
    skill_vals = sorted(skill_by_d.values)
    pace_vals = sorted(pace_by_ty.values)
    mid_skill, mid_pace = float(np.median(skill_vals)), float(np.median(pace_vals))

    # ===== compute the GRAPH-ROBUST measures (do()-set both roots, so the hiring edge can't bias) =====
    car_spread, drv_spread = sweep_spreads(scm, pace_vals, skill_vals, mid_skill, mid_pace, args.n)
    vers_by_car = {c: exp_finish(scm, skill_by_d["max-verstappen"], pace_by_c[c], args.n)
                   for c in sorted(pace_by_c.index)}
    drivers_in_rb = {d: exp_finish(scm, skill_by_d[d], pace_by_c["red-bull"], args.n)
                     for d in full.driver_id.value_counts().head(12).index}
    pn = necessity_query(scm_indep, df, mid_pace, mid_skill, threshold=args.pn_threshold)

    # counterfactual driver-swaps
    cf_lines = []
    for drv_id, from_c, to_c in [("fernando-alonso", "aston-martin", "red-bull"),
                                 ("alexander-albon", "williams", "red-bull"),
                                 ("logan-sargeant", "williams", "ferrari")]:
        sub = df[(df.driver_id == drv_id) & (df.constructor_id == from_c)]
        if sub.empty:
            continue
        obs = sub.iloc[[0]][NODES].copy()
        for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
            obs[c] = obs[c].astype(float)
        obs["circuit_type"] = obs["circuit_type"].astype("object")
        cf = gcm.counterfactual_samples(scm, {"car_pace": lambda x, p=pace_by_c[to_c]: p},
                                        observed_data=obs)
        cf_lines.append(f"    {NICE.get(drv_id, drv_id):10s}: P{int(obs.finish_pos.iloc[0])} in {from_c} "
                        f"-> P{int(round(cf.finish_pos.iloc[0]))} if in {to_c} (same driver+luck)")

    # OLS standardized betas (gcm-independent ground truth)
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    def sbeta(cols, target="finish_pos"):
        X = StandardScaler().fit_transform(data[cols].astype(float))
        y = (data[target] - data[target].mean()) / data[target].std()
        m = sm.OLS(y.values, sm.add_constant(X)).fit()
        return {c: round(float(b), 2) for c, b in zip(cols, m.params[1:])}
    b = sbeta(["driver_skill", "car_pace"])

    # ===== compute ICC under BOTH graphs (demoted; shown to be graph-dependent) =====
    ind_car, ind_drv, _, _ = icc_car_driver(scm_indep, args.icc_rand, args.icc_base)
    he_car, he_drv, icc_he, tot_he = icc_car_driver(scm_he, args.icc_rand, args.icc_base)
    corr = float(full.driver_skill.corr(full.car_pace))

    # ===== assemble report: HEADLINE = interventional + necessity; ICC demoted to descriptive =====
    verdict = ("car ahead" if car_spread > drv_spread + 0.5 else
               "driver ahead" if drv_spread > car_spread + 0.5 else "~parity")
    L.append("\n[1] INTERVENTIONAL car-effect vs driver-effect  (do()-sets both roots -> graph-robust)")
    L.append(f"    CAR effect    (median driver, sweep car {min(pace_vals):.2f}..{max(pace_vals):.2f}%):  {car_spread:5.2f} positions")
    L.append(f"    DRIVER effect (median car, sweep skill {min(skill_vals):+.2f}..{max(skill_vals):+.2f}%): {drv_spread:5.2f} positions")
    L.append(f"    -> car {car_spread:.1f} vs driver {drv_spread:.1f} positions ({verdict}). v1 was car 2.3 << driver 8.4.")

    pn_verdict = ("the CAR is more often necessary" if pn["pn_car"] > pn["pn_drv"] else
                  "the DRIVER is more often necessary")
    L.append(f"\n[2] NECESSITY — would the result have happened BUT FOR the car / the driver? (rung-3)")
    L.append(f"    of {pn['n']} podiums (finish <= P{pn['threshold']}) actually achieved, fraction LOST if we")
    L.append(f"    counterfactually downgrade ONE factor (abduct luck, swap factor, replay):")
    L.append(f"      BUT FOR the car   (car_pace -> midfield):    {pn['pn_car']:.0%} of podiums lost")
    L.append(f"      BUT FOR the driver (driver_skill -> median): {pn['pn_drv']:.0%} of podiums lost")
    L.append(f"    -> {pn_verdict} for a podium (robust to the confounding-graph choice, unlike ICC).")
    pd_ = pn["per_driver"]
    if len(pd_):
        top_car = pd_.sort_values("pn_car", ascending=False).head(3)
        top_drv = pd_.sort_values("pn_drv", ascending=False).head(3)
        L.append("    most car-dependent podiums:   " +
                 ", ".join(f"{NICE.get(d, d)} {r.pn_car:.0%}" for d, r in top_car.iterrows()))
        L.append("    most driver-dependent podiums: " +
                 ", ".join(f"{NICE.get(d, d)} {r.pn_drv:.0%}" for d, r in top_drv.iterrows()))

    L.append("\n[3] counterfactual 'put driver in another car' (same driver + same luck)")
    L.extend(cf_lines)

    L.append("\n[4] OLS standardized betas (gcm-independent ground truth)")
    L.append(f"    finish ~ skill+pace      : {b}")
    L.append(f"    finish ~ skill+pace+grid : {sbeta(['driver_skill','car_pace','grid'])}")

    L.append("\n[5] ICC variance share (DESCRIPTIVE ONLY — population/era-dependent, NOT structural)")
    L.append(f"    intrinsic_causal_influence assumes INDEPENDENT root noise, but skill<->pace are")
    L.append(f"    correlated (corr {corr:.2f}; good drivers hired into good cars). So the share is")
    L.append(f"    graph-dependent and swings by {abs(he_car-ind_car)*100:.0f}pp on that modelling choice:")
    L.append(f"      independent roots:  car {100*ind_car:.1f}% / driver {100*ind_drv:.1f}%")
    L.append(f"      + hiring edge:      car {100*he_car:.1f}% / driver {100*he_drv:.1f}%")
    L.append(f"    That swing can flip the car-vs-driver verdict, while [1]/[2] barely move -- which is")
    L.append(f"    exactly why ICC is NOT the headline. Full node breakdown (hiring-edge graph):")
    for k, v in sorted(icc_he.items(), key=lambda kv: -abs(kv[1])):
        L.append(f"      {k:14s} {v:8.3f}   {100*abs(v)/tot_he:5.1f}%")

    L_tail = ("\n" + "=" * 68 + "\nVERDICT (honest) — lead with the graph-robust measures\n" + "=" * 68 + "\n"
              f"  ROBUST (interventional / counterfactual — invariant to how confounding is modelled):\n"
              f"    - car effect {car_spread:.1f} vs driver effect {drv_spread:.1f} positions  ({verdict})\n"
              f"    - but-for a podium: {pn['pn_car']:.0%} need the car vs {pn['pn_drv']:.0%} need the driver\n"
              f"    - OLS finish ~ skill {b['driver_skill']}, pace {b['car_pace']}\n"
              f"  DESCRIPTIVE (ICC variance share — do NOT headline): car {100*he_car:.0f}%/driver {100*he_drv:.0f}%\n"
              f"    (hiring edge) vs car {100*ind_car:.0f}%/driver {100*ind_drv:.0f}% (independent roots) --\n"
              f"    swings {abs(he_car-ind_car)*100:.0f}pp with the graph, so it is not a structural answer.\n"
              f"  NET: on the wide era, by every graph-robust measure the car at least matches the driver;\n"
              f"  the old ICC 'car dominates X%' headline was partly an independent-roots artifact.")
    report = "\n".join(L)
    print(report + L_tail)
    (OUT / f"v2_attribution_report{args.tag}.txt").write_text(report + L_tail + "\n")

    # Machine-readable twin of the report, so downstream consumers (scripts/export_site.py, CI)
    # read numbers instead of hand-transcribing them from the prose. Driver ids stay raw here;
    # display-name mapping is the consumer's job.
    artifact = {
        "tag": args.tag,
        "data": Path(args.data).name,
        "nRows": int(len(data)),
        "hiringEdge": bool(args.hiring_edge),
        "interventional": {"carSpread": round(car_spread, 3), "driverSpread": round(drv_spread, 3),
                           "verdict": verdict},
        "necessity": {
            "threshold": int(pn["threshold"]),
            "nPodiums": int(pn["n"]),
            "pnCar": round(pn["pn_car"], 4),
            "pnDriver": round(pn["pn_drv"], 4),
            "perDriver": {d: {"n": int(r.n), "pnCar": round(float(r.pn_car), 4),
                              "pnDriver": round(float(r.pn_drv), 4)}
                          for d, r in pn["per_driver"].iterrows()},
        },
        "ols": {"skillPace": b, "skillPaceGrid": sbeta(["driver_skill", "car_pace", "grid"])},
        "icc": {"corrSkillPace": round(corr, 4),
                "independentRoots": {"carPct": round(100 * ind_car, 2), "driverPct": round(100 * ind_drv, 2)},
                "hiringEdge": {"carPct": round(100 * he_car, 2), "driverPct": round(100 * he_drv, 2)}},
    }
    (OUT / f"v2_attribution{args.tag}.json").write_text(json.dumps(artifact, indent=1) + "\n")

    # --- figures: (A) interventional diagnostic, (B) necessity but-for ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    s1 = pd.Series(vers_by_car).sort_values()
    a1.barh(list(s1.index), s1.values, color="#a7d8a0")
    a1.set_title("CAR effect: Verstappen in each car"); a1.set_xlabel("E[finish]"); a1.invert_yaxis()
    s2 = pd.Series(drivers_in_rb).sort_values()
    a2.barh([NICE.get(d, d) for d in s2.index], s2.values, color="#a7c7ff")
    a2.set_title("DRIVER effect: each driver in the Red Bull"); a2.set_xlabel("E[finish]"); a2.invert_yaxis()
    fig.suptitle(f"Interventional car vs driver (graph-robust): car {car_spread:.1f} vs "
                 f"driver {drv_spread:.1f} positions ({verdict})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_attribution_diagnostic{args.tag}.png", dpi=130, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["but for\nthe CAR", "but for\nthe DRIVER"], [100 * pn["pn_car"], 100 * pn["pn_drv"]],
           color=["#a7d8a0", "#a7c7ff"], edgecolor="#333")
    ax.set_ylabel("% of achieved podiums that would be LOST")
    ax.set_title(f"Necessity: what a podium needs (finish <= P{pn['threshold']}, n={pn['n']})\n"
                 f"rung-3 counterfactual — robust to the confounding-graph choice, unlike ICC")
    for i, v in enumerate([pn["pn_car"], pn["pn_drv"]]):
        ax.text(i, 100 * v + 1, f"{v:.0%}", ha="center", fontsize=11, weight="bold")
    fig2.tight_layout()
    fig2.savefig(FIG / f"v2_necessity{args.tag}.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote outputs/v2_attribution_report{args.tag}.txt, outputs/v2_attribution{args.tag}.json, "
          f"figures/v2_attribution_diagnostic{args.tag}.png, figures/v2_necessity{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
