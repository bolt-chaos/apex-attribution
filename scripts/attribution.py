"""Phase (f): attribution + counterfactual swap demo + unified metric.

IMPORTANT RESULT (read the verdict): the v1 SCM does NOT identify the driver-vs-car
split. Each driver is nested in (almost) one constructor (Cramér's V 0.84, phase e), so
gcm loads the car's pace onto the finer driver_id partition. Consequence: the model says
the DRIVER explains almost everything and the CAR almost nothing — the exact failure the
project's sanity check is meant to catch ("if driver dominates the modern era, something
is wrong"). We therefore present the required gcm outputs, the diagnostic that exposes the
failure, and a teammate-contrast anchor showing the signal IS in the data — then point at
the deferred hierarchical latent skill/pace model as the fix. Do NOT trust the v1 numbers.

Per Option A, the unified "expected finish including breakdown risk" combines the finish
model with the reliability node ONLY at this reporting stage (no imputed DNF positions are
ever fed back into the finish mechanism).

Usage:
    python scripts/attribution.py [--icc-rand 150] [--icc-base 500] [--n 1200]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import dowhy.gcm as gcm


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "f1_results.parquet"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SEED = 20260622
DNF_LAST = 20.0  # reporting convention: a mechanical DNF lands at the back of a ~20-car field

NICE = {  # human labels for the demo
    "max-verstappen": "Verstappen", "lewis-hamilton": "Hamilton",
    "fernando-alonso": "Alonso", "lando-norris": "Norris", "george-russell": "Russell",
    "alexander-albon": "Albon", "yuki-tsunoda": "Tsunoda", "lance-stroll": "Stroll",
}


def load():
    scm = pickle.load(open(MODELS / "scm_finish.pkl", "rb"))
    rel = json.loads((MODELS / "reliability_rates.json").read_text())
    df = pd.read_parquet(DATA)
    return scm, rel, df


def exp_finish(scm, driver, constructor, circuit=None, n=1200):
    iv = {"driver_id": lambda x, d=driver: d, "constructor_id": lambda x, c=constructor: c}
    if circuit is not None:
        iv["circuit_type"] = lambda x, c=circuit: c
    return float(gcm.interventional_samples(scm, iv, num_samples_to_draw=n)["finish_pos"].mean())


# ---------------------------------------------------------------- 1. ICC
def run_icc(scm, rand, base, L):
    L.append("\n[1] intrinsic_causal_influence on finish_pos (variance shares)")
    icc = gcm.intrinsic_causal_influence(scm, "finish_pos",
                                         num_samples_randomization=rand,
                                         num_samples_baseline=base)
    tot = sum(abs(v) for v in icc.values())
    shares = {k: abs(v) / tot for k, v in icc.items()}
    for k, v in sorted(icc.items(), key=lambda kv: -abs(kv[1])):
        L.append(f"    {k:16s} {v:8.3f}   {100*abs(v)/tot:5.1f}%")
    car, drv = shares.get("constructor_id", 0), shares.get("driver_id", 0)
    ok = car > drv
    L.append(f"    SANITY CHECK (car should dominate): constructor {100*car:.1f}% vs "
             f"driver {100*drv:.1f}%  -> {'PASS' if ok else 'FAIL'}")
    return shares, ok


# ---------------------------------------------------------------- 2. interventional spreads
def run_spreads(scm, df, n, L):
    L.append("\n[2] interventional car-effect vs driver-effect (the diagnostic)")
    cohort_drivers = df.driver_id.value_counts().head(12).index.tolist()
    cars = sorted(df.constructor_id.unique())
    grid = pd.DataFrame(index=cohort_drivers, columns=cars, dtype=float)
    for d in cohort_drivers:
        for c in cars:
            grid.loc[d, c] = exp_finish(scm, d, c, n=n)
    grid.to_csv(OUT / "intervention_grid.csv")

    car_spread = (grid.max(axis=1) - grid.min(axis=1)).mean()   # vary car, fix driver
    drv_spread = (grid.max(axis=0) - grid.min(axis=0)).mean()   # vary driver, fix car
    L.append(f"    avg CAR effect    (fix driver, swap car):    {car_spread:5.2f} positions")
    L.append(f"    avg DRIVER effect (fix car, swap driver):    {drv_spread:5.2f} positions")
    L.append("    Reality: the CAR should dominate. Here driver effect >> car effect — the")
    L.append("    model attributes the car's pace to driver identity (non-identification).")
    return grid, car_spread, drv_spread


# ---------------------------------------------------------------- 3. counterfactual swaps
def run_counterfactuals(scm, df, L):
    L.append("\n[3] counterfactual 'put driver in another car' (machinery demo)")
    L.append("    NOTE: inherits the bias in [2]; swaps look muted. Shown to demo the query.")
    demos = [("fernando-alonso", "aston-martin", "red-bull"),
             ("alexander-albon", "williams", "red-bull"),
             ("yuki-tsunoda", "rb", "ferrari")]
    rows = []
    for drv, from_c, to_c in demos:
        sub = df[(df.driver_id == drv) & (df.constructor_id == from_c) & df.classified]
        if sub.empty:
            continue
        obs = sub.iloc[[0]][["circuit_type", "constructor_id", "driver_id", "grid", "finish_pos"]].copy()
        for col in ["circuit_type", "constructor_id", "driver_id"]:
            obs[col] = obs[col].astype("object")
        obs["grid"] = obs["grid"].astype(float)
        obs["finish_pos"] = obs["finish_pos"].astype(float)
        cf = gcm.counterfactual_samples(scm, {"constructor_id": lambda x, t=to_c: t}, observed_data=obs)
        L.append(f"    {NICE.get(drv, drv):10s}: actually P{int(obs.finish_pos.iloc[0])} in {from_c} "
                 f"-> P{int(round(cf.finish_pos.iloc[0]))} if in {to_c} (same driver, same luck)")
        rows.append((drv, from_c, to_c, float(obs.finish_pos.iloc[0]), float(cf.finish_pos.iloc[0])))
    return rows


# ---------------------------------------------------------------- 4. unified metric
def run_unified(scm, rel, L):
    L.append("\n[4] unified metric: expected finish INCLUDING mechanical-breakdown risk")
    L.append(f"    E_all = (1 - p_dnf) * E[finish | finished] + p_dnf * {DNF_LAST:.0f}  (DNF=back of field)")
    L.append("    (combination at reporting only; no imputed DNF feeds the finish model — Option A)")
    scenarios = [("max-verstappen", "red-bull"), ("max-verstappen", "haas"),
                 ("fernando-alonso", "aston-martin"), ("fernando-alonso", "alpine")]
    for drv, c in scenarios:
        e_fin = exp_finish(scm, drv, c, n=3000)
        p = rel.get(c, rel["_overall"])["p_mech_dnf"]
        e_all = (1 - p) * e_fin + p * DNF_LAST
        L.append(f"    {NICE.get(drv, drv):10s} in {c:13s}: E[finish|fin]={e_fin:5.2f}, "
                 f"p_mech_dnf={p:.3f} -> E_all={e_all:5.2f}")


# ---------------------------------------------------------------- 5. teammate-contrast anchor
def run_teammate_anchor(df, L):
    L.append("\n[5] teammate-contrast anchor (the signal IS in the data; the SCM just misses it)")
    L.append("    Within each constructor-season, median finish gap vs teammate(s), avg per driver.")
    cl = df[df.classified].copy()
    deltas: dict[str, list[float]] = {}
    for (_, _), g in cl.groupby(["year", "constructor_id"]):
        med = g.groupby("driver_id").finish_pos.median()
        if len(med) < 2:
            continue
        for d in med.index:
            others = med.drop(d).mean()
            deltas.setdefault(d, []).append(others - med[d])  # +ve = beats teammate
    rank = pd.Series({d: np.mean(v) for d, v in deltas.items()}).sort_values(ascending=False)
    L.append("    top 5 (beat teammates most):")
    for d, v in rank.head(5).items():
        L.append(f"      {d:20s} +{v:.2f}")
    L.append("    bottom 3:")
    for d, v in rank.tail(3).items():
        L.append(f"      {d:20s} {v:+.2f}")
    L.append("    This associational teammate delta is the identifying lever a future hierarchical")
    L.append("    latent skill/pace model would feed into the SCM (deferred per project scope).")


def make_figure(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    drv = "max-verstappen"
    car = "red-bull"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    s1 = grid.loc[drv].sort_values()
    ax1.barh([c for c in s1.index], s1.values, color="#ffb3a7")
    ax1.set_title(f"CAR effect: {NICE.get(drv, drv)} in each car\n(nearly flat — the bug)")
    ax1.set_xlabel("E[finish]"); ax1.invert_yaxis()
    s2 = grid[car].sort_values()
    ax2.barh([NICE.get(d, d) for d in s2.index], s2.values, color="#a7c7ff")
    ax2.set_title(f"DRIVER effect: each driver in the {car}\n(huge spread — pace mislabeled as skill)")
    ax2.set_xlabel("E[finish]"); ax2.invert_yaxis()
    fig.suptitle("v1 non-identification: car's pace gets attributed to driver identity", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "attribution_diagnostic.png", dpi=130, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc-rand", type=int, default=150)
    ap.add_argument("--icc-base", type=int, default=500)
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)

    scm, rel, df = load()
    L = ["=" * 72, "PHASE (f): ATTRIBUTION + COUNTERFACTUALS + UNIFIED METRIC", "=" * 72]

    shares, icc_ok = run_icc(scm, args.icc_rand, args.icc_base, L)
    grid, car_spread, drv_spread = run_spreads(scm, df, args.n, L)
    run_counterfactuals(scm, df, L)
    run_unified(scm, rel, L)
    run_teammate_anchor(df, L)
    make_figure(grid)

    L.append("\n" + "=" * 72)
    L.append("VERDICT")
    L.append("=" * 72)
    L.append(
        "v1 FAILS the literature bar (car should dominate the modern era). Both ICC and the\n"
        "interventional spreads put almost all explanatory weight on the DRIVER, because each\n"
        "driver is nested in one constructor (selection confounding, pitfall #1) and the raw\n"
        "categorical SCM cannot separate latent skill from latent car pace (pitfall #2). The\n"
        "teammate structure was used only to pick a connected cohort, NOT to identify relative\n"
        "skill — so the identifying lever is unused in the SCM itself.\n\n"
        "DELIVERED: the required gcm machinery runs end-to-end (ICC, interventional/counterfactual\n"
        "swaps, and the reliability-combined unified metric). DO NOT trust the v1 attribution\n"
        "numbers. FIX (deferred, out of v1 scope): a hierarchical latent skill/pace model that\n"
        "estimates constructor-level car pace and driver-level skill as separate parameters\n"
        "(identified via the teammate contrasts in [5]) and feeds those into the SCM.")
    report = "\n".join(L)
    print(report)
    (OUT / "attribution_report.txt").write_text(report + "\n")
    print("\nWrote outputs/attribution_report.txt, outputs/intervention_grid.csv, "
          "figures/attribution_diagnostic.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
