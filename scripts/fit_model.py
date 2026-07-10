"""Phase (d): assign causal mechanisms and fit the SCMs.

Two fitted artifacts, per Option A (see dag.py / project memory):

  1. scm_finish  — InvertibleStructuralCausalModel over the finish_pos subgraph
                   (circuit_type, constructor_id, driver_id, grid, finish_pos),
                   fit on CLASSIFIED rows only. Invertible so phase (f) can run
                   counterfactual_samples on it (the "swap the car" demo).
  2. scm_reliability — StructuralCausalModel (constructor_id -> reliability_dnf),
                   fit on ALL started rows. The censoring leaf, combined with
                   scm_finish at phase (f) for the unified metric.

Both are pickled to models/. A small empirical P(mechanical DNF | constructor) table
is also written for transparency.

Usage:
    python scripts/fit_model.py [--quality GOOD|BETTER|BEST]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import dowhy.gcm as gcm
from dowhy.gcm import auto

from dag import build_dag, finish_scm_graph  # scripts/ is on sys.path when run as a script

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "f1_results.parquet"
MODELS = ROOT / "models"
SEED = 20260622

CATEGORICAL = ["circuit_type", "constructor_id", "driver_id"]
CONTINUOUS = ["grid", "finish_pos"]


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (classified_finish_df, started_df) with gcm-friendly dtypes."""
    df = pd.read_parquet(DATA)
    # gcm.auto detects categoricals by dtype; force plain-object strings to be safe.
    for c in CATEGORICAL:
        df[c] = df[c].astype("object")

    started = df.copy()
    # reliability target as explicit string labels (categorical classification target)
    started["reliability_dnf"] = np.where(started["reliability_dnf"], "dnf", "ok")

    classified = df[df["classified"]].copy()
    for c in CONTINUOUS:
        classified[c] = classified[c].astype(float)
    return classified, started


def fit_finish_scm(classified: pd.DataFrame, quality: auto.AssignmentQuality):
    g = finish_scm_graph(build_dag())  # drops reliability_dnf leaf
    cols = list(g.nodes)
    data = classified[cols].dropna()
    scm = gcm.InvertibleStructuralCausalModel(g)
    print(f"\n[finish SCM] assigning mechanisms on {len(data)} classified rows, nodes={sorted(cols)}")
    summary = auto.assign_causal_mechanisms(scm, data, quality=quality)
    print(summary)
    print("[finish SCM] fitting ...")
    gcm.fit(scm, data)
    return scm, data


def fit_reliability_scm(started: pd.DataFrame, quality: auto.AssignmentQuality):
    import networkx as nx
    g = nx.DiGraph([("constructor_id", "reliability_dnf")])
    data = started[["constructor_id", "reliability_dnf"]].copy()
    scm = gcm.StructuralCausalModel(g)
    print(f"\n[reliability SCM] assigning mechanisms on {len(data)} started rows")
    summary = auto.assign_causal_mechanisms(scm, data, quality=quality)
    print(summary)
    print("[reliability SCM] fitting ...")
    gcm.fit(scm, data)
    return scm, data


def empirical_reliability_table(started: pd.DataFrame) -> dict:
    g = started.assign(is_dnf=started.reliability_dnf.eq("dnf"))
    rates = g.groupby("constructor_id").is_dnf.mean().round(4)
    counts = g.groupby("constructor_id").size()
    table = {c: {"p_mech_dnf": float(rates[c]), "n_started": int(counts[c])}
             for c in rates.index}
    table["_overall"] = {"p_mech_dnf": float(g.is_dnf.mean()), "n_started": int(len(g))}
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="GOOD", choices=["GOOD", "BETTER", "BEST"])
    args = ap.parse_args()
    quality = getattr(auto.AssignmentQuality, args.quality)

    np.random.seed(SEED)
    gcm.util.general.set_random_seed(SEED) if hasattr(gcm.util.general, "set_random_seed") else None
    MODELS.mkdir(exist_ok=True)

    classified, started = load_frames()

    scm_finish, finish_data = fit_finish_scm(classified, quality)
    scm_reliability, _ = fit_reliability_scm(started, quality)
    rel_table = empirical_reliability_table(started)

    # --- smoke test: a fitted SCM must be able to generate samples ---
    print("\n[smoke test] draw 5 samples from fitted finish SCM:")
    samp = gcm.draw_samples(scm_finish, num_samples=5)
    print(samp.to_string(index=False))

    # --- persist ---
    with open(MODELS / "scm_finish.pkl", "wb") as f:
        pickle.dump(scm_finish, f)
    with open(MODELS / "scm_reliability.pkl", "wb") as f:
        pickle.dump(scm_reliability, f)
    (MODELS / "reliability_rates.json").write_text(json.dumps(rel_table, indent=2))

    print("\n" + "=" * 64)
    print("FIT SUMMARY")
    print("=" * 64)
    print(f"finish SCM: {len(finish_data)} classified rows; nodes {sorted(scm_finish.graph.nodes)}")
    print("assigned mechanisms:")
    for n in scm_finish.graph.nodes:
        print(f"  {n:16s} {type(scm_finish.causal_mechanism(n)).__name__}")
    print("\nreliability SCM: constructor_id -> reliability_dnf")
    print(f"  reliability_dnf  {type(scm_reliability.causal_mechanism('reliability_dnf')).__name__}")
    print("\nempirical P(mechanical DNF | constructor), worst 5:")
    worst = sorted(((v["p_mech_dnf"], k, v["n_started"]) for k, v in rel_table.items()
                    if k != "_overall"), reverse=True)[:5]
    for p, c, n in worst:
        print(f"  {c:16s} {p:.3f}  (n={n})")
    print(f"  {'OVERALL':16s} {rel_table['_overall']['p_mech_dnf']:.3f}")
    print("\nWrote models/scm_finish.pkl, models/scm_reliability.pkl, models/reliability_rates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
