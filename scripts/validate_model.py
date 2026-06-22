"""Phase (e): validate the causal model BEFORE trusting any attribution number.

Three checks:
  1. gcm.evaluate_causal_model  — do the fitted mechanisms reproduce the data?
     (per-mechanism quality, invertibility assumptions, overall KL divergence)
  2. falsify_graph              — is the assumed DAG consistent with the data, or do
     the implied conditional-independences get rejected? (kernel-based, subsampled +
     parallelized for tractability)
  3. root-independence chi-square — a fast, crisp test of the three independent-roots
     assumptions (driver ⊥ constructor ⊥ circuit_type). We EXPECT driver⊥constructor to
     fail: good drivers are hired into good cars (selection confounding, pitfall #1).
     This check quantifies that leak.

Writes outputs/validation_report.txt and prints a verdict.

Usage:
    python scripts/validate_model.py [--falsify-rows 500] [--falsify-perms 20]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import dowhy.gcm as gcm
from dowhy.gcm.falsify import falsify_graph

from dag import build_dag, finish_scm_graph

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "f1_results.parquet"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
SEED = 20260622
ROOTS = ["circuit_type", "constructor_id", "driver_id"]


def load():
    with open(MODELS / "scm_finish.pkl", "rb") as f:
        scm = pickle.load(f)
    df = pd.read_parquet(DATA)
    df = df[df.classified].copy()
    cols = list(finish_scm_graph(build_dag()).nodes)
    d = df[cols].dropna()
    for c in ROOTS:
        d[c] = d[c].astype("object")
    d["grid"] = d["grid"].astype(float)
    d["finish_pos"] = d["finish_pos"].astype(float)
    return scm, d


def cramers_v(table: np.ndarray, chi2: float) -> float:
    n = table.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def root_independence(d: pd.DataFrame, lines: list[str]) -> bool:
    lines.append("\n[3] ROOT-INDEPENDENCE CHI-SQUARE TESTS")
    lines.append("    (the DAG assumes all three roots are mutually independent)")
    any_violation = False
    for a, b in [("driver_id", "constructor_id"),
                 ("driver_id", "circuit_type"),
                 ("constructor_id", "circuit_type")]:
        tab = pd.crosstab(d[a], d[b])
        chi2, p, dof, _ = chi2_contingency(tab)
        v = cramers_v(tab.values, chi2)
        violated = p < 0.05
        any_violation |= violated
        verdict = "VIOLATED (dependent)" if violated else "ok (independent)"
        lines.append(f"    {a} ⊥ {b}: p={p:.2e}  Cramér's V={v:.2f}  -> {verdict}")
    lines.append("    Expected: driver⊥constructor fails -> selection confounding (pitfall #1).")
    return any_violation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsify-rows", type=int, default=500)
    ap.add_argument("--falsify-perms", type=int, default=20)
    args = ap.parse_args()

    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    scm, d = load()
    lines: list[str] = ["=" * 70, "PHASE (e) VALIDATION REPORT", "=" * 70,
                        f"classified rows: {len(d)}"]

    # --- 1. mechanism evaluation ---
    lines.append("\n[1] gcm.evaluate_causal_model (mechanisms reproduce the data?)")
    res = gcm.evaluate_causal_model(
        scm, d, evaluate_causal_mechanisms=True,
        evaluate_invertibility_assumptions=True, evaluate_overall_kl_divergence=True,
        evaluate_causal_structure=False)  # structure handled explicitly via falsify_graph
    lines.append(str(res))

    # --- 2. graph falsification (subsampled, parallel) ---
    lines.append("\n[2] falsify_graph (is the DAG consistent with the data?)")
    g = finish_scm_graph(build_dag())
    sub = d.sample(min(args.falsify_rows, len(d)), random_state=SEED)
    lines.append(f"    kernel-based, n={len(sub)} subsample, n_permutations={args.falsify_perms}")
    fres = falsify_graph(g, sub, n_permutations=args.falsify_perms,
                         n_jobs=-1, show_progress_bar=False)
    lines.append(str(fres))

    # --- 3. direct root-independence tests ---
    violated = root_independence(d, lines)

    # --- verdict ---
    lines.append("\n" + "=" * 70)
    lines.append("VERDICT")
    lines.append("=" * 70)
    lines.append(
        "Mechanism fits: see [1]. Structural assumption: the independent-roots\n"
        "assumption is " + ("VIOLATED" if violated else "supported") +
        " (see [3]) — as anticipated, driver and constructor are\n"
        "not independent. Consequence: intrinsic_causal_influence (phase f) is biased by\n"
        "fantasy driver×car pairings and must be reported with this caveat; lean on the\n"
        "observation-anchored counterfactual swaps, which are unaffected.")

    report = "\n".join(lines)
    print(report)
    (OUT / "validation_report.txt").write_text(report + "\n")
    print(f"\nWrote {(OUT / 'validation_report.txt').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
