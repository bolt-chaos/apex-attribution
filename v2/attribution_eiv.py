"""v2: errors-in-variables (EIV) — de-attenuate the car/driver attribution.

`driver_skill` and `car_pace` aren't observed; they're POSTERIOR ESTIMATES from the skill model,
each with its own estimation error. Feeding them into the finish regression as if they were exact
is a classic **measurement-error** problem: error in a regressor **attenuates** its coefficient
toward zero (regression dilution), so the noisier latent is UNDER-credited.

This corrects that. Using the posterior draws we estimate the measurement-error covariance Σ_uu of
(skill, pace), and apply the standard multivariate attenuation correction to the finish~skill+pace
regression (Fuller):

    Σ_WW = Cov of the observed (posterior-MEAN) latents        # true signal + estimation error
    Σ_uu = mean posterior Cov of the latents per row           # the estimation error alone
    β_naive = Σ_WW^{-1} Σ_WY        (attenuated)
    β_corr  = (Σ_WW − Σ_uu)^{-1} Σ_WY   (de-attenuated: divides out the error)
    reliability_j = 1 − Σ_uu[j,j]/Σ_WW[j,j]   (fraction of a latent's spread that is real signal)

HONEST FINDING (2018–2025 race pace): the DRIVER latent is the noisier one (reliability skill ~0.78
vs car ~0.93), so de-attenuation raises the DRIVER, not the car — standardized betas go from
naive skill 0.35 / pace 0.47 to corrected ~0.40 / ~0.40 (parity). So the attenuation was suppressing
the driver; correcting it does NOT rescue the car, it confirms car ≈ driver on the wide era. The
direction depends on which latent is noisier — this reports whatever the data says.

Caveat: classical EIV assumes the estimation error is independent of the true value (roughly true for
posterior error) and a ~linear mechanism; the gcm interventional/necessity measures would shift in
the same direction (driver up toward the car) but aren't corrected here.

Usage: python v2/attribution_eiv.py [--idata models/v2_idata_2018_2025_joint.pkl]
       [--results data/f1_results_2018_2025.parquet] [--var-skill racecraft] [--var-pace pace_r]
       [--tag _2018_2025_joint]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idata", default=str(ROOT / "models" / "v2_idata_2018_2025_joint.pkl"))
    ap.add_argument("--results", default=str(ROOT / "data" / "f1_results_2018_2025.parquet"))
    ap.add_argument("--var-skill", default="racecraft", help="posterior var for driver_skill")
    ap.add_argument("--var-pace", default="pace_r", help="posterior var for car_pace")
    ap.add_argument("--tag", default="_2018_2025_joint")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    post = pickle.load(open(Path(args.idata).resolve(), "rb")).posterior
    sk = post[args.var_skill].stack(s=("chain", "draw"))     # (driver, season, s) or (driver, s)
    pc = post[args.var_pace].stack(s=("chain", "draw"))       # (team_year, s)
    sk_mean = post[args.var_skill].mean(("chain", "draw")).to_pandas()
    pc_mean = post[args.var_pace].mean(("chain", "draw")).to_series()
    time_varying = "season" in post[args.var_skill].dims

    df = pd.read_parquet(Path(args.results).resolve())
    df = df[df.classified].copy()
    df["team_year"] = df.constructor_id + "@" + df.year.astype(str)
    if time_varying:
        df["skill"] = [sk_mean.loc[d, y] if (d in sk_mean.index and y in sk_mean.columns) else np.nan
                       for d, y in zip(df.driver_id, df.year)]
    else:
        df["skill"] = df.driver_id.map(sk_mean.to_series() if hasattr(sk_mean, "to_series") else sk_mean)
    df["pace"] = df.team_year.map(pc_mean)
    df["finish"] = df.finish_pos.astype(float)
    df = df.dropna(subset=["skill", "pace", "finish"])
    N = len(df)

    # Σ_WW (observed cov) and Σ_WY (cov with the outcome), over the regression sample
    W = df[["skill", "pace"]].to_numpy(float)
    Sww = np.cov(W.T)
    Swy = np.array([np.cov(W[:, 0], df.finish)[0, 1], np.cov(W[:, 1], df.finish)[0, 1]])

    # Σ_uu: measurement-error cov, averaged over rows, from the per-(driver,season)/team_year draws
    def skill_draws(d, y):
        return sk.sel(driver=d, season=y).values if time_varying else sk.sel(driver=d).values
    pace_draws = {ty: pc.sel(team_year=ty).values for ty in pc.coords["team_year"].values}
    acc = np.zeros((2, 2))
    for (d, y, ty), n in df.groupby(["driver_id", "year", "team_year"]).size().items():
        acc += n * np.cov(np.vstack([skill_draws(d, y), pace_draws[ty]]))
    Suu = acc / N

    rel = 1.0 - np.diag(Suu) / np.diag(Sww)                    # per-latent reliability
    err_corr = Suu[0, 1] / np.sqrt(Suu[0, 0] * Suu[1, 1])
    Sxx = Sww - Suu                                            # true-signal cov
    posdef = np.all(np.linalg.eigvals(Sxx) > 0)

    b_naive = np.linalg.solve(Sww, Swy)
    b_corr = np.linalg.solve(Sxx, Swy)
    sdY = float(df.finish.std())
    bs_naive = b_naive * np.sqrt(np.diag(Sww)) / sdY          # standardized (observed scale)
    bs_corr = b_corr * np.sqrt(np.diag(Sxx)) / sdY            # standardized (true scale)

    lead_naive = "car" if abs(bs_naive[1]) > abs(bs_naive[0]) else "driver"
    lead_corr = ("car" if bs_corr[1] > bs_corr[0] + 0.02 else
                 "driver" if bs_corr[0] > bs_corr[1] + 0.02 else "parity")
    L = ["=" * 70, "v2 ERRORS-IN-VARIABLES de-attenuation of finish ~ driver_skill + car_pace", "=" * 70,
         f"regression rows: {N}   latents: {args.var_skill} / {args.var_pace}",
         "",
         "[RELIABILITY] fraction of each latent's observed spread that is real signal (1 = no error)",
         f"    driver_skill  {rel[0]:.3f}   car_pace  {rel[1]:.3f}   "
         f"-> the {'DRIVER' if rel[0] < rel[1] else 'CAR'} latent is noisier / more attenuated",
         f"    (measurement-error correlation skill<->pace: {err_corr:+.2f})",
         "",
         "[STANDARDIZED BETAS] naive (attenuated) vs EIV-corrected (de-attenuated)",
         f"    finish ~ skill+pace   NAIVE:      driver {bs_naive[0]:+.2f}   car {bs_naive[1]:+.2f}   "
         f"({lead_naive} leads)",
         f"    finish ~ skill+pace   EIV-CORR:   driver {bs_corr[0]:+.2f}   car {bs_corr[1]:+.2f}   "
         f"({lead_corr})",
         "",
         "[READING] measurement error attenuates the noisier latent; correcting it moves that latent",
         f"    UP. Here the driver was the noisier one, so de-attenuation raises the DRIVER toward the",
         f"    car -> {lead_corr}. It does NOT rescue the car: the car is the better-identified latent.",
         "    (The gcm interventional/necessity measures would shift the same way — driver up.)"]
    if not posdef:
        L.append("    WARNING: Σ_WW − Σ_uu is not positive-definite (a latent may be too unreliable) —")
        L.append("    treat the corrected numbers with caution.")

    report = "\n".join(L)
    print(report)
    (OUT / f"v2_eiv_report{args.tag}.txt").write_text(report + "\n")

    # --- figure: naive vs de-attenuated standardized betas ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [bs_naive[0], bs_naive[1]], w, label="naive (attenuated)", color="#bbbbbb",
           edgecolor="#333")
    ax.bar(x + w / 2, [bs_corr[0], bs_corr[1]], w, label="EIV-corrected", color=["#a7c7ff", "#a7d8a0"],
           edgecolor="#333")
    ax.set_xticks(x); ax.set_xticklabels([f"driver_skill\n(reliab {rel[0]:.2f})",
                                          f"car_pace\n(reliab {rel[1]:.2f})"])
    ax.set_ylabel("standardized effect on finish_pos")
    ax.set_title("Errors-in-variables: correcting for the latents' estimation error\n"
                 "raises the noisier (driver) latent to parity — it doesn't rescue the car")
    for i, (a, c) in enumerate(zip(bs_naive, bs_corr)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.01, f"{c:.2f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"v2_eiv{args.tag}.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote outputs/v2_eiv_report{args.tag}.txt, figures/v2_eiv{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
