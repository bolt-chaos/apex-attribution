"""Export precomputed JSON artifacts for the static interactive website (see site/DATA.md).

The site is fully static: every interactive is either a lookup into downsampled posterior draws or a
smooth function of (driver_skill, car_pace). The one gcm-heavy quantity, E[finish], is a 2-D surface
over (skill, pace) — `exp_finish` sets both roots and marginalizes circuit_type — so we bake it onto
a mesh here and bilinearly interpolate in the browser. Nothing below needs Python at runtime.

Two scales are involved and MUST NOT be mixed:
  - the MAIN site (car-swap, career arcs, H2H) lives on the JOINT 2018-2026 model:
    driver_skill = racecraft, car_pace = pace_r  (matches build_scm_data.py --skill-source race).
  - CROSS-ERA ("Senna in a modern Red Bull") lives on the 1988-2025 sess_rw model's single `skill`
    latent, with its own SCM fit on modern rows and its own mesh (cross_era.py's scale).

The main era's final season may be INCOMPLETE (2026 is half a season at the time of writing).
`partial_season_info()` reads the actual round counts out of f1db and ships them in the manifest so
the UI can caveat 2026 numbers; it self-corrects once the season finishes — nothing is hardcoded.

Models/data are gitignored (~GBs); this script runs LOCALLY and its small JSON outputs are committed
to site/public/data/. CI only builds the Vite app. Re-run this whenever a model changes.

Usage: .venv/bin/python scripts/export_site.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sqlite3
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from attribution_v2 import build_scm, exp_finish, NODES  # noqa: E402
import dowhy.gcm as gcm  # noqa: E402

DB = ROOT / "data" / "f1db.sqlite"
MODELS = ROOT / "models"
DATA = ROOT / "data"
OUT = ROOT / "site" / "public" / "data"
SEED = 20260708
ICC_FITS = 5           # seeded SCM fits averaged per era for era.json (see export_eras)
MESH_FITS = 10         # seeded SCM fits averaged per E[finish] mesh (see fit_mesh)
N_DRAWS = 200          # downsampled posterior draws shipped per element
MESH_N = 21            # grid points per axis for the E[finish] surface
EXP_N = 2000           # gcm samples per exp_finish evaluation (era spreads)
MESH_EXP_N = 8000      # more samples per mesh cell -> a smoother surface for the flagship car-swap
MAIN_START, MAIN_END = 2018, 2026          # the era the main site runs on
MAIN_TAG = f"{MAIN_START}_{MAIN_END}"
# The "modern field" reference window for CROSS-ERA only. Deliberately NOT tied to MAIN_END: it
# indexes the 1988-2025 sess_rw model, which stops at 2025, and it is the yardstick a legend's
# era-normalized skill is mapped onto — so it must stay a complete-season window even after the
# main era moves on. Widening it to a half-raced season would skew the modern spread.
MODERN = range(2018, 2026)

# cross-era legends -> the peak seasons their skill is averaged over (mirrors v2/cross_era.py)
PEAK = {"ayrton-senna": range(1988, 1995), "alain-prost": range(1988, 1994),
        "michael-schumacher": range(1994, 2005), "mika-hakkinen": range(1998, 2002),
        "nigel-mansell": range(1988, 1993), "damon-hill": range(1994, 1997)}

# era windows for the slider: (label, start, end, scm_parquet). Each uses its own era-specific fit,
# so skill/pace scales differ across windows — the shares (%) and position spreads stay comparable.
ERAS = [
    ("2018-2026", 2018, 2026, "f1_scm_v2_2018_2026_joint.parquet"),
    ("2018-2025", 2018, 2025, "f1_scm_v2_2018_2025_joint.parquet"),
    ("2006-2025", 2006, 2025, "f1_scm_v2_2006_2025_rw.parquet"),
    ("1988-2025", 1988, 2025, "f1_scm_v2_1988_2025_sess_rw.parquet"),
]

# The rung-3 necessity result for the masthead hook ("would this podium have happened BUT FOR the
# car / the driver?"), read from attribution_v2.py's machine-readable artifact — the same joint
# 2018-2026 model the main site runs on. Regenerate the artifact with:
#   python v2/attribution_v2.py --data data/f1_scm_v2_2018_2026_joint.parquet --tag _2018_2026_joint
ATTRIBUTION_JSON = ROOT / "outputs" / f"v2_attribution_{MAIN_TAG}_joint.json"
NECESSITY_ERA = f"{MAIN_START}–{MAIN_END}"


def partial_season_info() -> dict | None:
    """Is the main era's final season still in progress? Reads real round counts from f1db.

    Returns {year, roundsComplete, roundsTotal} when the season is incomplete, else None — so the
    UI caveat appears automatically while a season is half-run and disappears once it finishes.
    """
    try:
        con = sqlite3.connect(DB)
        total = con.execute("SELECT COUNT(*) FROM race WHERE year=?", (MAIN_END,)).fetchone()[0]
        done = con.execute(
            "SELECT COUNT(*) FROM race r WHERE r.year=? AND EXISTS "
            "(SELECT 1 FROM race_result rr WHERE rr.race_id=r.id)", (MAIN_END,)).fetchone()[0]
        con.close()
    except Exception:
        return None
    if not total or done >= total:
        return None
    return {"year": MAIN_END, "roundsComplete": int(done), "roundsTotal": int(total)}


def export_necessity(names: dict) -> dict:
    art = json.loads(ATTRIBUTION_JSON.read_text())
    pn = art["necessity"]
    per = pn["perDriver"]  # insertion order = alphabetical driver id; sorted() below is stable

    def top3(key: str) -> list:
        best = sorted(per.items(), key=lambda kv: -kv[1][key])[:3]
        return [{"name": names["drivers"].get(d) or titlecase(d), "pct": round(100 * v[key])}
                for d, v in best]

    return {"era": NECESSITY_ERA, "threshold": pn["threshold"], "nPodiums": pn["nPodiums"],
            "carPct": round(100 * pn["pnCar"]), "driverPct": round(100 * pn["pnDriver"]),
            "mostCarDependent": top3("pnCar"), "mostDriverDependent": top3("pnDriver")}


def load_names() -> dict:
    """id -> display name for drivers and constructors, from the f1db SQLite."""
    out = {"drivers": {}, "constructors": {}}
    try:
        con = sqlite3.connect(DB)
        out["drivers"] = dict(con.execute("SELECT id, name FROM driver").fetchall())
        out["constructors"] = dict(con.execute("SELECT id, name FROM constructor").fetchall())
        con.close()
    except Exception as e:  # pragma: no cover - fallback path
        print(f"  (names: SQLite unavailable, will titlecase ids: {e})")
    return out


def titlecase(idv: str) -> str:
    return " ".join(w.capitalize() for w in idv.replace("@", " ").split("-"))


def section_seed(key: str) -> np.random.Generator:
    """Re-seed gcm (numpy global state) and return a fresh Generator, both derived from SEED + key.

    Each export section / era fit gets its own stream, so a section's output depends only on its own
    inputs — adding a season to the main era no longer shifts the random stream that later sections
    (other eras, cross-era) consume, and an unchanged input re-exports byte-identically.
    """
    s = (SEED + zlib.crc32(key.encode())) % 2**32
    gcm.util.general.set_random_seed(s)
    return np.random.default_rng(s)


def draws_summary(arr: np.ndarray, rng: np.random.Generator) -> dict:
    """5/50/95 quantiles + N_DRAWS downsampled draws, rounded to keep JSON small."""
    arr = np.asarray(arr, float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.5, 0.95))
    idx = rng.choice(arr.size, size=min(N_DRAWS, arr.size), replace=False)
    return {"lo": round(lo, 4), "med": round(mid, 4), "hi": round(hi, 4),
            "draws": [round(float(v), 4) for v in arr[idx]]}


def fit_mesh(data: pd.DataFrame, key: str, skill_axis: list, pace_axis: list) -> list:
    """E[finish] on the (skill x pace) grid, averaged over MESH_FITS seeded SCM fits.

    Rows index skill, cols index pace. A single fit is unstable: auto-assignment's choice of the
    finish_pos mechanism (linear vs gradient-boosted, a near-tie in its CV) flips with the seed and
    moves cells by ~1.8 positions on average (max ~5.6). Averaging the fits marginalizes over that
    choice: at 10 fits two independent seed sets agree to ~0.4 positions on average (max ~1.1). The
    MESH_EXP_N sample budget per cell is split across fits, so only the extra fitting adds runtime.
    """
    n = MESH_EXP_N // MESH_FITS
    z = np.zeros((len(skill_axis), len(pace_axis)))
    for k in range(MESH_FITS):
        section_seed(f"mesh:{key}:{k}")
        scm = build_scm(data)
        z += [[exp_finish(scm, float(s), float(p), n) for p in pace_axis] for s in skill_axis]
    return np.round(z / MESH_FITS, 3).tolist()


def axis(vals: pd.Series, n: int = MESH_N, pad: float = 0.05) -> list:
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    return [round(v, 4) for v in np.linspace(lo - pad * span, hi + pad * span, n)]


# ---------------------------------------------------------------------------------------------------

def export_main(names: dict, rng: np.random.Generator) -> tuple[dict, dict, dict]:
    """drivers.json, cars.json, finish_mesh.json on the JOINT main-era (racecraft / pace_r) scale."""
    post = pickle.load(open(MODELS / f"v2_idata_{MAIN_TAG}_joint.pkl", "rb")).posterior
    racecraft = post["racecraft"]                          # (chain, draw, driver, season)
    pace_r = post["pace_r"]                                # (chain, draw, team_year)
    rc = racecraft.stack(s=("chain", "draw"))              # (driver, season, s)
    pc = pace_r.stack(s=("chain", "draw"))                 # (team_year, s)

    results = pd.read_parquet(DATA / f"f1_results_{MAIN_TAG}.parquet")
    active = {d: sorted(int(y) for y in g.year.unique())
              for d, g in results.groupby("driver_id")}
    dn = names["drivers"]

    # --- drivers.json: per-season mean/band trajectory + career-pooled draws (for car-swap + H2H) ---
    drivers = {}
    for d in rc.coords["driver"].values:
        d = str(d)
        seasons = [y for y in active.get(d, []) if y in rc.coords["season"].values]
        if not seasons:
            continue
        by_season = {}
        pooled = []
        for y in seasons:
            dr = rc.sel(driver=d, season=y).values
            pooled.append(dr)
            by_season[str(y)] = {k: draws_summary(dr, rng)[k] for k in ("lo", "med", "hi")}
        drivers[d] = {"name": dn.get(d, titlecase(d)), "seasons": seasons,
                      "bySeason": by_season, "career": draws_summary(np.concatenate(pooled), rng)}

    # --- cars.json: per team_year pace draws ---
    cn = names["constructors"]
    cars = {}
    for ty in pc.coords["team_year"].values:
        ty = str(ty)
        cons, yr = ty.split("@")
        cars[ty] = {"constructor": cons, "year": int(yr),
                    "name": f"{cn.get(cons, titlecase(cons))} {yr}",
                    **draws_summary(pc.sel(team_year=ty).values, rng)}

    # --- finish_mesh.json: E[finish] surface on the joint scale ---
    df = pd.read_parquet(DATA / f"f1_scm_v2_{MAIN_TAG}_joint.parquet")
    df = df[df.classified].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        df[c] = df[c].astype(float)
    df["circuit_type"] = df["circuit_type"].astype("object")
    skill_axis, pace_axis = axis(df.driver_skill), axis(df.car_pace)
    mesh = {"skill_axis": skill_axis, "pace_axis": pace_axis,
            "z": fit_mesh(df[NODES].dropna(), "main", skill_axis, pace_axis),
            "nFits": MESH_FITS,
            "note": "E[finish] marginalized over circuit_type; lower skill/pace = faster."}
    print(f"  drivers={len(drivers)} cars={len(cars)} mesh={MESH_N}x{MESH_N}")
    return drivers, cars, mesh


def export_eras() -> list:
    """era.json: ICC shares + interventional car/driver position spreads per era window.

    The ICC share is fit-sensitive: auto-assignment's cross-validation scores finish_pos's
    LinearRegression and HistGradientBoosting as a near-tie, so the chosen mechanism flips with the
    seed and moves the car share by several points (2018-2025: 25-35% over six seeds). A single fit
    therefore re-rolls that coin on every export. Instead each era averages ICC_FITS independently
    seeded fits (per-era seeds, so windows don't perturb each other) and ships the between-fit SD.
    """
    from attribution_v2 import icc_car_driver, sweep_spreads
    rows = []
    for label, start, end, fname in ERAS:
        path = DATA / fname
        if not path.exists():
            print(f"  era {label}: {fname} missing, skipped"); continue
        df = pd.read_parquet(path)
        df = df[df.classified & df.year.between(start, end)].copy()
        for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
            df[c] = df[c].astype(float)
        df["circuit_type"] = df["circuit_type"].astype("object")
        d = df[NODES].dropna()
        mid_s, mid_p = float(d.driver_skill.median()), float(d.car_pace.median())
        pace_vals = np.linspace(d.car_pace.min(), d.car_pace.max(), 12)
        skill_vals = np.linspace(d.driver_skill.min(), d.driver_skill.max(), 12)
        fits = []                                          # (car%, driver%, car spread, drv spread)
        for k in range(ICC_FITS):
            section_seed(f"era:{label}:{k}")
            scm = build_scm(d, hiring_edge=False)          # independent roots for the ICC headline
            car_pct, drv_pct, _, _ = icc_car_driver(scm, rand=150, base=500)
            fits.append((100 * car_pct, 100 * drv_pct,
                         *sweep_spreads(scm, pace_vals, skill_vals, mid_s, mid_p, EXP_N)))
        mean, sd = np.mean(fits, axis=0), np.std(fits, axis=0, ddof=1)
        rows.append({"label": label, "start": start, "end": end,
                     "carPct": round(mean[0], 1), "driverPct": round(mean[1], 1),
                     "carSpread": round(mean[2], 2), "driverSpread": round(mean[3], 2),
                     "carPctSd": round(sd[0], 1), "driverPctSd": round(sd[1], 1), "nFits": ICC_FITS})
        print(f"  era {label}: car {mean[0]:.1f}% (sd {sd[0]:.1f}) / driver {mean[1]:.1f}% "
              f"(sd {sd[1]:.1f}) over {ICC_FITS} fits  spread car {mean[2]:.1f} / drv {mean[3]:.1f}")
    return rows


def export_cross_era(names: dict, rng: np.random.Generator) -> dict:
    """cross_era.json: legend draws + era/modern field spreads + a dedicated 1988-model-scale mesh."""
    full = pd.read_parquet(DATA / "f1_scm_v2_1988_2025_sess_rw.parquet")
    mod = full[full.classified & full.year.isin(MODERN)].copy()
    for c in ["grid", "finish_pos", "driver_skill", "car_pace"]:
        mod[c] = mod[c].astype(float)
    mod["circuit_type"] = mod["circuit_type"].astype("object")
    skill_axis, pace_axis = axis(mod.driver_skill), axis(mod.car_pace)
    mesh = {"skill_axis": skill_axis, "pace_axis": pace_axis,
            "z": fit_mesh(mod[NODES].dropna(), "cross_era", skill_axis, pace_axis),
            "nFits": MESH_FITS}

    post = pickle.load(open(MODELS / "v2_idata_1988_2025_sess_rw.pkl", "rb")).posterior
    sd = post["skill"].stack(s=("chain", "draw"))          # (driver, season, s)
    smean = post["skill"].mean(("chain", "draw")).to_pandas()
    qd = pd.read_parquet(DATA / "f1_quali_1988_2025_sess.parquet")
    activ = {int(y): list(g.driver_id.unique()) for y, g in qd.groupby("year")}

    def spread(years):
        ms, ss = [], []
        for y in years:
            vals = [smean.loc[dd, y] for dd in activ.get(y, [])
                    if dd in smean.index and y in smean.columns]
            if len(vals) > 3:
                ms.append(np.mean(vals)); ss.append(np.std(vals))
        return float(np.mean(ms)), float(np.mean(ss))

    mod_mean, mod_sd = spread([y for y in MODERN])
    dn = names["drivers"]
    legends = {}
    for lg, peak in PEAK.items():
        if lg not in smean.index:
            continue
        yrs = [y for y in peak if y in smean.columns]
        draws = np.concatenate([sd.sel(driver=lg, season=y).values for y in yrs])
        era_mean, era_sd = spread(yrs)
        idx = rng.choice(draws.size, size=min(N_DRAWS, draws.size), replace=False)
        legends[lg] = {"name": dn.get(lg, titlecase(lg)),
                       "peak": [int(yrs[0]), int(yrs[-1])],
                       "eraMean": round(era_mean, 4), "eraSd": round(era_sd, 4),
                       "rawDraws": [round(float(v), 4) for v in draws[idx]]}

    # modern target cars (this scale) + a Verstappen-2024 reference for the plot line
    cn = names["constructors"]
    cars = {}
    for ty, g in full[full.year.isin(MODERN)].groupby("team_year"):
        cons, yr = ty.split("@")
        cars[ty] = {"name": f"{cn.get(cons, titlecase(cons))} {yr}",
                    "pace": round(float(g.car_pace.iloc[0]), 4)}
    vers = round(float(smean.loc["max-verstappen", 2024]), 4) if "max-verstappen" in smean.index else None
    print(f"  cross-era: {len(legends)} legends, {len(cars)} modern cars, "
          f"modern field mean {mod_mean:+.2f}/sd {mod_sd:.2f}")
    return {"mesh": mesh, "modMean": round(mod_mean, 4), "modSd": round(mod_sd, 4),
            "legends": legends, "cars": cars, "verstappen2024": vers,
            "caveat": "Illustrative extrapolation under an era-scale (z-score) assumption — not an "
                      "identified effect. Source model max R-hat 1.04; intervals are wide on purpose."}


def export_teammates(names: dict) -> dict:
    """teammates.json: driver nodes + shared-team_year edges over full history (for the pathfinder)."""
    res = pd.read_parquet(DATA / "f1_results_1988_2025.parquet")
    dn = names["drivers"]
    edges = {}
    node_ids = set()
    for ty, g in res.groupby(res.constructor_id + "@" + res.year.astype(str)):
        drv = sorted(g.driver_id.unique())
        node_ids.update(drv)
        for i in range(len(drv)):
            for j in range(i + 1, len(drv)):
                key = (drv[i], drv[j])
                edges.setdefault(key, []).append(ty)
    nodes = [{"id": d, "name": dn.get(d, titlecase(d))} for d in sorted(node_ids)]
    edge_list = [{"a": a, "b": b, "teams": sorted(set(t))} for (a, b), t in edges.items()]
    print(f"  teammates: {len(nodes)} drivers, {len(edge_list)} shared-team edges")
    return {"nodes": nodes, "edges": edge_list}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def write(name, obj):
        (out / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")))
        kb = (out / f"{name}.json").stat().st_size / 1024
        print(f"  wrote {name}.json ({kb:.0f} KB)")

    print("names…");       names = load_names();                 write("names", names)
    print("main model…");  drivers, cars, mesh = export_main(names, section_seed("main"))
    write("drivers", drivers); write("cars", cars); write("finish_mesh", mesh)
    print("eras…");        write("era", export_eras())
    print("cross-era…");   write("cross_era", export_cross_era(names, section_seed("cross_era")))
    print("teammates…");   write("teammates", export_teammates(names))
    print("necessity…");   write("necessity", export_necessity(names))

    for src in [f"incident_rates_{MAIN_TAG}.json", "reliability_rates.json"]:
        shutil.copy(MODELS / src, out / src)
        print(f"  copied {src}")

    partial = partial_season_info()
    if partial:
        print(f"  partial season: {partial['year']} is "
              f"{partial['roundsComplete']}/{partial['roundsTotal']} rounds -> UI caveat ON")

    write("manifest", {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nDraws": N_DRAWS, "meshN": MESH_N, "expN": EXP_N,
        "mainModel": f"v2_idata_{MAIN_TAG}_joint (racecraft / pace_r)",
        "crossEraModel": "v2_idata_1988_2025_sess_rw (skill)",
        "eras": [e[0] for e in ERAS],
        "partialSeason": partial,
        "meshRanges": {"skill": [mesh["skill_axis"][0], mesh["skill_axis"][-1]],
                       "pace": [mesh["pace_axis"][0], mesh["pace_axis"][-1]]},
    })
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
