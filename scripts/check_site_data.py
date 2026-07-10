"""Sanity-check the committed site data artifacts (site/public/data/*.json).

The site has no backend — these JSONs ARE the runtime (see site/DATA.md). They're regenerated
locally from gitignored models, so CI can't rebuild them; what it CAN do is catch a broken or
half-committed export before it deploys. Stdlib only, so the CI job needs no dependencies.

Usage: python scripts/check_site_data.py   (exit 0 = ok, 1 = problems; prints each failure)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "site" / "public" / "data"

problems: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if not cond:
        problems.append(msg)
    return cond


def load(name: str):
    path = DATA / f"{name}.json"
    if not check(path.exists(), f"{name}.json: missing"):
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        problems.append(f"{name}.json: invalid JSON ({e})")
        return None


def band_ok(b: dict) -> bool:
    return all(k in b for k in ("lo", "med", "hi")) and b["lo"] <= b["med"] <= b["hi"]


def main() -> int:
    names = load("names")
    if names is not None:
        check(bool(names.get("drivers")) and bool(names.get("constructors")),
              "names.json: empty drivers/constructors map")

    drivers = load("drivers")
    if drivers is not None and check(bool(drivers), "drivers.json: empty"):
        for did, d in drivers.items():
            if not (d.get("name") and d.get("seasons") and band_ok(d.get("career", {}))
                    and d["career"].get("draws")):
                problems.append(f"drivers.json[{did}]: missing name/seasons/career band/draws")
                break

    cars = load("cars")
    if cars is not None and check(bool(cars), "cars.json: empty"):
        for cid, c in cars.items():
            if not (c.get("name") and isinstance(c.get("year"), int) and band_ok(c) and c.get("draws")):
                problems.append(f"cars.json[{cid}]: missing name/year/band/draws")
                break

    mesh = load("finish_mesh")
    if mesh is not None:
        sk, pa, z = mesh.get("skill_axis", []), mesh.get("pace_axis", []), mesh.get("z", [])
        check(sk == sorted(sk) and len(sk) >= 2, "finish_mesh.json: skill_axis not ascending/too short")
        check(pa == sorted(pa) and len(pa) >= 2, "finish_mesh.json: pace_axis not ascending/too short")
        check(len(z) == len(sk) and all(len(row) == len(pa) for row in z),
              "finish_mesh.json: z shape != len(skill_axis) x len(pace_axis)")

    era = load("era")
    if era is not None and check(isinstance(era, list) and era, "era.json: empty"):
        keys = {"label", "start", "end", "carPct", "driverPct", "carSpread", "driverSpread"}
        for row in era:
            check(keys <= row.keys(), f"era.json[{row.get('label')}]: missing keys {keys - row.keys()}")

    xe = load("cross_era")
    if xe is not None:
        check(bool(xe.get("legends")) and bool(xe.get("cars")) and bool(xe.get("caveat")),
              "cross_era.json: missing legends/cars/caveat")
        check(all(l.get("rawDraws") and l.get("eraSd", 0) > 0 for l in xe.get("legends", {}).values()),
              "cross_era.json: legend missing rawDraws or non-positive eraSd")

    tm = load("teammates")
    if tm is not None and check(bool(tm.get("nodes")) and bool(tm.get("edges")),
                                "teammates.json: empty nodes/edges"):
        ids = {n["id"] for n in tm["nodes"]}
        check(all(e["a"] in ids and e["b"] in ids for e in tm["edges"]),
              "teammates.json: edge endpoint not in nodes")

    pn = load("necessity")
    if pn is not None:
        check(0 < pn.get("carPct", -1) <= 100 and 0 < pn.get("driverPct", -1) <= 100,
              "necessity.json: carPct/driverPct out of (0, 100]")
        check(pn.get("nPodiums", 0) > 0, "necessity.json: nPodiums not positive")
        check(len(pn.get("mostCarDependent", [])) == 3 and len(pn.get("mostDriverDependent", [])) == 3,
              "necessity.json: expected 3 most-car/driver-dependent entries")

    manifest = load("manifest")
    if manifest is not None:
        check({"generated", "nDraws", "eras", "mainModel"} <= manifest.keys(),
              "manifest.json: missing keys")
        if era is not None and isinstance(era, list):
            check([r["label"] for r in era] == manifest.get("eras"),
                  "manifest.json: eras list disagrees with era.json labels")

    for extra in ("incident_rates_2018_2025", "reliability_rates"):
        obj = load(extra)
        if obj is not None:
            check(bool(obj), f"{extra}.json: empty")

    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"ok — {len(list(DATA.glob('*.json')))} artifacts in {DATA} pass all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
