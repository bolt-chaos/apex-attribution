"""Download and cache the f1db SQLite database.

f1db: https://github.com/f1db/f1db (CC-BY-4.0).
We pull the prebuilt SQLite artifact from the GitHub release and cache it under data/.
Idempotent: re-running is a no-op unless --force or the version changed.

Usage:
    python scripts/download_data.py            # latest release
    python scripts/download_data.py --tag v2026.7.0
    python scripts/download_data.py --force
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSET_NAME = "f1db-sqlite.zip"  # prebuilt .db file (not the SQL-dump variants)
DB_PATH = DATA_DIR / "f1db.sqlite"
VERSION_PATH = DATA_DIR / "f1db.version"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "apex-attribution"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def resolve_release(tag: str | None) -> dict:
    if tag:
        url = f"https://api.github.com/repos/f1db/f1db/releases/tags/{tag}"
    else:
        url = "https://api.github.com/repos/f1db/f1db/releases/latest"
    return _get_json(url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="release tag, e.g. v2026.7.0 (default: latest)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    release = resolve_release(args.tag)
    tag = release["tag_name"]

    cached_version = VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else None
    if DB_PATH.exists() and cached_version == tag and not args.force:
        print(f"Cached f1db {tag} already present at {DB_PATH} (use --force to refresh).")
        return 0

    asset = next((a for a in release["assets"] if a["name"] == ASSET_NAME), None)
    if asset is None:
        print(f"Asset {ASSET_NAME!r} not found in release {tag}.", file=sys.stderr)
        print("Available:", [a["name"] for a in release["assets"]], file=sys.stderr)
        return 1

    url = asset["browser_download_url"]
    print(f"Downloading f1db {tag} :: {ASSET_NAME} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "apex-attribution"})
    with urllib.request.urlopen(req) as resp:
        blob = resp.read()
    print(f"  got {len(blob) / 1e6:.1f} MB, extracting ...")

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        db_members = [n for n in zf.namelist() if n.endswith((".db", ".sqlite", ".sqlite3"))]
        if not db_members:
            print(f"No .db/.sqlite file inside zip; contents: {zf.namelist()}", file=sys.stderr)
            return 1
        member = db_members[0]
        DB_PATH.write_bytes(zf.read(member))

    VERSION_PATH.write_text(tag + "\n")
    print(f"Wrote {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB), version {tag}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
