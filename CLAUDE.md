# CLAUDE.md

Guidance for agents working in this repo.

## Workflow

- **Use pull requests to land changes — do not commit directly to `main`.** For each unit of
  work: branch off `main`, commit there, push, and open a PR with `gh pr create`. Wait for the
  PR to be merged before syncing `main` and starting the next branch.

## Project orientation

Causal driver-vs-car attribution for F1 (DoWhy `gcm` over the f1db dataset), built in phases
(a)→(f). See `README.md` for status and `docs/SCHEMA_NOTES.md` for the data schema, the DNF taxonomy,
the connectivity finding, and the verified `gcm`-0.14 API notes.

- Reproduce data/models with `scripts/download_data.py` → `build_dataset.py` → `fit_model.py`.
- Use the `.venv` (Python 3.12); deps are pinned in `requirements.txt`.
