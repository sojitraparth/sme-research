# SME Research Lab — pandas vs Polars vs DuckDB (single-node analytics)

This repo contains the code, data artefacts, and figures for my study on
**embedded analytics engines** for SMEs: pandas, Polars, and DuckDB.
It includes: a benchmark runner, an API, a Streamlit UI, and the camera-ready
figures/tables used in the report.

> **TL;DR**  
> Single-table transforms: **Polars** is consistently faster with lower memory.  
> Join-heavy analytics: **DuckDB** is decisively faster and scales better.  
> pandas remains the baseline; Arrow I/O/dtypes narrow ingest gaps.

---

## Repo layout

api/            # FastAPI service (upload, run-benchmark, results)
app/            # Streamlit UI
labbench/       # Engine-neutral benchmark kernels (ingest/filter/groupby/sort/join)
analysis/       # Analysis helpers / scripts
results/        # Raw outputs (CSV)
figures/        # Exported SVGs for the paper
tables/         # Exported CSV tables for the paper

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start back end (port 8000)
uvicorn api.main:app --reload --port 8000

# Start UI
streamlit run app/app.py

	1.	Upload a CSV/Parquet file.
	2.	Pick engines & tasks; set repetitions (≥10 for headline plots).
	3.	Results land in results/benchmarks.csv and results/latency.csv.

Headless run (example): see analysis/notebook.py for a scripted run that
reads repro.yml, executes tasks, and exports all figures/tables into
figures/ and tables/.

⸻

Reproduce my paper figures

All camera-ready assets live in figures/ and tables/ and are listed in
Appendix E. The key Chapter-4 filenames:
	•	Coverage: fig_4_coverage_heatmap.svg
	•	Ingest/filter/groupby/sort/join (box, ECDF, pooled, speedup): see fig_4_*
	•	Scaling: fig_4_ingest_scaling.svg, fig_4_groupby_scaling.svg, fig_4_join_scaling.svg
	•	Tables 4-0…4-6: see tables/table_4_*.csv

⸻

API endpoints
	•	POST /upload → {dataset_id, path}
	•	POST /run-benchmark → runs selected engines/tasks
	•	GET /results → last N rows from results/benchmarks.csv
	•	GET /versions → engine versions

⸻

How to cite

See CITATION.cff, or:

Sojitra, P. (2025). SME Research Lab: pandas vs Polars vs DuckDB for single-node analytics. GitHub. https://github.com/sojitraparth/sme-research

⸻

License

Code: MIT. Data: see dataset-specific licenses.
pandas (BSD-3), Polars (MIT), DuckDB (MIT), Apache Arrow (Apache-2.0).

---

# Drop-in `LICENSE` (MIT)

```text
MIT License

Copyright (c) 2025 Parth Sojitra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[...standard MIT text continues...]

(Use the full MIT text; I truncated here for space — copy the full standard block.)

⸻

Drop-in CITATION.cff

cff-version: 1.2.0
message: If you use this software, please cite it as below.
title: SME Research Lab: pandas vs Polars vs DuckDB for single-node analytics
authors:
  - family-names: Sojitra
    given-names: Parth
date-released: 2025-08-29
version: 1.0.0
repository-code: https://github.com/sojitraparth/sme-research
license: MIT
abstract: >
  Benchmark harness, API, and UI to compare pandas, Polars, and DuckDB on
  SME-typical workloads (ingest, filter, group-by, sort, join) with
  time/ΔRSS/latency instrumentation. Includes camera-ready figures/tables.


⸻

requirements.txt (pin what you used)

fastapi==0.111.0
uvicorn[standard]==0.30.0
pandas==2.2.2
polars==1.5.0
duckdb==1.0.0
pyarrow==16.1.0
psutil==5.9.8
streamlit==1.36.0
plotly==5.22.0
numpy==1.26.4


⸻

GitHub Actions CI (.github/workflows/ci.yml)

name: ci
on:
  push: { branches: [ main ] }
  pull_request: { branches: [ main ] }

jobs:
  lint-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [ "3.10", "3.11" ]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - run: python -m pip install -U pip
      - run: pip install -r requirements.txt black ruff
      - name: Lint (ruff)
        run: ruff check .
      - name: Format check (black)
        run: black --check .
      - name: Smoke import
        run: python - <<'PY'
import importlib; importlib.import_module("labbench.runner"); importlib.import_module("api.main"); print("ok")
PY

(This avoids heavy tests but protects against broken imports.)

⸻

Release & DOI (recommended)
	1.	Tag & Release

git tag -a v1.0.0 -m "camera-ready"
git push origin v1.0.0

On GitHub → Releases → Create release and upload the figures/ and tables/ zips.

	2.	Zenodo DOI
	•	Connect the repo to Zenodo (one click), create a DOI for the v1.0.0 release.
	•	Add the DOI badge to the top of your README.md.

Badge snippet:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)


⸻

GitHub Pages (optional but slick)
	•	Create docs/index.md with a short landing page and figure thumbnails.
	•	In repository settings → Pages → docs/ → deploy from main.

Minimal docs/index.md:

# SME Research Lab (paper artefacts)

This site hosts figures and tables for the study.
