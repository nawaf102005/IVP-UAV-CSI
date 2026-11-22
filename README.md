# IVP — Reproducibility package

Research code and archived experimental records for **Valuing Large-Scale CSI for Bidirectional UAV Admission: Allocation and Decision Costs**, submitted to *Physical Communication*.

This is a **curated copy** of the supplied code/data archive, not a fresh rerun. Original numerical results, experiment protocols, raw records and source code have been retained without modification. Historical manuscript drafts and internal editorial/provenance notes were excluded. The canonical manuscript and supplementary materials are provided separately.

## Requirements

Python 3.12, a C++17 compiler, and the Python packages pinned in `requirements.txt`. Some ray-tracing generation routines require optional dependencies listed in `requirements-ray.txt`; the archived channel inputs in `ray_data/`, `fresh_ray_data/`, and `dynamic_ray_data/` are included.

```bash
python -m pip install -r requirements.txt
python run_all.py --mode verify
python run_all.py --mode equations
python run_all.py --mode break-even --workers 4
```

For additional runs, inspect `python run_all.py --help`. Simulations may take considerable time; execution benchmarks will vary by platform. **Archived results should not be modified**; run commands write into separate `reproduced_*` directories. The legacy all-figures route may expect the older `manuscript/` working directory, which is intentionally excluded from this public package. The final paper figures are included as PDFs under `figures/` and related figure assets under `figures_r5/`.

## Contents

- `code/` — original numerical methods, simulation scripts, audits and regression tests. Files named `audit_*`, `test_*`, or `round*` are **functional verification software** and are intentionally preserved.
- `*_results/`, `results/` — immutable archived experiment outputs supporting the paper and supplementary studies, including paired-frame records and timing logs.
- `ray_data/`, `fresh_ray_data/`, `dynamic_ray_data/` — archived channel inputs.
- `figures/`, `figures_r5/` — retained scientific PDFs and figure metadata; redundant raster/vector exports from `figures_r5` were omitted.
- `run_all.py`, `requirements*.txt`, `SHA256SUMS.json` — execution entrypoint, dependencies and per-file manifest.

## GitHub setup

The archived `revision_results/revision_records.jsonl` is larger than GitHub's 50 MB warning threshold, so `.gitattributes` enables **Git LFS** for it. Install Git LFS and execute `git lfs install` **before** `git add .` and pushing to GitHub. GitHub's web upload is not recommended for this package. All other kept files are below 50 MB. Git LFS use may depend on your account's storage/bandwidth quota.

```bash
git init
git lfs install
git add .
git commit -m "Release IVP reproducibility package"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Replace `USERNAME/REPOSITORY` with your actual repository. Only publish results/data you are authorized to share. No licensing claim is made here; select a license separately if desired.

## Integrity and limits

`SHA256SUMS.json` contains SHA-256 hashes of every packaged scientific input/file (excluding the manifest itself and this README). The separately excluded editorial/provenance history is not required for the archived `--mode verify` path; however, experiment reruns, optional dependencies and full end-to-end reproducibility have not been certified as part of this repackaging. In particular, this package does not independently verify that every reported journal result matches the latest manuscript revision.
