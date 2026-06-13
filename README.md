# Thermal Breath Research of Granular Bed

Code and processed data for the study:

**Memory-induced thermal breathing of a granular force network**

This repository is intended as the public code/data companion repository. It does not contain the manuscript text, cover letter, editorial correspondence or old draft files.

## Repository Contents

- `scripts/`: Python scripts used for figure generation, source-data audits, mechanism mining and archive bookkeeping.
- `source_data/`: processed CSV/JSON source data underlying the main figures, Extended Data figures and numerical-claim audits.
- `figures/`: lightweight generated figure outputs for reference, in PDF/SVG/PNG formats. Large TIFF production files are not included.
- `archive_deposition/`: raw-data deposition plan, DataCite metadata draft and SHA256 checksum manifest for required raw DEM files.
- `docs/`: source-data inventory, raw-data archive manifest and data-availability notes.

## What Is Not Included

- Manuscript PDF/TeX files.
- Cover letters, presubmission enquiries and editorial materials.
- Exploratory notes and old PRL draft files.
- Raw DEM atom dumps, restart files and contact-local pair-force outputs.
- Large TIFF production figures.

The raw DEM archive contains 2,823 required files and is approximately 5.01 GB locally. It should be deposited separately in a DOI-backed repository before formal journal submission. The checksum manifest is provided in `archive_deposition/required_raw_file_sha256.csv`.

## Quick Start

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Examples:

```bash
python scripts/audit_nature_physics_numerical_claims.py
python scripts/audit_nature_physics_submission_assets.py
python scripts/build_nphys_fig1_reservoir.py
python scripts/mine_dimensionless_loop_collapse.py
```

Most scripts expect to be run from the repository root and write outputs to `source_data/` and/or `figures_selected/` when adapted for public use. Some mining scripts refer to raw DEM files that are not included in this repository; those require the DOI-backed raw archive.

## Data Boundary

The processed source data are sufficient to inspect plotted quantities, numerical claims and audit tables. Full reruns from raw DEM output require the separate raw archive described in `docs/nature_physics_raw_data_archive_manifest.md`.

## Corresponding Author

Jian Wang, wjfttt@mail.ustc.edu.cn
