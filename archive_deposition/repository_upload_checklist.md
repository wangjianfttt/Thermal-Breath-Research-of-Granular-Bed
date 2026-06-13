# Repository upload checklist

## Before upload

- Choose repository: Zenodo, Figshare, OSF, Dryad or institutional repository with DOI support.
- Confirm author names, affiliations, ORCID identifiers and funder metadata.
- Choose a data licence compatible with institutional rules.
- Decide whether to upload one archive or several records for processed data, raw force outputs and simulation restarts.
- Keep raw and processed data separate in the uploaded file tree.

## Files to include

- Processed source data and source-data inventory.
- Main and Extended Data figure scripts.
- Mining/audit scripts used for force-loop, return-map, irreversibility and onset analyses.
- LIGGGHTS input decks and executable provenance notes.
- Required raw DEM files: 2823 files, 5.01 GB.
- This README, DataCite metadata draft, raw archive inventory and figure-to-source-data map.

## After upload

- Verify DOI resolves outside the author account.
- Download at least one raw archive and one processed source-data file from the public/private-review link.
- Replace all `[DOI]` placeholders in the manuscript, cover letter, Data Availability statement and source-data inventory.
- Re-run `audit_nature_physics_submission_assets.py` after DOI wording is inserted.
