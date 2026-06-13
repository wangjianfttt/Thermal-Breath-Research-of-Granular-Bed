# Nature Physics raw-data archive manifest

Purpose: define the raw and supporting files that should be deposited in a DOI-backed archive before final submission.

## Summary

- Required raw archive files: 2823
- Required raw archive size: 5.01 GB
- All matched raw/supporting files: 3854
- All matched size: 5.64 GB
- Machine-readable inventory: `source_data/nature_physics_raw_data_archive_inventory.csv`

## Required archive tiers

- archive_required / atom_dump: 2142 files, 3.39 GB
- archive_required / contact_local: 600 files, 1.36 GB
- archive_required / restart: 81 files, 0.25 GB
- supporting_or_optional / atom_dump: 471 files, 0.45 GB
- supporting_or_optional / input_deck: 481 files, 0.05 GB
- supporting_or_optional / restart: 79 files, 0.13 GB

## Recommended DOI package structure

1. `processed_source_data/`: copy `manuscript_prl/source_data/` and run-level summary CSV files used by the figures.
2. `analysis_scripts/`: copy manuscript plotting and mining scripts plus the top-level `scripts/` directory needed to rebuild processed tables.
3. `simulation_inputs/`: copy LIGGGHTS input decks, generated route inputs and material-property scripts.
4. `raw_contact_force_outputs/`: archive the required `.local` true-contact-force files from `runs/contact_force_probe/` and `runs/long_cycle_force_probe/`.
5. `restart_files/`: archive final and settled restart files for the run families used in the manuscript.
6. `manifest/`: include this manifest, the CSV inventory, code-version notes and a README explaining which files reproduce each figure.

## Boundary

The processed CSV source data are sufficient to reproduce manuscript and Extended Data plots. The raw `.local` contact-force files, restart files and input decks are required for independent reanalysis and simulation reruns. Raw files remain local until deposited in a public DOI-backed repository.

## Data availability wording after DOI assignment

Replace the placeholder DOI in the manuscript with: `Raw DEM restart files, contact-local outputs, processed source data and analysis scripts have been deposited at [repository] under DOI [DOI]. Processed source data for all main and Extended Data figures are also included with the manuscript package.`
