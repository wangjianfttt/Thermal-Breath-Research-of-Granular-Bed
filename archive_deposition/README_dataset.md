# Dataset README

## Title

Memory-induced thermal breathing of a granular force network: DEM source data, raw contact-network outputs and analysis scripts

## Summary

This dataset supports the manuscript `Memory-induced thermal breathing of a granular force network`. It contains processed source data for the main and Extended Data figures, analysis and plotting scripts, LIGGGHTS input/provenance materials, and a raw-data inventory for atom dumps, restart files and contact-local pair-force outputs used to construct force-network and breathing-map diagnostics.

The central raw archive contains files needed for independent reanalysis of the DEM contact networks. Processed CSV files are sufficient to regenerate the manuscript plots, whereas the raw `.local`, dump and restart files support deeper inspection and reruns.

## Required raw archive scale

- Required raw files: 2823
- Required raw size: 5.01 GB

## File groups

- `archive_required/atom_dump`: 2142 files, 3.39 GB
- `archive_required/contact_local`: 600 files, 1.36 GB
- `archive_required/restart`: 81 files, 0.25 GB
- `supporting_or_optional/atom_dump`: 471 files, 0.45 GB
- `supporting_or_optional/input_deck`: 481 files, 0.05 GB
- `supporting_or_optional/restart`: 79 files, 0.13 GB

## Recommended repository layout

- `processed_source_data/`: all CSV/JSON files from `manuscript_prl/source_data/` used for figures, statistics and audits.
- `analysis_scripts/`: manuscript plotting scripts, mining scripts and audit scripts.
- `simulation_inputs/`: LIGGGHTS input decks and generated run inputs.
- `raw_contact_force_outputs/`: contact-local pair-force `.local` files used for true-force network analyses.
- `restart_files/`: settled, final and preloaded restart files used for reruns.
- `raw_atom_dumps/`: atom dump files used for microstructure, topology and phase-state reconstruction.
- `manifest/`: archive inventories, this README, DataCite metadata and figure-to-source-data maps.

## Variables and units

Processed source-data tables retain the column names used by the analysis scripts. Physical wall loads are simulation pressure proxies in Pa. Contact forces are DEM pair-force magnitudes in the LIGGGHTS unit convention used by the input decks. Dimensionless quantities include normalized force shares, route-centred coordinates, Spearman statistics, prediction scores and the loop number `Psi` defined in the manuscript.

## Methods and provenance

DEM calculations used LIGGGHTS-style input decks to cycle Li2TiO3-like spherical particles between cold and hot states. Post-processing scripts reconstruct contact networks, force-filtration variables, route-conditioned return maps, breathing parameters and figure source data. The raw-data inventory records relative paths, file groups, required/optional tiers and file sizes.

## Access and licence

The code/data companion repository is archived on Zenodo at https://doi.org/10.5281/zenodo.20674801. The lightweight public snapshot contains processed source data, scripts, selected figure outputs and raw-file checksum metadata. The full raw DEM dump, restart and contact-local pair-force files are represented by checksum manifests and should be deposited separately if full raw-output reanalysis is required.

## Preferred citation

Wang, J. (2026). Memory-induced thermal breathing of a granular force network: code and processed data repository. Zenodo. https://doi.org/10.5281/zenodo.20674801.
