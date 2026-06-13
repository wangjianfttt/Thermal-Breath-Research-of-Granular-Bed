#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
OUT = ROOT / "archive_staging"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def inventory_stats(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], list[int]], int, int]:
    stats: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        stats[(row["tier"], row["group"])].append(int(row["bytes"]))
    required = [r for r in rows if r["tier"] == "archive_required"]
    return stats, len(required), sum(int(r["bytes"]) for r in required)


def build_readme(raw_rows: list[dict[str, str]]) -> str:
    stats, required_n, required_bytes = inventory_stats(raw_rows)
    lines = [
        "# Dataset README",
        "",
        "## Title",
        "",
        "Memory-induced thermal breathing of a granular force network: DEM source data, raw contact-network outputs and analysis scripts",
        "",
        "## Summary",
        "",
        "This dataset supports the manuscript `Memory-induced thermal breathing of a granular force network`. It contains processed source data for the main and Extended Data figures, analysis and plotting scripts, LIGGGHTS input/provenance materials, and a raw-data inventory for atom dumps, restart files and contact-local pair-force outputs used to construct force-network and breathing-map diagnostics.",
        "",
        "The central raw archive contains files needed for independent reanalysis of the DEM contact networks. Processed CSV files are sufficient to regenerate the manuscript plots, whereas the raw `.local`, dump and restart files support deeper inspection and reruns.",
        "",
        "## Required raw archive scale",
        "",
        f"- Required raw files: {required_n}",
        f"- Required raw size: {required_bytes / 1024**3:.2f} GB",
        "",
        "## File groups",
        "",
    ]
    for (tier, group), sizes in sorted(stats.items()):
        lines.append(f"- `{tier}/{group}`: {len(sizes)} files, {sum(sizes) / 1024**3:.2f} GB")
    lines.extend(
        [
            "",
            "## Recommended repository layout",
            "",
            "- `processed_source_data/`: all CSV/JSON files from `manuscript_prl/source_data/` used for figures, statistics and audits.",
            "- `analysis_scripts/`: manuscript plotting scripts, mining scripts and audit scripts.",
            "- `simulation_inputs/`: LIGGGHTS input decks and generated run inputs.",
            "- `raw_contact_force_outputs/`: contact-local pair-force `.local` files used for true-force network analyses.",
            "- `restart_files/`: settled, final and preloaded restart files used for reruns.",
            "- `raw_atom_dumps/`: atom dump files used for microstructure, topology and phase-state reconstruction.",
            "- `manifest/`: archive inventories, this README, DataCite metadata and figure-to-source-data maps.",
            "",
            "## Variables and units",
            "",
            "Processed source-data tables retain the column names used by the analysis scripts. Physical wall loads are simulation pressure proxies in Pa. Contact forces are DEM pair-force magnitudes in the LIGGGHTS unit convention used by the input decks. Dimensionless quantities include normalized force shares, route-centred coordinates, Spearman statistics, prediction scores and the loop number `Psi` defined in the manuscript.",
            "",
            "## Methods and provenance",
            "",
            "DEM calculations used LIGGGHTS-style input decks to cycle Li2TiO3-like spherical particles between cold and hot states. Post-processing scripts reconstruct contact networks, force-filtration variables, route-conditioned return maps, breathing parameters and figure source data. The raw-data inventory records relative paths, file groups, required/optional tiers and file sizes.",
            "",
            "## Access and licence",
            "",
            "The repository record should specify a public data licence chosen by the author and institution. Until DOI deposition is complete, raw files remain local and the manuscript should state that DOI-backed deposition is pending.",
            "",
            "## Preferred citation",
            "",
            "Wang, J. (2026). Memory-induced thermal breathing of a granular force network: DEM source data and raw contact-network outputs. [Repository]. [DOI to be assigned].",
            "",
        ]
    )
    return "\n".join(lines)


def build_datacite_metadata(raw_rows: list[dict[str, str]]) -> dict[str, object]:
    _, required_n, required_bytes = inventory_stats(raw_rows)
    return {
        "identifier": {"identifier": "DOI_TO_BE_ASSIGNED", "identifierType": "DOI"},
        "creators": [{"name": "Wang, Jian", "nameType": "Personal"}],
        "titles": [
            {
                "title": "Memory-induced thermal breathing of a granular force network: DEM source data and raw contact-network outputs"
            }
        ],
        "publisher": "Repository to be assigned",
        "publicationYear": date.today().year,
        "types": {"resourceTypeGeneral": "Dataset", "resourceType": "Simulation data"},
        "version": "1.0-review",
        "language": "en",
        "subjects": [
            {"subject": "granular matter"},
            {"subject": "discrete element method"},
            {"subject": "material memory"},
            {"subject": "force networks"},
            {"subject": "thermal ratcheting"},
            {"subject": "non-equilibrium soft matter"},
        ],
        "descriptions": [
            {
                "descriptionType": "Abstract",
                "description": (
                    "Dataset supporting a Nature Physics Article candidate on memory-induced thermal breathing in a thermally cycled granular force network. "
                    f"The DOI package should include processed source data, analysis scripts, input decks and {required_n} required raw DEM files "
                    f"({required_bytes / 1024**3:.2f} GB) comprising atom dumps, restart files and contact-local pair-force outputs."
                ),
            }
        ],
        "rightsList": [{"rights": "Licence to be selected before deposition"}],
        "relatedIdentifiers": [
            {
                "relationType": "IsSupplementTo",
                "relatedIdentifier": "MANUSCRIPT_DOI_OR_PREPRINT_TO_BE_ASSIGNED",
                "relatedIdentifierType": "DOI",
            }
        ],
        "fundingReferences": [],
    }


def build_figure_map(crossref_rows: list[dict[str, str]]) -> str:
    lines = [
        "# Figure-to-source-data map",
        "",
        "This map links curated Extended Data items to figure assets, source-data files, supported claims and evidence boundaries.",
        "",
        "| Extended Data | Figure base | Source data | Claim supported | Boundary |",
        "|---|---|---|---|---|",
    ]
    for row in crossref_rows:
        src = row["primary_source_data"]
        if row["secondary_source_data"]:
            src += "; " + row["secondary_source_data"]
        lines.append(
            f"| {row['extended_data']} | `{row['file_base']}` | `{src}` | {row['claim_supported']} | {row['boundary']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_upload_checklist(raw_rows: list[dict[str, str]]) -> str:
    stats, required_n, required_bytes = inventory_stats(raw_rows)
    return "\n".join(
        [
            "# Repository upload checklist",
            "",
            "## Before upload",
            "",
            "- Choose repository: Zenodo, Figshare, OSF, Dryad or institutional repository with DOI support.",
            "- Confirm author names, affiliations, ORCID identifiers and funder metadata.",
            "- Choose a data licence compatible with institutional rules.",
            "- Decide whether to upload one archive or several records for processed data, raw force outputs and simulation restarts.",
            "- Keep raw and processed data separate in the uploaded file tree.",
            "",
            "## Files to include",
            "",
            "- Processed source data and source-data inventory.",
            "- Main and Extended Data figure scripts.",
            "- Mining/audit scripts used for force-loop, return-map, irreversibility and onset analyses.",
            "- LIGGGHTS input decks and executable provenance notes.",
            f"- Required raw DEM files: {required_n} files, {required_bytes / 1024**3:.2f} GB.",
            "- This README, DataCite metadata draft, raw archive inventory and figure-to-source-data map.",
            "",
            "## After upload",
            "",
            "- Verify DOI resolves outside the author account.",
            "- Download at least one raw archive and one processed source-data file from the public/private-review link.",
            "- Replace all `[DOI]` placeholders in the manuscript, cover letter, Data Availability statement and source-data inventory.",
            "- Re-run `audit_nature_physics_submission_assets.py` after DOI wording is inserted.",
            "",
        ]
    )


def main() -> None:
    raw_rows = read_csv(SRC / "nature_physics_raw_data_archive_inventory.csv")
    crossref_rows = read_csv(SRC / "nature_physics_extended_data_crossref.csv")
    OUT.mkdir(exist_ok=True)
    write(OUT / "README_dataset.md", build_readme(raw_rows))
    write(OUT / "figure_to_source_data_map.md", build_figure_map(crossref_rows))
    write(OUT / "repository_upload_checklist.md", build_upload_checklist(raw_rows))
    write(OUT / "datacite_metadata_draft.json", json.dumps(build_datacite_metadata(raw_rows), indent=2))
    print(f"wrote {OUT / 'README_dataset.md'}")
    print(f"wrote {OUT / 'figure_to_source_data_map.md'}")
    print(f"wrote {OUT / 'repository_upload_checklist.md'}")
    print(f"wrote {OUT / 'datacite_metadata_draft.json'}")


if __name__ == "__main__":
    main()
