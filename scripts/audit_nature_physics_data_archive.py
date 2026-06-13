#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SRC = ROOT / "source_data"
OUT_CSV = SRC / "nature_physics_raw_data_archive_inventory.csv"
OUT_MD = ROOT / "nature_physics_raw_data_archive_manifest.md"


RAW_PATTERNS = {
    "contact_local": ["contacts_", ".local"],
    "restart": ["restart"],
    "atom_dump": ["dump"],
    "input_deck": ["in."],
    "liggghts_input": [".liggghts", ".lmp"],
}

KEY_RUN_DIRS = [
    PROJECT / "runs" / "free_quasistatic_10k_prod",
    PROJECT / "runs" / "confined_structural_10k_prod",
    PROJECT / "runs" / "confined_structural_10k_precompressed_prod",
    PROJECT / "runs" / "regime_matrix_10k",
    PROJECT / "runs" / "targeted_uncertainty_matrix",
    PROJECT / "runs" / "contact_force_probe",
    PROJECT / "runs" / "long_cycle_force_probe",
    PROJECT / "runs" / "long_cycles_10k",
    PROJECT / "runs" / "stiffness_sensitivity",
    PROJECT / "runs" / "relaxed_ensemble_matrix",
]

PROCESSED_DIRS = [
    ROOT / "source_data",
    PROJECT / "runs",
    PROJECT / "scripts",
]


def classify(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("in."):
        return "input_deck"
    for group, tokens in RAW_PATTERNS.items():
        if all(token in name for token in tokens):
            return group
    return None


def under_any(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def main() -> None:
    rows: list[dict[str, str]] = []
    for path in PROJECT.rglob("*"):
        if not path.is_file():
            continue
        group = classify(path)
        if group is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        in_key_run = under_any(path, KEY_RUN_DIRS)
        tier = "archive_required" if group in {"contact_local", "restart", "atom_dump"} and in_key_run else "supporting_or_optional"
        rows.append(
            {
                "group": group,
                "tier": tier,
                "bytes": str(stat.st_size),
                "mb": f"{stat.st_size / 1024**2:.3f}",
                "path": str(path.relative_to(PROJECT)),
            }
        )

    rows.sort(key=lambda r: (r["tier"], r["group"], -float(r["mb"]), r["path"]))
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "tier", "bytes", "mb", "path"])
        writer.writeheader()
        writer.writerows(rows)

    stats: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        stats[(r["tier"], r["group"])].append(int(r["bytes"]))
    archive_required = [r for r in rows if r["tier"] == "archive_required"]
    optional = [r for r in rows if r["tier"] != "archive_required"]
    total_required = sum(int(r["bytes"]) for r in archive_required)
    total_all = sum(int(r["bytes"]) for r in rows)

    lines = [
        "# Nature Physics raw-data archive manifest",
        "",
        "Purpose: define the raw and supporting files that should be deposited in a DOI-backed archive before final submission.",
        "",
        "## Summary",
        "",
        f"- Required raw archive files: {len(archive_required)}",
        f"- Required raw archive size: {total_required / 1024**3:.2f} GB",
        f"- All matched raw/supporting files: {len(rows)}",
        f"- All matched size: {total_all / 1024**3:.2f} GB",
        f"- Machine-readable inventory: `{OUT_CSV.relative_to(ROOT)}`",
        "",
        "## Required archive tiers",
        "",
    ]
    for (tier, group), sizes in sorted(stats.items()):
        lines.append(f"- {tier} / {group}: {len(sizes)} files, {sum(sizes) / 1024**3:.2f} GB")
    lines.extend(
        [
            "",
            "## Recommended DOI package structure",
            "",
            "1. `processed_source_data/`: copy `manuscript_prl/source_data/` and run-level summary CSV files used by the figures.",
            "2. `analysis_scripts/`: copy manuscript plotting and mining scripts plus the top-level `scripts/` directory needed to rebuild processed tables.",
            "3. `simulation_inputs/`: copy LIGGGHTS input decks, generated route inputs and material-property scripts.",
            "4. `raw_contact_force_outputs/`: archive the required `.local` true-contact-force files from `runs/contact_force_probe/` and `runs/long_cycle_force_probe/`.",
            "5. `restart_files/`: archive final and settled restart files for the run families used in the manuscript.",
            "6. `manifest/`: include this manifest, the CSV inventory, code-version notes and a README explaining which files reproduce each figure.",
            "",
            "## Boundary",
            "",
            "The processed CSV source data are sufficient to reproduce manuscript and Extended Data plots. The raw `.local` contact-force files, restart files and input decks are required for independent reanalysis and simulation reruns. Raw files remain local until deposited in a public DOI-backed repository.",
            "",
            "## Data availability wording after DOI assignment",
            "",
            "Replace the placeholder DOI in the manuscript with: `Raw DEM restart files, contact-local outputs, processed source data and analysis scripts have been deposited at [repository] under DOI [DOI]. Processed source data for all main and Extended Data figures are also included with the manuscript package.`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
