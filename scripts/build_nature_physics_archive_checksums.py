#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SRC = ROOT / "source_data"
OUT = ROOT / "archive_staging"
INVENTORY = SRC / "nature_physics_raw_data_archive_inventory.csv"
CHECKSUM_CSV = OUT / "required_raw_file_sha256.csv"
PACKAGE_PLAN_MD = OUT / "archive_package_plan.md"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_inventory() -> list[dict[str, str]]:
    with INVENTORY.open(newline="") as f:
        return list(csv.DictReader(f))


def package_name(row: dict[str, str]) -> str:
    path = Path(row["path"])
    parts = path.parts
    group = row["group"]
    if len(parts) >= 3 and parts[0] == "runs":
        family = parts[1]
    else:
        family = "misc"
    return f"{row['tier']}__{group}__{family}.tar.zst"


def main() -> None:
    rows = [r for r in read_inventory() if r["tier"] == "archive_required"]
    OUT.mkdir(exist_ok=True)
    checksum_rows: list[dict[str, str]] = []
    missing: list[str] = []
    package_groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for i, row in enumerate(rows, start=1):
        rel = Path(row["path"])
        path = PROJECT / rel
        pkg = package_name(row)
        package_groups[pkg].append(row)
        if not path.exists():
            missing.append(row["path"])
            digest = "MISSING"
            size = "0"
        else:
            digest = sha256_file(path)
            size = str(path.stat().st_size)
        checksum_rows.append(
            {
                "path": row["path"],
                "group": row["group"],
                "bytes": size,
                "sha256": digest,
                "suggested_package": pkg,
            }
        )
        if i % 250 == 0:
            print(f"hashed {i}/{len(rows)}")

    with CHECKSUM_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "group", "bytes", "sha256", "suggested_package"])
        writer.writeheader()
        writer.writerows(checksum_rows)

    lines = [
        "# Archive package plan",
        "",
        "Purpose: provide a deterministic packaging plan for DOI deposition without moving or copying raw files.",
        "",
        f"- Required raw files covered by checksum manifest: {len(checksum_rows)}",
        f"- Missing files during checksum generation: {len(missing)}",
        f"- Checksum manifest: `{CHECKSUM_CSV.relative_to(ROOT)}`",
        "",
        "## Suggested packages",
        "",
        "| Suggested package | Files | Size (GB) | Contents |",
        "|---|---:|---:|---|",
    ]
    for pkg, pkg_rows in sorted(package_groups.items()):
        total = sum(int(r["bytes"]) for r in pkg_rows)
        groups = ", ".join(sorted({r["group"] for r in pkg_rows}))
        lines.append(f"| `{pkg}` | {len(pkg_rows)} | {total / 1024**3:.2f} | {groups} |")
    lines.extend(
        [
            "",
            "## Example packaging commands",
            "",
            "Run from the project root, not from inside `manuscript_prl`:",
            "",
            "```bash",
            "# Example for one package. Repeat using paths from required_raw_file_sha256.csv.",
            "tar --zstd -cf archive_required__contact_local__long_cycle_force_probe.tar.zst \\",
            "  runs/long_cycle_force_probe/*/contacts_cycle_*.local",
            "```",
            "",
            "After creating archives, verify extracted file checksums against `required_raw_file_sha256.csv` before DOI upload.",
            "",
        ]
    )
    if missing:
        lines.extend(["## Missing files", ""])
        lines.extend(f"- `{p}`" for p in missing)
        lines.append("")
    PACKAGE_PLAN_MD.write_text("\n".join(lines))
    print(f"wrote {CHECKSUM_CSV}")
    print(f"wrote {PACKAGE_PLAN_MD}")


if __name__ == "__main__":
    main()
