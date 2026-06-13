#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
REMOTE = os.environ.get("REMOTE", "wangjian@192.168.3.6")
REMOTE_PROJECT = os.environ.get("REMOTE_PROJECT", "/data1/codex_runs/heat_Li2TiO3_route_generality_force_probe")
EXPECTED_CONTACT_LOCAL = 60
TAGS = [
    "a150_mu010_g020_c30",
    "a150_mu030_g020_c30",
    "a050_mu060_g020_c30",
]


def ssh_command(remote_script: str) -> list[str]:
    base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", REMOTE]
    if os.environ.get("SSHPASS"):
        return ["sshpass", "-e", *base, remote_script]
    return [*base, remote_script]


def fetch_remote_rows() -> tuple[list[dict[str, str]], str]:
    tags = " ".join(shlex.quote(t) for t in TAGS)
    script = f"""
set -euo pipefail
cd {shlex.quote(REMOTE_PROJECT)}
pid=""
running=0
if [[ -f runs/route_generality_force_probe_logs/batch.pid ]]; then
  pid="$(cat runs/route_generality_force_probe_logs/batch.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    running=1
  fi
fi
echo "tag,remote_project,pid,batch_running,log_exists,log_size_bytes,contact_local_count,dump_count,last_log_line"
for tag in {tags}; do
  log="runs/route_generality_force_probe_logs/$tag.log"
  if [[ -f "$log" ]]; then
    log_exists=1
    log_size="$(wc -c < "$log" | tr -d " ")"
    last_line="$(tail -n 1 "$log" | tr "," ";")"
  else
    log_exists=0
    log_size=0
    last_line=""
  fi
  folder="runs/long_cycle_force_probe/$tag"
  if [[ -d "$folder" ]]; then
    contact_count="$(find "$folder" -name 'contacts_cycle_*_*.local' | wc -l | tr -d " ")"
    dump_count="$(find "$folder" -name 'cycle_*_*.dump' | wc -l | tr -d " ")"
  else
    contact_count=0
    dump_count=0
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\\n" "$tag" {shlex.quote(REMOTE_PROJECT)} "$pid" "$running" "$log_exists" "$log_size" "$contact_count" "$dump_count" "$last_line"
done
"""
    proc = subprocess.run(ssh_command(script), check=True, text=True, capture_output=True)
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    return rows, proc.stderr


def write_outputs(rows: list[dict[str, str]], stderr: str) -> None:
    SRC.mkdir(exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    out_csv = SRC / "nphys_route_generality_remote_manifest.csv"
    fieldnames = [
        "tag",
        "remote_project",
        "pid",
        "batch_running",
        "log_exists",
        "log_size_bytes",
        "contact_local_count",
        "dump_count",
        "expected_contact_local_count",
        "complete",
        "last_log_line",
        "checked_at",
    ]
    enriched: list[dict[str, str]] = []
    for row in rows:
        count = int(row.get("contact_local_count") or 0)
        item = dict(row)
        item["expected_contact_local_count"] = str(EXPECTED_CONTACT_LOCAL)
        item["complete"] = "yes" if count >= EXPECTED_CONTACT_LOCAL else "no"
        item["checked_at"] = now
        enriched.append(item)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    complete = sum(1 for row in enriched if row["complete"] == "yes")
    running = any(row.get("batch_running") == "1" for row in enriched)
    lines = [
        "# Route-generality remote manifest",
        "",
        f"Checked: {now}.",
        f"Remote project: `{REMOTE_PROJECT}`.",
        f"Batch running: `{str(running).lower()}`.",
        f"Complete routes: {complete} / {len(TAGS)}.",
        "",
        "| tag | contact local | dump files | log size | complete |",
        "|---|---:|---:|---:|---|",
    ]
    for row in enriched:
        lines.append(
            f"| `{row['tag']}` | {row['contact_local_count']} / {row['expected_contact_local_count']} | "
            f"{row['dump_count']} | {row['log_size_bytes']} | {row['complete']} |"
        )
    lines += [
        "",
        "Interpretation: this manifest is a completion gate only. It is not mechanism evidence until all routes are complete, pulled locally and post-processed.",
    ]
    if stderr.strip():
        lines += ["", "SSH stderr was non-empty but the manifest command succeeded."]
    (ROOT / "nature_physics_route_generality_remote_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, stderr = fetch_remote_rows()
    write_outputs(rows, stderr)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
