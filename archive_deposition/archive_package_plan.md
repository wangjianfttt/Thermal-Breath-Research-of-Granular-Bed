# Archive package plan

Purpose: provide a deterministic packaging plan for DOI deposition without moving or copying raw files.

- Required raw files covered by checksum manifest: 2823
- Missing files during checksum generation: 0
- Checksum manifest: `archive_staging/required_raw_file_sha256.csv`

## Suggested packages

| Suggested package | Files | Size (GB) | Contents |
|---|---:|---:|---|
| `archive_required__atom_dump__confined_structural_10k_precompressed_prod.tar.zst` | 40 | 0.07 | atom_dump |
| `archive_required__atom_dump__confined_structural_10k_prod.tar.zst` | 40 | 0.07 | atom_dump |
| `archive_required__atom_dump__contact_force_probe.tar.zst` | 120 | 0.19 | atom_dump |
| `archive_required__atom_dump__free_quasistatic_10k_prod.tar.zst` | 80 | 0.10 | atom_dump |
| `archive_required__atom_dump__long_cycle_force_probe.tar.zst` | 480 | 0.77 | atom_dump |
| `archive_required__atom_dump__long_cycles_10k.tar.zst` | 180 | 0.29 | atom_dump |
| `archive_required__atom_dump__regime_matrix_10k.tar.zst` | 540 | 0.87 | atom_dump |
| `archive_required__atom_dump__relaxed_ensemble_matrix.tar.zst` | 180 | 0.28 | atom_dump |
| `archive_required__atom_dump__stiffness_sensitivity.tar.zst` | 122 | 0.19 | atom_dump |
| `archive_required__atom_dump__targeted_uncertainty_matrix.tar.zst` | 360 | 0.56 | atom_dump |
| `archive_required__contact_local__contact_force_probe.tar.zst` | 120 | 0.29 | contact_local |
| `archive_required__contact_local__long_cycle_force_probe.tar.zst` | 480 | 1.07 | contact_local |
| `archive_required__restart__confined_structural_10k_precompressed_prod.tar.zst` | 1 | 0.00 | restart |
| `archive_required__restart__confined_structural_10k_prod.tar.zst` | 1 | 0.00 | restart |
| `archive_required__restart__contact_force_probe.tar.zst` | 6 | 0.02 | restart |
| `archive_required__restart__free_quasistatic_10k_prod.tar.zst` | 1 | 0.00 | restart |
| `archive_required__restart__long_cycle_force_probe.tar.zst` | 8 | 0.02 | restart |
| `archive_required__restart__long_cycles_10k.tar.zst` | 3 | 0.01 | restart |
| `archive_required__restart__regime_matrix_10k.tar.zst` | 27 | 0.08 | restart |
| `archive_required__restart__relaxed_ensemble_matrix.tar.zst` | 9 | 0.03 | restart |
| `archive_required__restart__stiffness_sensitivity.tar.zst` | 7 | 0.02 | restart |
| `archive_required__restart__targeted_uncertainty_matrix.tar.zst` | 18 | 0.06 | restart |

## Example packaging commands

Run from the project root, not from inside `manuscript_prl`:

```bash
# Example for one package. Repeat using paths from required_raw_file_sha256.csv.
tar --zstd -cf archive_required__contact_local__long_cycle_force_probe.tar.zst \
  runs/long_cycle_force_probe/*/contacts_cycle_*.local
```

After creating archives, verify extracted file checksums against `required_raw_file_sha256.csv` before DOI upload.
