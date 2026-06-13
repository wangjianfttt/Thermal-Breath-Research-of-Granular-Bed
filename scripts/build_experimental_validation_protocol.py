#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
OUT_CSV = SRC / "nphys_experimental_validation_protocol.csv"
OUT_MD = ROOT / "nature_physics_experimental_validation_protocol.md"


def corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    r, p = stats.pearsonr(x.astype(float), y.astype(float))
    return float(r), float(p)


def model_score(model_tests: pd.DataFrame, model: str, validation: str) -> float:
    row = model_tests[(model_tests["model"] == model) & (model_tests["validation"] == validation)]
    return float(row["r2_vs_training_mean"].iloc[0])


def main() -> None:
    memory = pd.read_csv(SRC / "nphys_fig1_memory_source.csv")
    semi = pd.read_csv(SRC / "nphys_fig2_boundary_semi_confined_source.csv")
    pre = pd.read_csv(SRC / "nphys_fig2_boundary_precompressed_source.csv")
    two = pd.read_csv(SRC / "nature_physics_two_channel_summary_source.csv")
    robust = pd.read_csv(SRC / "nphys_force_loop_robustness_conditional.csv")
    hierarchy = pd.read_csv(SRC / "nphys_mechanism_hierarchy_model_tests.csv")
    breath = pd.read_csv(SRC / "nphys_breathing_cycle_correlations.csv")

    z_gain = memory["z_mean"].iloc[-1] / memory["z_mean"].iloc[0] - 1
    survival_gain_pp = 100 * (memory["contact_survival"].iloc[-1] - memory["contact_survival"].iloc[0])
    rearrangement_ratio = memory["rms_displacement_m"].iloc[-1] / memory["rms_displacement_m"].iloc[0]

    heat_semi = semi[semi["Phase"] == "Heating"]
    heat_pre = pre[pre["Phase"] == "Heating"]
    side_amp = heat_semi["SidePressure_Pa"].iloc[-1] / heat_semi["SidePressure_Pa"].iloc[0]
    semi_bottom_amp = heat_semi["BottomPressure_Pa"].iloc[-1] / heat_semi["BottomPressure_Pa"].iloc[0]
    pre_bottom_release = heat_pre["BottomPressure_Pa"].iloc[-1] / heat_pre["BottomPressure_Pa"].iloc[0]
    pre_lid_release = heat_pre["LidPressure_Pa"].iloc[-1] / heat_pre["LidPressure_Pa"].iloc[0]

    cold_r, cold_p = corr(two["Z_cold_N_mean"], two["cold_bottom_pN_mean"])
    hot_r, hot_p = corr(two["hot_force_mean"], two["hot_bottom_pN_mean"])

    loop_row = robust[robust["test"] == "loop_after_top5_and_cycle"].iloc[0]
    top5_row = robust[robust["test"] == "top5_after_loop_and_cycle"].iloc[0]
    psi_r2 = model_score(hierarchy, "dimensionless loop", "leave_one_route_out")
    force_tail_r2 = model_score(hierarchy, "force tail", "leave_one_route_out")

    lag_fabric = breath[breath["relationship"] == "inhaled_geometry_to_next_cold_fabric"].iloc[0]
    lag_loop = breath[breath["relationship"] == "inhaled_positive_cycles_to_next_cold_loop_memory"].iloc[0]

    rows = [
        {
            "test_id": "T1",
            "claim": "Thermal cycling trains a persistent fabric register.",
            "recommended_system": "Thermally cycled transparent or X-ray-accessible granular column; ceramic or analogue beads.",
            "phase_window": "cold state after each cycle, with hot-state snapshots if available",
            "primary_observable": "Cycle-resolved height/porosity plus coordination/contact-survival proxy.",
            "instrumentation": "dilatometry or image-derived bed height; X-ray/optical tomography, acoustic transmission or repeated unload-reload stiffness as fabric proxies",
            "control": "repeat identical thermal cycles at fixed bed mass and boundary condition; compare with a no-thermal-cycle settled control when possible",
            "dem_target": f"coordination gain {100*z_gain:.1f}%; contact-survival gain {survival_gain_pp:.1f} pp; rearrangement ratio {rearrangement_ratio:.2f}",
            "source_data_anchor": "source_data/nphys_fig1_memory_source.csv; source_data/nphys_training_order_parameter_flow_cycle_metrics.csv",
            "minimum_directional_test": "Fabric/contact-survival proxy should increase and rearrangement amplitude should damp over repeated identical cycles.",
            "falsifier": "No persistent rise in fabric/contact-survival proxy and no damping of rearrangement under repeated identical cycles.",
            "priority": "near-term",
        },
        {
            "test_id": "T2",
            "claim": "Boundary projection changes the sign of load response.",
            "recommended_system": "Same bed cycled under semi-confined and precompressed boundary protocols with phase-resolved wall-force arrays.",
            "phase_window": "hot peak and cooled return at each cycle",
            "primary_observable": "Final/initial hot-load ratios at side wall, bottom and lid.",
            "instrumentation": "segmented side-wall, bottom and lid force sensors or calibrated pressure films with thermal compensation",
            "control": "use the same particle batch and thermal ramp while changing only boundary clearance or precompression",
            "dem_target": f"semi side amplification {side_amp:.2f}x; semi bottom amplification {semi_bottom_amp:.2f}x; precompressed bottom {pre_bottom_release:.2f}x; precompressed lid {pre_lid_release:.3f}x",
            "source_data_anchor": "source_data/nphys_fig2_boundary_semi_confined_source.csv; source_data/nphys_fig2_boundary_precompressed_source.csv",
            "minimum_directional_test": "Semi-confined routes should amplify selected hot wall loads while precompressed routes release stored preload.",
            "falsifier": "All boundary protocols collapse onto one monotonic scalar pressure-growth curve.",
            "priority": "near-term",
        },
        {
            "test_id": "T3",
            "claim": "Cold and hot readouts require different state variables.",
            "recommended_system": "Phase-resolved imaging or acoustic fabric readout paired with hot wall-force intermittency.",
            "phase_window": "paired cold and hot states within the same cycle",
            "primary_observable": "Cold load versus fabric coordinate, and hot load versus hot force/intermittency coordinate.",
            "instrumentation": "tomography/acoustic fabric proxy plus high-rate wall-force array to capture intermittent hot overload",
            "control": "compare cold-retained-load ranking with hot-overload ranking across at least three boundary/friction/expansion routes",
            "dem_target": f"cold load vs coordination r={cold_r:.2f}; hot load vs direct hot force scale r={hot_r:.2f}",
            "source_data_anchor": "source_data/nature_physics_two_channel_summary_source.csv; source_data/nphys_readout_orthogonality_regression.csv",
            "minimum_directional_test": "Cold retained load should be better organised by fabric, while transient hot load should be better organised by force/intermittency.",
            "falsifier": "Residual cold load and transient hot overload collapse onto the same scalar density or pressure coordinate.",
            "priority": "near-term",
        },
        {
            "test_id": "T4",
            "claim": "Hot overload is organised by force-loop activation, not only force tails.",
            "recommended_system": "Photoelastic or index-matched analogue bed, or calibrated DEM constrained by phase-resolved wall-force arrays.",
            "phase_window": "hot states, paired with immediately preceding cold state",
            "primary_observable": "Graph-cycle embedding of high-force contacts after force-tail controls.",
            "instrumentation": "photoelastic force-chain imaging, index-matched refractive imaging, or DEM-calibrated contact graph constrained by wall-force arrays",
            "control": "shuffle or delete graph-cycle embedding while preserving the measured force-tail distribution",
            "dem_target": f"loop residual rho={float(loop_row['spearman']):.2f}; top5 residual rho={float(top5_row['spearman']):.2f}; dimensionless loop LORO R2={psi_r2:.2f}; force-tail LORO R2={force_tail_r2:.2f}",
            "source_data_anchor": "source_data/nphys_force_loop_robustness_conditional.csv; source_data/nphys_dimensionless_loop_collapse_model_tests.csv; source_data/nphys_two_scale_variance_partition_shuffle_tests.csv",
            "minimum_directional_test": "Loop embedding should retain overload predictability after force-tail and cycle controls.",
            "falsifier": "Force-tail metrics predict overload after graph-loop embedding is destroyed, shuffled or controlled.",
            "priority": "hard but decisive",
        },
        {
            "test_id": "T5",
            "claim": "A hot excursion leaves a next-cold imprint.",
            "recommended_system": "Cycle-resolved tomography, acoustic memory or unload-reload probing before and after hot excursions.",
            "phase_window": "hot state n and next cold state n+1",
            "primary_observable": "Next-cold fabric or loop-memory proxy conditioned on previous hot excursion.",
            "instrumentation": "tomography/acoustic fabric proxy or repeated small-amplitude mechanical probe before and after hot excursions",
            "control": "route-centre the observable or compare against circularly shifted hot-excursion labels",
            "dem_target": f"fabric imprint rho={float(lag_fabric['spearman_within_regime_centered']):.2f}; loop imprint rho={float(lag_loop['spearman_within_regime_centered']):.2f}",
            "source_data_anchor": "source_data/nphys_breathing_cycle_correlations.csv; source_data/nphys_five_route_breath_memory_spectrum_lag_scan.csv",
            "minimum_directional_test": "Hot-phase excursions should predict at least one next-cold fabric or loop-memory observable after route means are removed.",
            "falsifier": "Hot-phase excursions do not predict any next-cold fabric, acoustic or loop-memory observable.",
            "priority": "medium-term",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = [
        "# Experimental validation protocol",
        "",
        "Purpose: translate the DEM mechanism into falsifiable experimental or analogue-test protocols. The thresholds below are DEM-derived targets and directional tests, not experimental results.",
        "",
        "## Protocol summary",
        "",
        "| Test | Claim | Primary observable | DEM-derived target | Falsifier | Priority |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['test_id']} | {row['claim']} | {row['primary_observable']} | {row['dem_target']} | {row['falsifier']} | {row['priority']} |"
        )
    lines.extend(
        [
            "",
            "## Experimental design details",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['test_id']} design",
                "",
                f"- Phase window: {row['phase_window']}.",
                f"- Instrumentation: {row['instrumentation']}.",
                f"- Critical control: {row['control']}.",
                f"- Source-data anchor: {row['source_data_anchor']}.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended sequence",
            "",
            "1. Start with T1--T3 because they require only phase-resolved wall forces plus independent fabric or density observables.",
            "2. Use T4 as the decisive force-network test in photoelastic, index-matched or DEM-calibrated analogue beds.",
            "3. Use T5 to test whether the breathing map has memory beyond the immediately measured hot state.",
            "",
            "## Manuscript boundary",
            "",
            "These tests make the theory falsifiable, but they do not replace direct experimental validation. Until at least one test is performed, the manuscript should continue to state that wall loads are DEM pressure proxies and that force-loop observability is easiest in analogue beds or calibrated DEM constrained by wall-force arrays.",
            "",
            f"Machine-readable protocol: `{OUT_CSV.relative_to(ROOT)}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
