#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
OUT_CSV = SRC / "nature_physics_numerical_claim_audit.csv"
OUT_MD = ROOT / "nature_physics_numerical_claim_audit.md"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    location: str
    claim_text: str
    source_file: str
    source_selector: str
    value_name: str
    claimed_value: float
    tolerance: float
    note: str
    extractor: Callable[[pd.DataFrame], float]


def one(df: pd.DataFrame, **equals: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in equals.items():
        mask &= df[key].astype(str).eq(value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {equals}, found {len(rows)}")
    return rows.iloc[0]


def first_value(df: pd.DataFrame, column: str) -> float:
    return float(df.iloc[0][column])


CLAIMS: list[Claim] = [
    Claim(
        "abstract_minimal_state_psi_loro_asinh",
        "Abstract; Extended Data hierarchy text",
        "dimensionless loop number gives R^2=0.82 for asinh-transformed overload",
        "nphys_minimal_state_hot_overload.csv",
        "target=overload_number_asinh; model_family=dimensionless loop; validation=leave_one_route_out",
        "r2_vs_training_mean",
        0.82,
        0.005,
        "Asinh-overload minimal-state hierarchy, not the raw-overload collapse.",
        lambda df: float(
            one(
                df,
                target="overload_number_asinh",
                model_family="dimensionless loop",
                validation="leave_one_route_out",
            )["r2_vs_training_mean"]
        ),
    ),
    Claim(
        "main_dimensionless_loop_raw_loro",
        "Main text, dimensionless loop paragraph",
        "Psi achieved R^2=0.62 for raw overload in leave-one-route-out prediction",
        "nphys_dimensionless_loop_collapse_model_tests.csv",
        "predictor=dimensionless_loop_number; target=overload_number; validation=leave_one_route_out",
        "r2_vs_training_mean",
        0.62,
        0.005,
        "Raw-overload collapse audit; deliberately distinct from the asinh hierarchy audit.",
        lambda df: float(
            one(
                df,
                predictor="dimensionless_loop_number",
                target="overload_number",
                validation="leave_one_route_out",
            )["r2_vs_training_mean"]
        ),
    ),
    Claim(
        "main_loop_activation_raw_loro_control",
        "Main text, dimensionless loop paragraph",
        "pure loop activation gave R^2=0.087 in the raw-overload collapse audit",
        "nphys_dimensionless_loop_collapse_model_tests.csv",
        "predictor=loop_activation; target=overload_number; validation=leave_one_route_out",
        "r2_vs_training_mean",
        0.087,
        0.0005,
        "Control model for raw overload.",
        lambda df: float(
            one(
                df,
                predictor="loop_activation",
                target="overload_number",
                validation="leave_one_route_out",
            )["r2_vs_training_mean"]
        ),
    ),
    Claim(
        "main_top5_dimensionless_raw_loro_control",
        "Main text, dimensionless loop paragraph",
        "dimensionless top-5% force-tail control gave R^2=-0.24 in the raw-overload collapse audit",
        "nphys_dimensionless_loop_collapse_model_tests.csv",
        "predictor=dimensionless_top5_number; target=overload_number; validation=leave_one_route_out",
        "r2_vs_training_mean",
        -0.24,
        0.005,
        "Negative transfer score supports force-tail insufficiency.",
        lambda df: float(
            one(
                df,
                predictor="dimensionless_top5_number",
                target="overload_number",
                validation="leave_one_route_out",
            )["r2_vs_training_mean"]
        ),
    ),
    Claim(
        "main_loop_to_overload_within_route",
        "Main force-loop mechanism paragraph",
        "force-loop activation correlates with overload within routes, rho=0.74",
        "nphys_force_loop_chain_correlations.csv",
        "predictor=force_h1_birth_force_share_hot_minus_cold; target=force_p99_hot_minus_cold",
        "spearman_within_regime_centered",
        0.74,
        0.005,
        "Five-route true-force rerun, 150 paired states.",
        lambda df: float(
            one(
                df,
                predictor="force_h1_birth_force_share_hot_minus_cold",
                target="force_p99_hot_minus_cold",
            )["spearman_within_regime_centered"]
        ),
    ),
    Claim(
        "main_top5_tail_negative_control",
        "Main force-loop mechanism paragraph",
        "hot-minus-cold top-5% force-share control is anti-correlated within routes, rho=-0.48",
        "nphys_force_loop_chain_correlations.csv",
        "predictor=force_share_top5_edges_hot_minus_cold; target=force_p99_hot_minus_cold",
        "spearman_within_regime_centered",
        -0.48,
        0.005,
        "Force concentration is not a substitute overload coordinate.",
        lambda df: float(
            one(
                df,
                predictor="force_share_top5_edges_hot_minus_cold",
                target="force_p99_hot_minus_cold",
            )["spearman_within_regime_centered"]
        ),
    ),
    Claim(
        "main_route_generality_loop_birth_rho",
        "Main force-loop mechanism paragraph; Extended Data Fig. 6",
        "additional true-force routes retain route-centred loop-birth specificity, rho=0.499",
        "nphys_route_generality_true_force_extension_tests.csv",
        "scope=route_centered_hot_minus_cold; predictor=force_h1_birth_force_share_hot_minus_cold; target=force_p99_hot_minus_cold",
        "spearman_rho",
        0.499,
        0.0005,
        "Three-route true-force extension; route-centred diagnostic, not a universal material law.",
        lambda df: float(
            one(
                df,
                scope="route_centered_hot_minus_cold",
                predictor="force_h1_birth_force_share_hot_minus_cold",
                target="force_p99_hot_minus_cold",
            )["spearman_rho"]
        ),
    ),
    Claim(
        "main_route_generality_loop_birth_p",
        "Main force-loop mechanism paragraph; Extended Data Fig. 6",
        "additional true-force routes retain route-centred loop-birth specificity, P=0.0002",
        "nphys_route_generality_true_force_extension_tests.csv",
        "scope=route_centered_hot_minus_cold; predictor=force_h1_birth_force_share_hot_minus_cold; target=force_p99_hot_minus_cold",
        "permutation_p_two_sided",
        0.0002,
        0.00005,
        "Route-preserving residual permutation test for the three-route true-force extension.",
        lambda df: float(
            one(
                df,
                scope="route_centered_hot_minus_cold",
                predictor="force_h1_birth_force_share_hot_minus_cold",
                target="force_p99_hot_minus_cold",
            )["permutation_p_two_sided"]
        ),
    ),
    Claim(
        "main_eight_route_hot_cost_loop_rho",
        "Abstract; Main force-loop mechanism paragraph; reserve Fig. 67",
        "eight-route hot-cost audit gives route-centred loop activation rho=0.77",
        "nphys_eight_route_hot_breath_cost_tests.csv",
        "predictor=loop activation; target=route-centred asinh overload",
        "spearman_rho_route_centered",
        0.77,
        0.005,
        "Eight-route hot-breath cost diagnostic; extends hot-cost arm, not next-cold imprint.",
        lambda df: float(one(df, predictor="loop activation")["spearman_rho_route_centered"]),
    ),
    Claim(
        "main_eight_route_hot_cost_tail_rho",
        "Abstract; Main force-loop mechanism paragraph; reserve Fig. 67",
        "eight-route hot-cost audit gives force-tail control rho=-0.36",
        "nphys_eight_route_hot_breath_cost_tests.csv",
        "predictor=force tail; target=route-centred asinh overload",
        "spearman_rho_route_centered",
        -0.36,
        0.005,
        "Eight-route force-tail control remains oppositely signed.",
        lambda df: float(one(df, predictor="force tail")["spearman_rho_route_centered"]),
    ),
    Claim(
        "main_eight_route_hot_cost_loop_p",
        "Main force-loop mechanism paragraph; reserve Fig. 67",
        "eight-route hot-cost audit gives route-preserving P=0.0002 for loop activation",
        "nphys_eight_route_hot_breath_cost_tests.csv",
        "predictor=loop activation; target=route-centred asinh overload",
        "route_preserving_permutation_p",
        0.0002,
        0.00005,
        "Route-preserving permutation after route centring.",
        lambda df: float(one(df, predictor="loop activation")["route_preserving_permutation_p"]),
    ),
    Claim(
        "main_spatial_conduit_loop_rho",
        "Reserve spatial-fingerprint audit",
        "eight-route wall-coupled conduit subset of cycle-closing loop force has route-centred rho=0.66 with overload",
        "nphys_force_loop_spatial_fingerprint_correlations.csv",
        "predictor=loop conduit force share",
        "spearman_route_centered",
        0.66,
        0.005,
        "Eight-route spatial fingerprint audit over 480 true-force states; sub-readout, not a replacement coordinate.",
        lambda df: float(one(df, predictor="loop conduit force share")["spearman_route_centered"]),
    ),
    Claim(
        "main_spatial_conduit_loop_loro",
        "Reserve spatial-fingerprint audit",
        "eight-route wall-coupled conduit loop force gives leave-one-route R^2=0.73",
        "nphys_force_loop_spatial_fingerprint_transfer_tests.csv",
        "model=conduit loops",
        "r2_vs_training_mean",
        0.73,
        0.005,
        "Spatial sub-readout transfer score.",
        lambda df: float(one(df, model="conduit loops")["r2_vs_training_mean"]),
    ),
    Claim(
        "main_spatial_total_loop_loro",
        "Main force-loop mechanism paragraph",
        "eight-route total loop force share remains more transferable, R^2=0.83",
        "nphys_force_loop_spatial_fingerprint_transfer_tests.csv",
        "model=loop total",
        "r2_vs_training_mean",
        0.83,
        0.005,
        "Total cycle-closing loop sector remains the primary overload coordinate.",
        lambda df: float(one(df, model="loop total")["r2_vs_training_mean"]),
    ),
    Claim(
        "reserve_force_filtration_betti_area_rho",
        "Reserve force-filtration topology audit",
        "eight-route Betti-1 birth area has route-centred rho=0.81 with overload",
        "nphys_force_filtration_topology_correlations.csv",
        "predictor=Betti-1 birth area",
        "spearman_route_centered",
        0.81,
        0.005,
        "Threshold-free force-filtration topology diagnostic; not a phase-transition order parameter.",
        lambda df: float(one(df, predictor="Betti-1 birth area")["spearman_route_centered"]),
    ),
    Claim(
        "reserve_force_filtration_betti_area_loro",
        "Reserve force-filtration topology audit",
        "eight-route Betti-1 birth area gives leave-one-route R^2=0.74",
        "nphys_force_filtration_topology_transfer_tests.csv",
        "model=Betti area",
        "r2_vs_training_mean",
        0.74,
        0.005,
        "Threshold robustness support; total loop force remains the primary coordinate.",
        lambda df: float(one(df, model="Betti area")["r2_vs_training_mean"]),
    ),
    Claim(
        "main_causal_partial_loop",
        "Main causal-path paragraph",
        "route-fixed partial rank correlation of 0.90 for dimensionless loop number",
        "nphys_causal_path_partial_residuals.csv",
        "predictor=loop_number; controls=route_fixed_effects+cycle",
        "partial_spearman",
        0.90,
        0.005,
        "Rounded from 0.895; path audit after route and cycle effects.",
        lambda df: float(one(df, predictor="loop_number", controls="route_fixed_effects+cycle")["partial_spearman"]),
    ),
    Claim(
        "main_causal_loop_after_tail_gain",
        "Main causal-path paragraph",
        "adding loop after force-tail control increases R^2 by 0.103",
        "nphys_causal_path_permutation_gain.csv",
        "test=loop_after_tail",
        "delta_r2",
        0.103,
        0.0005,
        "Within-route shuffled null path audit.",
        lambda df: float(one(df, test="loop_after_tail")["delta_r2"]),
    ),
    Claim(
        "main_causal_tail_after_loop_gain",
        "Main causal-path paragraph",
        "adding force-tail after loop changes R^2 by 0.001",
        "nphys_causal_path_permutation_gain.csv",
        "test=tail_after_loop",
        "delta_r2",
        0.001,
        0.0005,
        "Rounded from 0.000636; reciprocal increment is negligible.",
        lambda df: float(one(df, test="tail_after_loop")["delta_r2"]),
    ),
    Claim(
        "main_two_scale_variance_loop_after_route_tail_gain",
        "Main two-scale mechanism paragraph",
        "adding L+S L after the route-tail model increases R^2 by 0.105",
        "nphys_two_scale_variance_partition_shuffle_tests.csv",
        "test=loop_sector_after_tail_and_route",
        "observed_delta_r2",
        0.105,
        0.0005,
        "Two-scale variance-partition audit; diagnostic, not randomized causal proof.",
        lambda df: float(one(df, test="loop_sector_after_tail_and_route")["observed_delta_r2"]),
    ),
    Claim(
        "main_two_scale_variance_loop_shuffle_p",
        "Main two-scale mechanism paragraph",
        "route-preserving loop-shuffle null gives P=0.0002",
        "nphys_two_scale_variance_partition_shuffle_tests.csv",
        "test=loop_sector_after_tail_and_route",
        "route_preserving_shuffle_p",
        0.0002,
        0.00005,
        "Route-preserving null keeps route distributions but breaks loop-overload cycle alignment.",
        lambda df: float(one(df, test="loop_sector_after_tail_and_route")["route_preserving_shuffle_p"]),
    ),
    Claim(
        "main_two_scale_route_gain_exact_p",
        "Main two-scale mechanism paragraph",
        "slow route coordinate S orders fitted G(S), exact P=0.017",
        "nphys_two_scale_response_collapse_metrics.csv",
        "test=route_severity_orders_susceptibility",
        "p_value",
        0.017,
        0.0005,
        "Exact five-route Spearman permutation check for route severity ordering loop-to-overload gain.",
        lambda df: float(one(df, test="route_severity_orders_susceptibility")["p_value"]),
    ),
    Claim(
        "main_two_scale_full_model_loro",
        "Main two-scale mechanism paragraph",
        "full two-scale model gave leave-one-route R^2=0.86",
        "nphys_two_scale_response_collapse_model_tests.csv",
        "model=loop + severity + SxL",
        "r2_vs_mean",
        0.86,
        0.005,
        "Huber leave-one-route diagnostic for slow susceptibility plus fast loop drive.",
        lambda df: float(one(df, model="loop + severity + SxL")["r2_vs_mean"]),
    ),
    Claim(
        "main_two_scale_top5_tail_control_loro",
        "Main two-scale mechanism paragraph",
        "top-5% force-tail control gave leave-one-route R^2=0.65",
        "nphys_two_scale_response_collapse_model_tests.csv",
        "model=top-5 force tail",
        "r2_vs_mean",
        0.65,
        0.005,
        "Tail-only transfer control in the two-scale response-collapse audit.",
        lambda df: float(one(df, model="top-5 force tail")["r2_vs_mean"]),
    ),
    Claim(
        "main_two_scale_gain_normalized_not_universal",
        "Main two-scale mechanism paragraph",
        "dividing by G(S) did not produce a universal one-slope law",
        "nphys_two_scale_response_collapse_metrics.csv",
        "test=susceptibility_normalized_collapse",
        "value",
        0.067,
        0.005,
        "Low Spearman rho after gain normalization bounds the route-conditioned diagnostic.",
        lambda df: float(one(df, test="susceptibility_normalized_collapse")["value"]),
    ),
    Claim(
        "main_phase_displacement_downward_exact_p",
        "Main return-map paragraph",
        "downward hot-coordinate displacement co-orders overload/gain/severity, exact P=0.017",
        "nphys_phase_displacement_geometry_correlations.csv",
        "relationship=downward hot displacement vs mean overload",
        "exact_p_two_sided",
        0.017,
        0.0005,
        "Exact five-route Spearman permutation check for the Fig. 5a displacement geometry.",
        lambda df: float(one(df, relationship="downward hot displacement vs mean overload")["exact_p_two_sided"]),
    ),
    Claim(
        "main_breathing_loop_activation_rho_raw",
        "Main response-function paragraph",
        "positive loop activation orders hot overload across routes, rho=0.919",
        "nphys_breathing_quality_factor_correlations.csv",
        "predictor=loop_activation_positive; target=overload_asinh",
        "spearman_raw",
        0.919,
        0.0005,
        "Breathing hierarchy in the lagged three-route subset.",
        lambda df: float(one(df, predictor="loop_activation_positive", target="overload_asinh")["spearman_raw"]),
    ),
    Claim(
        "main_breathing_loop_activation_rho_centered",
        "Main response-function paragraph",
        "positive loop activation remains strong after route centring, rho=0.909",
        "nphys_breathing_quality_factor_correlations.csv",
        "predictor=loop_activation_positive; target=overload_asinh",
        "spearman_within_route",
        0.909,
        0.0005,
        "Centred response hierarchy.",
        lambda df: float(one(df, predictor="loop_activation_positive", target="overload_asinh")["spearman_within_route"]),
    ),
    Claim(
        "main_hazard_rho_centered",
        "Main response-function paragraph",
        "breathing hazard retained route-centred predictive power, rho=0.800",
        "nphys_breathing_quality_factor_correlations.csv",
        "predictor=log_hazard_number; target=overload_asinh",
        "spearman_within_route",
        0.800,
        0.0005,
        "Hazard is a modulation/observability coordinate, not a replacement law.",
        lambda df: float(one(df, predictor="log_hazard_number", target="overload_asinh")["spearman_within_route"]),
    ),
    Claim(
        "main_hazard_circular_shift_p",
        "Main response-function paragraph",
        "breathing hazard survived route-wise circular-shift nulls, P=0.0012",
        "nphys_breathing_quality_factor_circular_shift_null.csv",
        "two_sided_p constant over permutations",
        "two_sided_p",
        0.0012,
        0.00005,
        "Null table stores the same observed P on each permutation row.",
        lambda df: first_value(df, "two_sided_p"),
    ),
    Claim(
        "main_next_cold_fabric_lagged_imprint",
        "Main lagged prediction paragraph",
        "cold-to-hot fabric excursions predicted next-cold fabric imprints, rho=0.48",
        "nphys_breathing_cycle_correlations.csv",
        "relationship=inhaled_geometry_to_next_cold_fabric",
        "spearman_within_regime_centered",
        0.48,
        0.005,
        "Lagged breathing imprint, route-centred.",
        lambda df: float(one(df, relationship="inhaled_geometry_to_next_cold_fabric")["spearman_within_regime_centered"]),
    ),
    Claim(
        "main_next_cold_fabric_lagged_imprint_p",
        "Main lagged prediction paragraph",
        "cold-to-hot fabric excursions predicted next-cold fabric imprints, P=3.2e-6",
        "nphys_breathing_cycle_correlations.csv",
        "relationship=inhaled_geometry_to_next_cold_fabric",
        "p_within_regime_centered",
        3.2e-6,
        1.0e-7,
        "Lagged breathing imprint P value.",
        lambda df: float(one(df, relationship="inhaled_geometry_to_next_cold_fabric")["p_within_regime_centered"]),
    ),
    Claim(
        "main_next_cold_loop_memory_lagged_imprint",
        "Main lagged prediction paragraph",
        "positive graph cycles predicted next-cold loop-memory imprints, rho=0.55",
        "nphys_breathing_cycle_correlations.csv",
        "relationship=inhaled_positive_cycles_to_next_cold_loop_memory",
        "spearman_within_regime_centered",
        0.55,
        0.005,
        "Lagged loop-memory imprint, route-centred.",
        lambda df: float(one(df, relationship="inhaled_positive_cycles_to_next_cold_loop_memory")["spearman_within_regime_centered"]),
    ),
    Claim(
        "main_next_cold_loop_memory_lagged_imprint_p",
        "Main lagged prediction paragraph",
        "positive graph cycles predicted next-cold loop-memory imprints, P=4.5e-8",
        "nphys_breathing_cycle_correlations.csv",
        "relationship=inhaled_positive_cycles_to_next_cold_loop_memory",
        "p_within_regime_centered",
        4.5e-8,
        1.0e-9,
        "Lagged loop-memory imprint P value.",
        lambda df: float(one(df, relationship="inhaled_positive_cycles_to_next_cold_loop_memory")["p_within_regime_centered"]),
    ),
    Claim(
        "main_breath_amplitude_to_overload",
        "Main lagged prediction paragraph",
        "larger inhalation amplitude correlated with overload, rho=0.83",
        "nphys_breathing_parameter_effects_correlations.csv",
        "predictor=breath_amplitude; target=force_p99_hot_minus_cold",
        "spearman_within_regime",
        0.83,
        0.005,
        "Breathing parameter effect after route centring.",
        lambda df: float(one(df, predictor="breath_amplitude", target="force_p99_hot_minus_cold")["spearman_within_regime"]),
    ),
    Claim(
        "main_imprint_efficiency_to_overload",
        "Main lagged prediction paragraph",
        "imprint efficiency was anti-correlated with overload, rho=-0.44",
        "nphys_breathing_parameter_effects_correlations.csv",
        "predictor=imprint_efficiency; target=force_p99_hot_minus_cold",
        "spearman_within_regime",
        -0.44,
        0.005,
        "Buffering language is supported as an anti-correlation, not as proof of causality.",
        lambda df: float(one(df, predictor="imprint_efficiency", target="force_p99_hot_minus_cold")["spearman_within_regime"]),
    ),
    Claim(
        "main_lyapunov_storage_alone_transfer",
        "Main return-map paragraph",
        "Lyapunov storage alone did not transfer overload, R^2=0.002",
        "nphys_lyapunov_storage_transfer_tests.csv",
        "model=lyapunov_storage",
        "r2_vs_mean",
        0.002,
        0.0005,
        "Storage is not the overload coordinate.",
        lambda df: float(one(df, model="lyapunov_storage")["r2_vs_mean"]),
    ),
    Claim(
        "main_storage_plus_psi_transfer",
        "Main return-map paragraph",
        "Lyapunov storage plus Psi improved leave-one-route response prediction to R^2=0.81",
        "nphys_lyapunov_storage_transfer_tests.csv",
        "model=storage_plus_Psi",
        "r2_vs_mean",
        0.81,
        0.005,
        "Storage supports dissipative-state language when paired with Psi.",
        lambda df: float(one(df, model="storage_plus_Psi")["r2_vs_mean"]),
    ),
    Claim(
        "main_rare_event_loop_auc",
        "Main force-loop onset paragraph",
        "force-loop activation identified the top-20% overload tail with AUC 0.90",
        "nphys_force_loop_rare_event_metrics.csv",
        "predictor=force-loop activation",
        "auc",
        0.90,
        0.005,
        "Route-local rare-event readout; finite-route diagnostic, not a large-deviation rate function.",
        lambda df: float(one(df, predictor="force-loop activation")["auc"]),
    ),
    Claim(
        "main_rare_event_loop_top_quintile_risk",
        "Main force-loop onset paragraph",
        "force-loop activation raised the top-quintile tail risk to 0.67",
        "nphys_force_loop_rare_event_metrics.csv",
        "predictor=force-loop activation",
        "top_quintile_risk",
        0.67,
        0.005,
        "Conditional extreme-overload risk in the top force-loop quintile.",
        lambda df: float(one(df, predictor="force-loop activation")["top_quintile_risk"]),
    ),
    Claim(
        "main_rare_event_tail_inverse_control_auc",
        "Main force-loop onset paragraph",
        "top-5% force-tail surrogate was oppositely signed",
        "nphys_force_loop_rare_event_metrics.csv",
        "predictor=top-5% force tail",
        "auc",
        0.21,
        0.005,
        "AUC below 0.5 verifies inverse-control wording for the force-tail surrogate.",
        lambda df: float(one(df, predictor="top-5% force tail")["auc"]),
    ),
    Claim(
        "main_rare_event_lag2_loop_auc",
        "Main return-map precursor paragraph",
        "lag-2 loop activation predicted route-local rare overload with AUC 0.91",
        "nphys_force_loop_rare_event_precursor_lag_scan.csv",
        "predictor=lagged loop; lag_cycles=2",
        "auc",
        0.91,
        0.005,
        "Bounded early-warning statement; not a universal one-cycle alarm.",
        lambda df: float(one(df, predictor="lagged loop", lag_cycles="2")["auc"]),
    ),
    Claim(
        "main_rare_event_lag2_loop_top_quintile_risk",
        "Main return-map precursor paragraph",
        "lag-2 loop activation raised top-quintile two-cycle risk to 0.57",
        "nphys_force_loop_rare_event_precursor_conditional_risk.csv",
        "predictor=lagged loop; quintile=5",
        "rare_event_probability",
        0.57,
        0.005,
        "Two-cycle conditional risk from lagged force-loop activation.",
        lambda df: float(one(df, predictor="lagged loop", quintile="5")["rare_event_probability"]),
    ),
    Claim(
        "main_breathing_memory_loop_kernel_loro",
        "Main return-map precursor paragraph",
        "lag-0--4 loop kernel predicted held-out-route overload with R^2=0.64",
        "nphys_breathing_memory_kernel_transfer_tests.csv",
        "model=loop kernel; validation=leave_one_route_out",
        "r2_vs_training_mean",
        0.64,
        0.005,
        "Finite-memory diagnostic; not a universal frequency law.",
        lambda df: float(one(df, model="loop kernel", validation="leave_one_route_out")["r2_vs_training_mean"]),
    ),
    Claim(
        "main_breathing_memory_tail_kernel_loro",
        "Main return-map precursor paragraph",
        "top-5% force-tail kernel transferred negatively with R^2=-0.60",
        "nphys_breathing_memory_kernel_transfer_tests.csv",
        "model=tail kernel; validation=leave_one_route_out",
        "r2_vs_training_mean",
        -0.60,
        0.005,
        "Force-tail distributed-lag control remains insufficient.",
        lambda df: float(one(df, model="tail kernel", validation="leave_one_route_out")["r2_vs_training_mean"]),
    ),
    Claim(
        "main_normal_form_r3_dominant_eigenvalue",
        "Main return-map paragraph",
        "near-onset route has weakly damped flip-mode dominant eigenvalue -0.94",
        "nphys_return_map_normal_form_route_metrics.csv",
        "regime_id=R3",
        "dominant_eigenvalue",
        -0.94,
        0.005,
        "Normal-form reserve audit; consistency bridge to finite-cycle memory, not a bifurcation claim.",
        lambda df: float(one(df, regime_id="R3")["dominant_eigenvalue"]),
    ),
    Claim(
        "main_normal_form_loop_even_lag_mean",
        "Main return-map paragraph",
        "loop activation even-lag autocorrelation mean pooled rho=0.75",
        "nphys_return_map_normal_form_parity_summary.csv",
        "series=loop activation; parity=even",
        "mean_pooled_spearman",
        0.75,
        0.005,
        "Even-lag bridge from breathing memory-kernel autocorrelation.",
        lambda df: float(one(df, series="loop activation", parity="even")["mean_pooled_spearman"]),
    ),
    Claim(
        "main_normal_form_overload_even_lag_mean",
        "Main return-map paragraph",
        "overload even-lag autocorrelation mean pooled rho=0.71",
        "nphys_return_map_normal_form_parity_summary.csv",
        "series=overload; parity=even",
        "mean_pooled_spearman",
        0.71,
        0.005,
        "Even-lag overload memory bridge.",
        lambda df: float(one(df, series="overload", parity="even")["mean_pooled_spearman"]),
    ),
    Claim(
        "main_normal_form_loop_odd_lag_mean",
        "Main return-map paragraph",
        "loop activation odd-lag autocorrelation mean pooled rho=-0.32",
        "nphys_return_map_normal_form_parity_summary.csv",
        "series=loop activation; parity=odd",
        "mean_pooled_spearman",
        -0.32,
        0.005,
        "Odd-lag contrast for finite-cycle rhythm.",
        lambda df: float(one(df, series="loop activation", parity="odd")["mean_pooled_spearman"]),
    ),
    Claim(
        "main_normal_form_overload_odd_lag_mean",
        "Main return-map paragraph",
        "overload odd-lag autocorrelation mean pooled rho=-0.24",
        "nphys_return_map_normal_form_parity_summary.csv",
        "series=overload; parity=odd",
        "mean_pooled_spearman",
        -0.24,
        0.005,
        "Odd-lag overload contrast for finite-cycle rhythm.",
        lambda df: float(one(df, series="overload", parity="odd")["mean_pooled_spearman"]),
    ),
]


def audit_claims() -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for claim in CLAIMS:
        status = "pass"
        source_value = float("nan")
        error = ""
        try:
            df = pd.read_csv(SRC / claim.source_file)
            source_value = claim.extractor(df)
            if abs(source_value - claim.claimed_value) > claim.tolerance:
                status = "mismatch"
        except Exception as exc:  # The report should expose broken selectors.
            status = "error"
            error = str(exc)
        rows.append(
            {
                "claim_id": claim.claim_id,
                "location": claim.location,
                "claim_text": claim.claim_text,
                "source_file": f"source_data/{claim.source_file}",
                "source_selector": claim.source_selector,
                "value_name": claim.value_name,
                "claimed_value": claim.claimed_value,
                "source_value": source_value,
                "abs_error": abs(source_value - claim.claimed_value) if status != "error" else float("nan"),
                "tolerance": claim.tolerance,
                "status": status,
                "note": claim.note,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame) -> None:
    passed = int((df["status"] == "pass").sum())
    mismatched = int((df["status"] == "mismatch").sum())
    errored = int((df["status"] == "error").sum())
    lines = [
        "# Nature Physics numerical claim audit",
        "",
        "This audit checks high-leverage numerical claims in the main manuscript against source-data CSV files.",
        "It is intentionally curated: it targets values that affect the physical story, not every literal number in the paper.",
        "",
        f"- Claims checked: {len(df)}",
        f"- Passed: {passed}",
        f"- Mismatched: {mismatched}",
        f"- Errors: {errored}",
        "",
        "## Important scope notes",
        "",
        "- `R^2=0.82` refers to the asinh-overload minimal-state hierarchy.",
        "- `R^2=0.62` refers to the raw-overload dimensionless collapse audit.",
        "- The breathing hazard is audited as a modulation/observability coordinate, not as a replacement for force-loop activation.",
        "- P values and rounded correlations are checked at the precision used in the manuscript.",
        "",
        "## Claim table",
        "",
        "| claim_id | status | claimed | source | source file | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| `{row.claim_id}` | {row.status} | {row.claimed_value:.6g} | "
            f"{row.source_value:.6g} | `{row.source_file}` | {row.note} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = audit_claims()
    SRC.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    write_markdown(df)
    bad = df[df["status"] != "pass"]
    print(f"Checked {len(df)} numerical claims.")
    print(df["status"].value_counts().to_string())
    if len(bad):
        print("\nNon-passing claims:")
        print(bad[["claim_id", "status", "claimed_value", "source_value", "error"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
