#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

REGIME_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R6": "#C95F3F"}
REGIME_MARKERS = {"R1": "o", "R3": "s", "R6": "^"}
GRID = "#E8EBEF"
INK = "#252A31"
MUTED = "#818994"
HOT = "#B6423E"
COLD = "#3D6B9C"


STATE_COLS = [
    "Z_geom",
    "force_p99",
    "force_h1_birth_force_share",
    "force_share_top5_edges",
    "cycle_birth_positive_fraction",
    "orientation_entropy",
    "force_proxy_gini",
    "net_force_p99_N",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def zscore(values: pd.Series) -> pd.Series:
    sigma = values.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return values * 0.0
    return (values - values.mean()) / sigma


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def add_zero_lines(ax: plt.Axes) -> None:
    ax.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)), zorder=1)
    ax.axvline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)), zorder=1)


def scatter_by_regime(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    alpha: float = 0.86,
    s: float = 22.0,
) -> None:
    for rid, g in data.groupby("regime_id", sort=True):
        ax.scatter(
            g[x],
            g[y],
            s=s,
            marker=REGIME_MARKERS.get(rid, "o"),
            facecolor=REGIME_COLORS.get(rid, "#555555"),
            edgecolor="white",
            lw=0.45,
            alpha=alpha,
            label=rid,
            zorder=3,
        )


def spearman_pair(
    data: pd.DataFrame,
    predictor: str,
    target: str,
    *,
    centered: bool,
) -> tuple[float, float, int]:
    df = data[[predictor, target, "regime_id"]].replace([np.inf, -np.inf], np.nan).dropna()
    if centered:
        df[predictor] = df[predictor] - df.groupby("regime_id")[predictor].transform("mean")
        df[target] = df[target] - df.groupby("regime_id")[target].transform("mean")
    if len(df) < 5:
        return np.nan, np.nan, int(len(df))
    stat = spearmanr(df[predictor], df[target], nan_policy="omit")
    return float(stat.statistic), float(stat.pvalue), int(len(df))


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        (
            "inhaled_loop_activation_to_hot_overload",
            "force_h1_birth_force_share_hot_minus_cold",
            "force_p99_hot_minus_cold",
            "primary",
        ),
        (
            "inhaled_positive_cycle_creation_to_hot_overload",
            "cycle_birth_positive_fraction_inhale_delta",
            "force_p99_hot_minus_cold",
            "primary",
        ),
        (
            "top5_concentration_to_hot_overload",
            "force_share_top5_edges_hot_minus_cold",
            "force_p99_hot_minus_cold",
            "negative_control_like",
        ),
        (
            "inhaled_geometry_to_next_cold_fabric",
            "Z_geom_inhale_delta",
            "Z_geom_next_cold_minus_current",
            "lagged_memory",
        ),
        (
            "inhaled_positive_cycles_to_next_cold_loop_memory",
            "cycle_birth_positive_fraction_inhale_delta",
            "force_h1_birth_force_share_next_cold_minus_current",
            "lagged_memory",
        ),
        (
            "inhaled_loop_activation_to_next_cold_loop_memory",
            "force_h1_birth_force_share_hot_minus_cold",
            "force_h1_birth_force_share_next_cold_minus_current",
            "lagged_memory",
        ),
        (
            "contact_aperture_to_hot_overload_supporting",
            "breathing_aperture",
            "force_p99_hot_minus_cold",
            "supporting_contact_subset",
        ),
        (
            "contact_creation_to_next_cold_fabric_supporting",
            "created_fraction_of_hot",
            "Z_geom_next_cold_minus_current",
            "supporting_contact_subset",
        ),
        (
            "next_cold_retention_to_hot_overload",
            "next_cold_cycle_jaccard",
            "force_p99_hot_minus_cold",
            "diagnostic_not_causal",
        ),
    ]
    rows = []
    for name, predictor, target, evidence_class in pairs:
        raw_r, raw_p, n_raw = spearman_pair(data, predictor, target, centered=False)
        cen_r, cen_p, n_cen = spearman_pair(data, predictor, target, centered=True)
        rows.append(
            {
                "relationship": name,
                "predictor": predictor,
                "target": target,
                "evidence_class": evidence_class,
                "spearman_raw": raw_r,
                "p_raw": raw_p,
                "n_raw": n_raw,
                "spearman_within_regime_centered": cen_r,
                "p_within_regime_centered": cen_p,
                "n_within_regime_centered": n_cen,
            }
        )
    return pd.DataFrame(rows)


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contact = pd.read_csv(SRC / "nphys_contact_persistence_metrics.csv")
    edge = pd.read_csv(SRC / "nphys_deep_edge_persistence_decomposition.csv")
    joined = pd.read_csv(SRC / "nphys_long_cycle_true_force_geometry_join.csv")
    delta = pd.read_csv(SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv")

    run_regime = joined[["run", "regime_id"]].drop_duplicates()

    same_cycle = contact[contact["transition"] == "cold_to_hot_same_cycle"].copy()
    same_cycle = same_cycle.merge(run_regime, on="run", how="left")
    same_cycle["breathing_aperture"] = 1.0 - same_cycle["jaccard"]
    same_cycle = same_cycle[
        [
            "run",
            "regime_id",
            "cycle",
            "jaccard",
            "breathing_aperture",
            "survival_fraction_of_cold",
            "created_fraction_of_hot",
        ]
    ].rename(columns={"jaccard": "cold_hot_contact_jaccard"})

    state = joined.pivot_table(index=["run", "regime_id", "cycle"], columns="phase", values=STATE_COLS)
    state.columns = [f"{name}_{phase}" for name, phase in state.columns]
    state = state.reset_index()
    for name in STATE_COLS:
        state[f"{name}_inhale_delta"] = state[f"{name}_hot"] - state[f"{name}_cold"]

    cold = joined[joined["phase"] == "cold"][["run", "regime_id", "cycle", *STATE_COLS]].copy()
    cold = cold.sort_values(["run", "cycle"])
    for name in STATE_COLS:
        cold[f"{name}_next_cold_minus_current"] = cold.groupby("run")[name].shift(-1) - cold[name]
    next_cols = ["run", "regime_id", "cycle", *[f"{name}_next_cold_minus_current" for name in STATE_COLS]]
    state = state.merge(cold[next_cols], on=["run", "regime_id", "cycle"], how="left")

    for name in STATE_COLS:
        denom = state[f"{name}_inhale_delta"].abs() + 1e-12
        state[f"{name}_rectification_fraction"] = state[f"{name}_next_cold_minus_current"] / denom

    metrics = state.merge(delta, on=["run", "regime_id", "cycle"], how="left", suffixes=("", "_delta_file"))
    metrics = metrics.merge(same_cycle, on=["run", "regime_id", "cycle"], how="left")

    for phase in ["cold", "hot"]:
        phase_edges = edge[edge["phase"] == phase].copy().merge(run_regime, on="run", how="left")
        phase_edges = phase_edges[
            [
                "run",
                "regime_id",
                "cycle",
                "jaccard",
                "persistent_fraction_of_current",
                "created_fraction_of_current",
                "broken_fraction_of_previous",
            ]
        ].rename(
            columns={
                "jaccard": f"{phase}_cycle_jaccard",
                "persistent_fraction_of_current": f"{phase}_cycle_persistent_fraction_of_current",
                "created_fraction_of_current": f"{phase}_cycle_created_fraction_of_current",
                "broken_fraction_of_previous": f"{phase}_cycle_broken_fraction_of_previous",
            }
        )
        metrics = metrics.merge(phase_edges, on=["run", "regime_id", "cycle"], how="left")

        next_phase_edges = phase_edges.copy()
        next_phase_edges["cycle"] = next_phase_edges["cycle"] - 1
        next_phase_edges = next_phase_edges.rename(
            columns={
                col: f"next_{col}"
                for col in next_phase_edges.columns
                if col not in ["run", "regime_id", "cycle"]
            }
        )
        metrics = metrics.merge(next_phase_edges, on=["run", "regime_id", "cycle"], how="left")

    metrics = metrics[metrics["cycle"] <= 29].copy()

    regime_summary = (
        metrics.groupby("regime_id")
        .agg(
            n_cycles=("cycle", "count"),
            mean_contact_breathing_aperture=("breathing_aperture", "mean"),
            mean_contact_creation_fraction=("created_fraction_of_hot", "mean"),
            mean_loop_activation=("force_h1_birth_force_share_hot_minus_cold", "mean"),
            mean_positive_cycle_creation_inhale=("cycle_birth_positive_fraction_inhale_delta", "mean"),
            mean_hot_overload=("force_p99_hot_minus_cold", "mean"),
            mean_next_cold_force_imprint=("force_p99_next_cold_minus_current", "mean"),
            mean_next_cold_fabric_imprint=("Z_geom_next_cold_minus_current", "mean"),
            mean_loop_memory_imprint=("force_h1_birth_force_share_next_cold_minus_current", "mean"),
            mean_Z_rectification_fraction=("Z_geom_rectification_fraction", "mean"),
            mean_force_rectification_fraction=("force_p99_rectification_fraction", "mean"),
            mean_loop_rectification_fraction=("force_h1_birth_force_share_rectification_fraction", "mean"),
            mean_next_cold_retention=("next_cold_cycle_jaccard", "mean"),
        )
        .reset_index()
    )

    return metrics, correlation_table(metrics), regime_summary


def add_regression_annotation(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    loc: tuple[float, float],
    centered: bool = True,
) -> None:
    rho, p, n = spearman_pair(data, x, y, centered=centered)
    label = rf"$\rho_{{\rm within}}={rho:.2f}$" + f"\nP={p:.1e}, n={n}"
    ax.text(
        loc[0],
        loc[1],
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.8},
    )


def plot_breathing_figure(metrics: pd.DataFrame, regime_summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)

    plot_data = metrics.copy()
    for col in [
        "Z_geom_cold",
        "Z_geom_hot",
        "force_h1_birth_force_share_cold",
        "force_h1_birth_force_share_hot",
    ]:
        plot_data[f"{col}_z"] = zscore(plot_data[col])
    centered_cols = [
        "force_h1_birth_force_share_hot_minus_cold",
        "force_p99_hot_minus_cold",
        "Z_geom_inhale_delta",
        "Z_geom_next_cold_minus_current",
        "cycle_birth_positive_fraction_inhale_delta",
        "force_h1_birth_force_share_next_cold_minus_current",
    ]
    for col in centered_cols:
        plot_data[f"{col}_wc"] = plot_data[col] - plot_data.groupby("regime_id")[col].transform("mean")

    fig = plt.figure(figsize=(7.2, 5.35))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel(ax_a, "a")
    for rid, g in plot_data.groupby("regime_id", sort=True):
        g = g.sort_values("cycle")
        color = REGIME_COLORS[rid]
        ax_a.plot(g["Z_geom_cold_z"], g["force_h1_birth_force_share_cold_z"], color=color, lw=0.7, alpha=0.35)
        ax_a.scatter(
            g["Z_geom_cold_z"],
            g["force_h1_birth_force_share_cold_z"],
            s=14,
            facecolor="white",
            edgecolor=color,
            lw=0.65,
            alpha=0.85,
            zorder=3,
        )
        ax_a.scatter(
            g["Z_geom_hot_z"],
            g["force_h1_birth_force_share_hot_z"],
            s=15,
            marker=REGIME_MARKERS[rid],
            facecolor=color,
            edgecolor="white",
            lw=0.35,
            alpha=0.88,
            label=rid,
            zorder=4,
        )
        for _, row in g.iloc[::4].iterrows():
            ax_a.annotate(
                "",
                xy=(row["Z_geom_hot_z"], row["force_h1_birth_force_share_hot_z"]),
                xytext=(row["Z_geom_cold_z"], row["force_h1_birth_force_share_cold_z"]),
                arrowprops={"arrowstyle": "->", "lw": 0.55, "color": color, "alpha": 0.55, "shrinkA": 1.5, "shrinkB": 1.5},
            )
    add_zero_lines(ax_a)
    ax_a.set_xlabel("fabric coordinate, $Z$ (z-score)")
    ax_a.set_ylabel("loop-force coordinate (z-score)")
    ax_a.set_title("phase-space breathing trajectory", loc="left", pad=2)
    ax_a.legend(ncol=3, loc="lower left", handletextpad=0.2, columnspacing=0.85)
    finish(ax_a)

    panel(ax_b, "b")
    scatter_by_regime(
        ax_b,
        plot_data,
        "force_h1_birth_force_share_hot_minus_cold_wc",
        "force_p99_hot_minus_cold_wc",
    )
    x = plot_data["force_h1_birth_force_share_hot_minus_cold_wc"]
    y = plot_data["force_p99_hot_minus_cold_wc"]
    ok = np.isfinite(x) & np.isfinite(y)
    coef = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
    ax_b.plot(xx, coef[0] * xx + coef[1], color=INK, lw=0.75, alpha=0.8)
    add_zero_lines(ax_b)
    add_regression_annotation(
        ax_b,
        metrics,
        "force_h1_birth_force_share_hot_minus_cold",
        "force_p99_hot_minus_cold",
        loc=(0.05, 0.96),
    )
    ax_b.set_xlabel("regime-centred inhaled loop activation\n$\\Delta$H1 force share")
    ax_b.set_ylabel("regime-centred hot overload\n$\\Delta$force p99")
    ax_b.set_title("hot overload follows loop activation", loc="left", pad=2)
    finish(ax_b)

    panel(ax_c, "c")
    scatter_by_regime(
        ax_c,
        plot_data,
        "Z_geom_inhale_delta_wc",
        "Z_geom_next_cold_minus_current_wc",
    )
    x = plot_data["Z_geom_inhale_delta_wc"]
    y = plot_data["Z_geom_next_cold_minus_current_wc"]
    ok = np.isfinite(x) & np.isfinite(y)
    coef = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
    ax_c.plot(xx, coef[0] * xx + coef[1], color=INK, lw=0.75, alpha=0.8)
    add_zero_lines(ax_c)
    add_regression_annotation(
        ax_c,
        metrics,
        "Z_geom_inhale_delta",
        "Z_geom_next_cold_minus_current",
        loc=(0.05, 0.96),
    )
    ax_c.set_xlabel("regime-centred inhaled fabric excursion\n$\\Delta Z_{{hot-cold}}$")
    ax_c.set_ylabel("regime-centred next-cold fabric imprint\n$Z_{{cold},n+1}-Z_{{cold},n}$")
    ax_c.set_title("exhalation leaves a fabric trace", loc="left", pad=2)
    finish(ax_c)

    panel(ax_d, "d")
    scatter_by_regime(
        ax_d,
        plot_data,
        "cycle_birth_positive_fraction_inhale_delta_wc",
        "force_h1_birth_force_share_next_cold_minus_current_wc",
    )
    x = plot_data["cycle_birth_positive_fraction_inhale_delta_wc"]
    y = plot_data["force_h1_birth_force_share_next_cold_minus_current_wc"]
    ok = np.isfinite(x) & np.isfinite(y)
    coef = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
    ax_d.plot(xx, coef[0] * xx + coef[1], color=INK, lw=0.75, alpha=0.8)
    add_zero_lines(ax_d)
    add_regression_annotation(
        ax_d,
        metrics,
        "cycle_birth_positive_fraction_inhale_delta",
        "force_h1_birth_force_share_next_cold_minus_current",
        loc=(0.05, 0.96),
    )
    ax_d.set_xlabel("regime-centred inhaled positive-cycle creation")
    ax_d.set_ylabel("regime-centred next-cold loop-memory imprint")
    ax_d.set_title("loop sector carries lagged memory", loc="left", pad=2)
    finish(ax_d)

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig11_breathing_cycle_coupling.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    # A compact phase-space area diagnostic is useful for audits, but not shown in the main figure.
    rows = []
    phase = plot_data.copy()
    for rid, g in phase.groupby("regime_id", sort=True):
        ordered_x: list[float] = []
        ordered_y: list[float] = []
        for _, row in g.sort_values("cycle").iterrows():
            ordered_x.extend([row["Z_geom_cold_z"], row["Z_geom_hot_z"]])
            ordered_y.extend([row["force_h1_birth_force_share_cold_z"], row["force_h1_birth_force_share_hot_z"]])
        rows.append(
            {
                "regime_id": rid,
                "signed_state_space_area": polygon_area(np.asarray(ordered_x), np.asarray(ordered_y)),
                "absolute_state_space_area": abs(polygon_area(np.asarray(ordered_x), np.asarray(ordered_y))),
            }
        )
    pd.DataFrame(rows).merge(regime_summary, on="regime_id", how="left").to_csv(
        SRC / "nphys_breathing_cycle_regime_phase_area.csv", index=False
    )


def write_report(metrics: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    primary = corr[corr["evidence_class"].isin(["primary", "lagged_memory"])].copy()
    support = corr[corr["evidence_class"].str.contains("supporting", na=False)].copy()
    lines = [
        "# Breathing-cycle coupling audit",
        "",
        "This audit tests whether the breathing language is backed by phase-resolved data, rather than only by rewritten prose.",
        "",
        "## Operational definitions",
        "",
        "- Inhalation: same-cycle cold-to-hot excursion of fabric and force-loop coordinates.",
        "- Hot overload: hot-minus-cold force p99.",
        "- Exhalation imprint: next cold state minus current cold state.",
        "- Contact aperture: 1 minus cold-to-hot contact Jaccard; available only for a smaller contact-persistence subset and therefore treated as supporting evidence.",
        "",
        "## Main grounded findings",
        "",
        "- The strongest relation remains the hot-state one: inhaled force-loop activation tracks overload after regime centering.",
        "- Geometry is not merely decorative: the cold-to-hot fabric excursion predicts the next-cold fabric imprint after regime centering.",
        "- Positive-cycle creation during inhalation predicts a next-cold loop-memory imprint, supporting a lagged network-memory story.",
        "- Contact-aperture metrics are suggestive but sparse; they should be used as supporting diagnostics, not as the main quantitative pillar.",
        "",
        "## Primary and lagged correlations",
        "",
        primary[
            [
                "relationship",
                "spearman_raw",
                "p_raw",
                "spearman_within_regime_centered",
                "p_within_regime_centered",
                "n_within_regime_centered",
            ]
        ].to_markdown(index=False),
        "",
        "## Contact-subset diagnostics",
        "",
        support[
            [
                "relationship",
                "spearman_raw",
                "p_raw",
                "spearman_within_regime_centered",
                "p_within_regime_centered",
                "n_within_regime_centered",
            ]
        ].to_markdown(index=False),
        "",
        "## Regime means",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "The data support a phase-resolved breathing-cycle mechanism: hot expansion excites a loop sector that controls overload, while part of the fabric/loop excursion is rectified into the next cold state. The evidence does not yet prove a universal granular law or a single-cycle causal mechanism for every observable. The manuscript should therefore present this as a measured cycle map and a falsifiable organizing principle.",
        "",
        "## Generated files",
        "",
        "- `source_data/nphys_breathing_cycle_metrics.csv`",
        "- `source_data/nphys_breathing_cycle_correlations.csv`",
        "- `source_data/nphys_breathing_cycle_regime_summary.csv`",
        "- `source_data/nphys_breathing_cycle_regime_phase_area.csv`",
        "- `figures/nphys_fig11_breathing_cycle_coupling.{svg,pdf,png,tiff}`",
    ]
    (ROOT / "nature_physics_breathing_cycle_coupling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metrics, corr, summary = build_metrics()
    SRC.mkdir(exist_ok=True)
    metrics.to_csv(SRC / "nphys_breathing_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_cycle_correlations.csv", index=False)
    summary.to_csv(SRC / "nphys_breathing_cycle_regime_summary.csv", index=False)
    plot_breathing_figure(metrics, summary)
    write_report(metrics, corr, summary)
    print(f"Wrote {len(metrics)} cycle-pair rows")
    print(corr[["relationship", "spearman_within_regime_centered", "p_within_regime_centered", "n_within_regime_centered"]].to_string(index=False))


if __name__ == "__main__":
    main()
