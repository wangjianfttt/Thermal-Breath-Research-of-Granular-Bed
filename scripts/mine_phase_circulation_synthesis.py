#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

ROUTES = ["R1", "R3", "R5", "R6", "R6c"]
COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
ACCENT = "#8D3138"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=INK)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "geometry": pd.read_csv(SRC / "nphys_phase_circulation_geometry.csv"),
        "geometry_corr": pd.read_csv(SRC / "nphys_phase_circulation_geometry_correlations.csv"),
        "writing_summary": pd.read_csv(SRC / "nphys_phase_circulation_writing_time_summary.csv"),
        "writing_curves": pd.read_csv(SRC / "nphys_phase_circulation_writing_time_curves.csv"),
        "writing_corr": pd.read_csv(SRC / "nphys_phase_circulation_writing_time_correlations.csv"),
        "order_null_rho": pd.read_csv(SRC / "nphys_phase_circulation_order_null_rho.csv"),
        "order_null_summary": pd.read_csv(SRC / "nphys_phase_circulation_order_null_summary.csv"),
        "coord_corr": pd.read_csv(SRC / "nphys_phase_circulation_coordinate_robustness_correlations.csv"),
    }


def synthesis_metrics(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    geom = tables["geometry"].set_index("regime_id").loc[ROUTES].reset_index()
    write = tables["writing_summary"].set_index("regime_id").loc[ROUTES].reset_index()
    route = geom[
        [
            "regime_id",
            "signed_circulation_area",
            "clockwise_circulation",
            "trajectory_arc_length",
            "abs_circulation_area",
            "mean_overload",
            "loop_gain",
            "S",
        ]
    ].merge(
        write[["regime_id", "half_area_cycle", "early_fraction", "middle_fraction", "late_fraction"]],
        on="regime_id",
    )
    geometry_corr = tables["geometry_corr"]
    writing_corr = tables["writing_corr"]
    order_null = tables["order_null_summary"].iloc[0]
    coord_corr = tables["coord_corr"]
    closed = coord_corr[(coord_corr["target"] == "mean_overload") & (coord_corr["geometry_metric"] == "closed_signed_area")]
    robust = closed[~closed["coordinate_family"].str.contains("nonlinear")]
    rank = closed[closed["coordinate_label"] == "route rank"].iloc[0]
    controls = pd.DataFrame(
        [
            {
                "control": "circulation orientation",
                "metric": "signed area vs overload",
                "value": float(
                    geometry_corr[
                        (geometry_corr["x"] == "signed_circulation_area") & (geometry_corr["y"] == "mean_overload")
                    ]["spearman"].iloc[0]
                ),
                "p_value": float(
                    geometry_corr[
                        (geometry_corr["x"] == "signed_circulation_area") & (geometry_corr["y"] == "mean_overload")
                    ]["exact_p_two_sided"].iloc[0]
                ),
                "interpretation": "orientation orders overload",
            },
            {
                "control": "path-length control",
                "metric": "arc length vs overload",
                "value": float(
                    geometry_corr[
                        (geometry_corr["x"] == "trajectory_arc_length") & (geometry_corr["y"] == "mean_overload")
                    ]["spearman"].iloc[0]
                ),
                "p_value": float(
                    geometry_corr[
                        (geometry_corr["x"] == "trajectory_arc_length") & (geometry_corr["y"] == "mean_overload")
                    ]["exact_p_two_sided"].iloc[0]
                ),
                "interpretation": "not just larger motion",
            },
            {
                "control": "cycle-order null",
                "metric": "random route-local point order",
                "value": float(order_null["observed_spearman"]),
                "p_value": float(order_null["one_sided_p_rho_le_observed"]),
                "interpretation": "temporal order carries signal",
            },
            {
                "control": "metric rescaling",
                "metric": "perfect closed-area ordering",
                "value": float((robust["spearman"].abs() >= 0.999).sum()),
                "p_value": np.nan,
                "interpretation": f"{int((robust['spearman'].abs() >= 0.999).sum())}/{len(robust)} metric scalings",
            },
            {
                "control": "rank-coordinate boundary",
                "metric": "rank closed area vs overload",
                "value": float(rank["spearman"]),
                "p_value": float(rank["exact_p_two_sided"]),
                "interpretation": "not arbitrary coordinate invariant",
            },
            {
                "control": "writing time",
                "metric": "half-area cycle vs overload",
                "value": float(
                    writing_corr[
                        (writing_corr["x"] == "half_area_cycle") & (writing_corr["y"] == "mean_overload")
                    ]["spearman"].iloc[0]
                ),
                "p_value": float(
                    writing_corr[
                        (writing_corr["x"] == "half_area_cycle") & (writing_corr["y"] == "mean_overload")
                    ]["exact_p_two_sided"].iloc[0]
                ),
                "interpretation": "risk grows with writing time",
            },
        ]
    )
    return route, controls


def draw_orientation_ladder(ax: plt.Axes, route: pd.DataFrame, controls: pd.DataFrame) -> None:
    d = route.sort_values("mean_overload").copy()
    x = np.arange(len(d))
    ax.plot(x, d["signed_circulation_area"], color="#AEB6C0", lw=0.85, zorder=1)
    for i, row in enumerate(d.itertuples(index=False)):
        ax.scatter(
            i,
            row.signed_circulation_area,
            s=42 + 18 * row.half_area_cycle,
            marker=MARKERS[row.regime_id],
            color=COLORS[row.regime_id],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(i, row.signed_circulation_area + 0.45, row.regime_id, color=COLORS[row.regime_id], ha="center", fontsize=6.2)
    stat = controls[controls["control"] == "circulation orientation"].iloc[0]
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.text(0.04, 0.08, rf"$\rho={stat.value:.2f}$, exact $P={stat.p_value:.3f}$", transform=ax.transAxes, fontsize=6.3, color=INK, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["low", "", "", "", "high"])
    ax.set_xlabel("route overload rank")
    ax.set_ylabel("signed circulation area")
    ax.set_title("overload is ordered by circulation orientation", loc="left", pad=4)
    finish(ax, axis="y")


def draw_writing(ax: plt.Axes, tables: dict[str, pd.DataFrame], controls: pd.DataFrame) -> None:
    curves = tables["writing_curves"]
    summary = tables["writing_summary"].set_index("regime_id")
    for rid in ROUTES:
        g = curves[curves["regime_id"] == rid].sort_values("cycle")
        s = summary.loc[rid]
        ax.plot(g["cycle"], g["signed_progress_to_final"], color=COLORS[rid], lw=1.15, alpha=0.88)
        ax.scatter([s["half_area_cycle"]], [0.5], marker=MARKERS[rid], s=30, color=COLORS[rid], edgecolor="white", linewidth=0.55, zorder=4)
    stat = controls[controls["control"] == "writing time"].iloc[0]
    ax.axhline(0.5, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axhline(1.0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.text(0.04, 0.08, rf"half-cycle $\rho={stat.value:.2f}$, $P={stat.p_value:.3f}$", transform=ax.transAxes, fontsize=6.15, color=INK, va="bottom")
    ax.set_xlim(1, 30)
    ax.set_ylim(-0.15, 1.35)
    ax.set_xlabel("cycle")
    ax.set_ylabel("progress to final area")
    ax.set_title("dangerous orientation is written more slowly", loc="left", pad=4)
    finish(ax)


def draw_controls(ax: plt.Axes, controls: pd.DataFrame) -> None:
    order = ["circulation orientation", "path-length control", "cycle-order null", "metric rescaling", "rank-coordinate boundary", "writing time"]
    d = controls.set_index("control").loc[order].reset_index()
    labels = ["orientation", "path length", "cycle-order null", "metric rescaling", "rank boundary", "writing time"]
    outcomes = []
    colors = []
    for row in d.itertuples(index=False):
        if row.control == "metric rescaling":
            colors.append("#345995")
            outcomes.append(row.interpretation)
        elif row.control == "path-length control":
            colors.append("#AEB6C0")
            outcomes.append(rf"$\rho={row.value:.1f}$, $P={row.p_value:.3f}$")
        elif row.control == "rank-coordinate boundary":
            colors.append("#737D89")
            outcomes.append(rf"$\rho={row.value:.1f}$, $P={row.p_value:.3f}$")
        else:
            colors.append(ACCENT)
            outcomes.append(rf"$\rho={row.value:.2f}$, $P={row.p_value:.3f}$")
    y = np.arange(len(d))[::-1]
    ax.scatter(np.full(len(d), 0.04), y, s=34, color=colors, clip_on=False, zorder=3)
    for yi, label, outcome, color in zip(y, labels, outcomes, colors):
        ax.text(0.10, yi, label, ha="left", va="center", fontsize=6.3, color=INK)
        ax.text(0.56, yi, outcome, ha="left", va="center", fontsize=5.9, color=color)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.65, len(d) - 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for yi in y[:-1] - 0.5:
        ax.axhline(yi, color=GRID, lw=0.45, zorder=0)
    ax.set_title("controls separate signal from artefact", loc="left", pad=4)


def draw_coordinate_heatmap(ax: plt.Axes, tables: dict[str, pd.DataFrame]) -> None:
    corr = tables["coord_corr"]
    d = corr[corr["target"].eq("mean_overload")].copy()
    labels = ["route-centred z", "route robust z", "route-centred raw", "global z", "global robust z", "route rank"]
    metrics = ["open_signed_area", "closed_signed_area", "net_displacement"]
    mat = np.zeros((len(labels), len(metrics)))
    for i, label in enumerate(labels):
        for j, metric in enumerate(metrics):
            mat[i, j] = d[(d["coordinate_label"] == label) & (d["geometry_metric"] == metric)]["spearman"].iloc[0]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["open\narea", "closed\narea", "net\ndispl."])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=5.8, color="white" if abs(mat[i, j]) > 0.65 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=5.6, length=2)
    cbar.set_label(r"$\rho$ with overload", fontsize=5.8)
    ax.set_title("metric coordinates retain closed-area ordering", loc="left", pad=4)


def build_figure(tables: dict[str, pd.DataFrame], route: pd.DataFrame, controls: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.0, 1.12], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:])
    draw_orientation_ladder(ax_a, route, controls)
    panel(ax_a, "a", x=-0.16)
    draw_writing(ax_b, tables, controls)
    panel(ax_b, "b", x=-0.08)
    draw_controls(ax_c, controls)
    panel(ax_c, "c", x=-0.16)
    draw_coordinate_heatmap(ax_d, tables)
    panel(ax_d, "d", x=-0.08)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig57_phase_circulation_synthesis.{ext}", **kwargs)
    plt.close(fig)


def write_report(route: pd.DataFrame, controls: pd.DataFrame) -> None:
    out = ROOT / "nature_physics_phase_circulation_synthesis.md"
    orientation = controls[controls["control"] == "circulation orientation"].iloc[0]
    null = controls[controls["control"] == "cycle-order null"].iloc[0]
    writing = controls[controls["control"] == "writing time"].iloc[0]
    lines = [
        "# Phase-circulation synthesis audit",
        "",
        "## Question",
        "",
        "Can the recent phase-displacement, phase-circulation, null, coordinate-robustness and writing-time audits be compressed into one coherent reviewer-facing mechanism figure?",
        "",
        "## Main result",
        "",
        f"Signed circulation orientation co-orders overload with rho = {orientation.value:.3f} (exact P = {orientation.p_value:.3f}); the same ordering is unlikely under a route-preserving cycle-order null (P = {null.p_value:.4f}); and half-circulation writing time increases with overload (rho = {writing.value:.3f}, exact P = {writing.p_value:.3f}). The consolidated interpretation is that overload follows a temporally written circulation orientation in the reduced memory/excitation plane, not simply path length or arbitrary coordinate scaling.",
        "",
        "## Route synthesis metrics",
        "",
        route.round(5).to_markdown(index=False),
        "",
        "## Control metrics",
        "",
        controls.round(5).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: phase circulation is a compact, reviewer-facing diagnostic that links overload, temporal order, coordinate-robust orientation and writing time in the five-route true-force ensemble.",
        "",
        "Not allowed: this is not a coordinate-free geometric phase, entropy production, universal breathing frequency, or independent causal proof.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig57_phase_circulation_synthesis.*`",
        "- `source_data/nphys_phase_circulation_synthesis_route_metrics.csv`",
        "- `source_data/nphys_phase_circulation_synthesis_control_metrics.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    tables = load_tables()
    route, controls = synthesis_metrics(tables)
    route.to_csv(SRC / "nphys_phase_circulation_synthesis_route_metrics.csv", index=False)
    controls.to_csv(SRC / "nphys_phase_circulation_synthesis_control_metrics.csv", index=False)
    build_figure(tables, route, controls)
    write_report(route, controls)
    print("Wrote phase-circulation synthesis")
    print(controls[["control", "metric", "value", "p_value", "interpretation"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
