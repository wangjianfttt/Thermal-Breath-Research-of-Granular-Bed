#!/usr/bin/env python3
from __future__ import annotations

from itertools import permutations
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

DISPLACEMENTS = SRC / "nphys_return_map_phase_portrait_route_displacements.csv"
ATLAS = SRC / "nphys_route_susceptibility_atlas.csv"

ROUTES = ["R1", "R3", "R5", "R6", "R6c"]
COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"


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


def exact_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rho = float(spearmanr(x_arr, y_arr).statistic)
    obs = abs(rho)
    hits = 0
    total = 0
    for perm in permutations(y_arr):
        trial = abs(float(spearmanr(x_arr, np.asarray(perm, dtype=float)).statistic))
        hits += int(trial >= obs - 1e-12)
        total += 1
    return rho, hits / total


def scaled(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def load_table() -> pd.DataFrame:
    disp = pd.read_csv(DISPLACEMENTS)
    atlas = pd.read_csv(ATLAS)
    df = disp.merge(atlas, on="regime_id", how="inner")
    df["downward_hot_displacement"] = np.maximum(-df["delta_hot_excitation_coordinate"], 0.0)
    df["upward_hot_displacement"] = np.maximum(df["delta_hot_excitation_coordinate"], 0.0)
    df["exhalation_polarity"] = -df["delta_hot_excitation_coordinate"] / df["displacement_norm"]
    df["memory_rewrite_polarity"] = -df["delta_memory_coordinate"] / df["displacement_norm"]
    df["route_order"] = pd.Categorical(df["regime_id"], categories=ROUTES, ordered=True)
    return df.sort_values("route_order").drop(columns=["route_order"])


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("downward hot displacement vs mean overload", "downward_hot_displacement", "mean_overload"),
        ("downward hot displacement vs loop gain", "downward_hot_displacement", "loop_gain"),
        ("downward hot displacement vs slow severity", "downward_hot_displacement", "S"),
        ("raw hot displacement vs mean overload", "delta_hot_excitation_coordinate", "mean_overload"),
        ("exhalation polarity vs mean overload", "exhalation_polarity", "mean_overload"),
        ("displacement norm vs mean overload", "displacement_norm", "mean_overload"),
        ("downward hot displacement vs lag-2 loop AUC", "downward_hot_displacement", "lag2_loop_auc"),
        ("downward hot displacement vs peak transient gain", "downward_hot_displacement", "peak_normalized_gain"),
    ]
    rows = []
    for name, x, y in tests:
        rho, p = exact_spearman(df[x], df[y])
        rows.append(
            {
                "relationship": name,
                "x": x,
                "y": y,
                "n_routes": len(df),
                "spearman": rho,
                "exact_p_two_sided": p,
            }
        )
    return pd.DataFrame(rows)


def draw_vector_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        rid = row["regime_id"]
        color = COLORS[rid]
        dx = row["delta_memory_coordinate"]
        dy = row["delta_hot_excitation_coordinate"]
        lw = 1.0 + 0.06 * max(row["loop_gain"], 0)
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, alpha=0.9, shrinkA=1, shrinkB=4),
            zorder=3,
        )
        ax.scatter([dx], [dy], s=35, marker=MARKERS[rid], color=color, edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(dx + (0.06 if dx >= -0.2 else -0.07), dy, rid, color=color, fontsize=6.4, va="center", ha="left" if dx >= -0.2 else "right")
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    for r in [1, 2]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#D7DDE4", lw=0.55, ls=(0, (2, 4)), zorder=1))
    ax.text(0.04, 0.94, "downward displacement\norders overload cost", transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color=MUTED)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.65, 0.45)
    ax.set_ylim(-2.7, 2.25)
    ax.set_xlabel(r"$\Delta M$ between early and late medians")
    ax.set_ylabel(r"$\Delta\Psi$ between early and late medians")
    ax.set_title("phase-displacement geometry", loc="left", pad=4)
    finish(ax)


def draw_rank_scatter(ax: plt.Axes, df: pd.DataFrame, y: str, ylabel: str, title: str, rel: pd.DataFrame) -> None:
    x = "downward_hot_displacement"
    for _, row in df.iterrows():
        rid = row["regime_id"]
        ax.scatter(row[x], row[y], s=42, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(row[x] + 0.035, row[y], rid, color=COLORS[rid], fontsize=6.3, va="center")
    order = np.argsort(df[x].to_numpy(float))
    ax.plot(df[x].to_numpy(float)[order], df[y].to_numpy(float)[order], color="#AEB6C0", lw=0.75, zorder=2)
    stat = rel[(rel["x"] == x) & (rel["y"] == y)].iloc[0]
    ax.text(0.04, 0.94, rf"$\rho={stat.spearman:.2f}$, exact $P={stat.exact_p_two_sided:.3f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK)
    ax.set_xlabel(r"downward displacement, $[-\Delta\Psi]_+$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    finish(ax)


def draw_mechanism_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    d = df.sort_values("mean_overload").copy()
    rows = [
        ("slow state S", "S"),
        (r"$[-\Delta\Psi]_+$", "downward_hot_displacement"),
        ("loop gain G(S)", "loop_gain"),
        ("mean overload", "mean_overload"),
        ("lag-2 loop AUC", "lag2_loop_auc"),
        ("peak gain", "peak_normalized_gain"),
    ]
    mat = np.vstack([scaled(d[col]).to_numpy(float) for _, col in rows])
    im = ax.imshow(mat, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["regime_id"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([name for name, _ in rows])
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] < 0.25 or mat[i, j] > 0.72 else INK
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.set_label("route-normalised value", fontsize=6.0)
    ax.set_title("slow susceptibility, breathing direction and risk co-order", loc="left", pad=4)


def build_figure(df: pd.DataFrame, rel: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])
    draw_vector_panel(ax_a, df)
    panel(ax_a, "a", x=-0.10)
    draw_rank_scatter(ax_b, df, "mean_overload", "mean asinh overload", "overload is a direction in phase space", rel)
    panel(ax_b, "b")
    draw_rank_scatter(ax_c, df, "loop_gain", r"loop-to-overload gain $G(S)$", "route gain follows the same geometry", rel)
    panel(ax_c, "c")
    draw_mechanism_matrix(ax_d, df)
    panel(ax_d, "d", x=-0.08)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig52_phase_displacement_geometry.{ext}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame, rel: pd.DataFrame) -> None:
    down_mean = rel[(rel["x"] == "downward_hot_displacement") & (rel["y"] == "mean_overload")].iloc[0]
    down_gain = rel[(rel["x"] == "downward_hot_displacement") & (rel["y"] == "loop_gain")].iloc[0]
    raw_mean = rel[(rel["x"] == "delta_hot_excitation_coordinate") & (rel["y"] == "mean_overload")].iloc[0]
    out = ROOT / "nature_physics_phase_displacement_geometry.md"
    lines = [
        "# Phase-displacement geometry audit",
        "",
        "## Question",
        "",
        "Does the Fig. 5a route-displacement vector encode a physical breathing direction, or is it only a visual summary of the return-map coordinates?",
        "",
        "## Main result",
        "",
        f"The downward hot-coordinate displacement `[-Delta Psi]_+` orders mean overload across the five true-force routes with Spearman rho = {down_mean.spearman:.3f} (exact P = {down_mean.exact_p_two_sided:.3f}). It also orders the fitted loop-to-overload gain with rho = {down_gain.spearman:.3f} (exact P = {down_gain.exact_p_two_sided:.3f}). Equivalently, the signed hot displacement `Delta Psi` orders mean overload with rho = {raw_mean.spearman:.3f}.",
        "",
        "This turns the Fig. 5a vector from a display choice into a bounded geometric diagnostic: buffered routes move upward or weakly downward in the hot loop-excitation coordinate, whereas overload-prone routes acquire a large downward phase displacement. The result is route-level and uses only five routes, so it should be framed as a co-ordering test rather than a calibrated constitutive law.",
        "",
        "## Route geometry table",
        "",
        df.round(4).to_markdown(index=False),
        "",
        "## Exact Spearman checks",
        "",
        rel.round(4).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: the direction of the reduced return-map displacement gives an operational breathing geometry that co-orders slow susceptibility, loop gain and overload cost.",
        "",
        "Not allowed: this does not prove a universal phase-space potential, a bifurcation, or a material law. The displacement is a route-level diagnostic extracted from measured early/middle/late medians.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig52_phase_displacement_geometry.*`",
        "- `source_data/nphys_phase_displacement_geometry.csv`",
        "- `source_data/nphys_phase_displacement_geometry_correlations.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_table()
    rel = correlation_table(df)
    df.to_csv(SRC / "nphys_phase_displacement_geometry.csv", index=False)
    rel.to_csv(SRC / "nphys_phase_displacement_geometry_correlations.csv", index=False)
    build_figure(df, rel)
    write_report(df, rel)
    print("Wrote phase-displacement geometry audit")
    print(rel[["relationship", "spearman", "exact_p_two_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
