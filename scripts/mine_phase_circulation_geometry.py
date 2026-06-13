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

CYCLES = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"
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


def route_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = pd.read_csv(CYCLES)
    atlas = pd.read_csv(ATLAS)
    route_rows = []
    segment_rows = []
    for rid, g in cycles.dropna(subset=["memory_coordinate", "hot_excitation_coordinate"]).groupby("regime_id", sort=True):
        g = g.sort_values("cycle").copy()
        x = g["memory_coordinate"].to_numpy(float)
        y = g["hot_excitation_coordinate"].to_numpy(float)
        dx = np.diff(x)
        dy = np.diff(y)
        step = np.hypot(dx, dy)
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        arc = float(step.sum())
        downward_flux = float(np.maximum(-dy, 0).sum())
        upward_flux = float(np.maximum(dy, 0).sum())
        turn_angles = []
        for i in range(len(dx) - 1):
            a = np.array([dx[i], dy[i]])
            b = np.array([dx[i + 1], dy[i + 1]])
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na > 0 and nb > 0:
                turn_angles.append(float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))
        route_rows.append(
            {
                "regime_id": rid,
                "signed_circulation_area": float(signed_area),
                "clockwise_circulation": float(max(-signed_area, 0)),
                "counterclockwise_circulation": float(max(signed_area, 0)),
                "abs_circulation_area": float(abs(signed_area)),
                "trajectory_arc_length": arc,
                "net_displacement": net,
                "path_persistence": float(net / arc) if arc > 0 else np.nan,
                "downward_hot_flux": downward_flux,
                "upward_hot_flux": upward_flux,
                "downward_bias": float(downward_flux / upward_flux) if upward_flux > 0 else np.nan,
                "memory_flux": float(np.abs(dx).sum()),
                "mean_step_length": float(step.mean()),
                "turn_angle_sum": float(np.sum(turn_angles)),
                "turn_angle_mean": float(np.mean(turn_angles)) if turn_angles else np.nan,
                "n_cycles": int(len(g)),
            }
        )
        for i in range(len(dx)):
            segment_rows.append(
                {
                    "regime_id": rid,
                    "cycle": int(g["cycle"].iloc[i]),
                    "next_cycle": int(g["cycle"].iloc[i + 1]),
                    "memory_coordinate": x[i],
                    "hot_excitation_coordinate": y[i],
                    "next_memory_coordinate": x[i + 1],
                    "next_hot_excitation_coordinate": y[i + 1],
                    "delta_memory_coordinate": dx[i],
                    "delta_hot_excitation_coordinate": dy[i],
                    "step_length": step[i],
                    "signed_area_increment": 0.5 * (x[i] * y[i + 1] - x[i + 1] * y[i]),
                }
            )
    route = pd.DataFrame(route_rows).merge(atlas, on="regime_id", how="inner")
    route["route_order"] = pd.Categorical(route["regime_id"], ROUTES, ordered=True)
    return route.sort_values("route_order").drop(columns=["route_order"]), pd.DataFrame(segment_rows)


def correlation_table(route: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("signed circulation vs mean overload", "signed_circulation_area", "mean_overload"),
        ("signed circulation vs loop gain", "signed_circulation_area", "loop_gain"),
        ("signed circulation vs slow severity", "signed_circulation_area", "S"),
        ("clockwise circulation vs mean overload", "clockwise_circulation", "mean_overload"),
        ("net displacement vs mean overload", "net_displacement", "mean_overload"),
        ("trajectory arc length vs mean overload", "trajectory_arc_length", "mean_overload"),
        ("absolute area vs mean overload", "abs_circulation_area", "mean_overload"),
        ("downward bias vs mean overload", "downward_bias", "mean_overload"),
    ]
    rows = []
    for name, x, y in tests:
        rho, p = exact_spearman(route[x], route[y])
        rows.append(
            {
                "relationship": name,
                "x": x,
                "y": y,
                "n_routes": len(route),
                "spearman": rho,
                "exact_p_two_sided": p,
            }
        )
    return pd.DataFrame(rows)


def draw_trajectories(ax: plt.Axes, segments: pd.DataFrame) -> None:
    for rid, g in segments.groupby("regime_id", sort=True):
        color = COLORS[rid]
        x = np.r_[g["memory_coordinate"].to_numpy(float), g["next_memory_coordinate"].iloc[-1]]
        y = np.r_[g["hot_excitation_coordinate"].to_numpy(float), g["next_hot_excitation_coordinate"].iloc[-1]]
        ax.plot(x, y, color=color, lw=1.0, alpha=0.75, zorder=2)
        idx = np.linspace(2, len(x) - 3, 3, dtype=int)
        for j in idx:
            ax.annotate(
                "",
                xy=(x[j + 1], y[j + 1]),
                xytext=(x[j], y[j]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, alpha=0.75, shrinkA=0, shrinkB=0),
                zorder=3,
            )
        ax.scatter([x[0]], [y[0]], s=28, marker=MARKERS[rid], facecolor="white", edgecolor=color, linewidth=0.9, zorder=4)
        ax.scatter([x[-1]], [y[-1]], s=34, marker=MARKERS[rid], facecolor=color, edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(x[-1] + 0.05, y[-1], rid, color=color, fontsize=6.4, va="center")
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.text(0.04, 0.94, "open=start, filled=end\narrows follow cycle order", transform=ax.transAxes, fontsize=6.1, color=MUTED, va="top")
    ax.set_xlabel("cold loop-memory coordinate")
    ax.set_ylabel(r"hot loop-excitation coordinate $\Psi$")
    ax.set_title("full breathing trajectories", loc="left", pad=4)
    finish(ax)


def draw_single_trajectory(ax: plt.Axes, g: pd.DataFrame, rid: str, show_xlabel: bool) -> None:
    color = COLORS[rid]
    x = np.r_[g["memory_coordinate"].to_numpy(float), g["next_memory_coordinate"].iloc[-1]]
    y = np.r_[g["hot_excitation_coordinate"].to_numpy(float), g["next_hot_excitation_coordinate"].iloc[-1]]
    ax.plot(x, y, color=color, lw=1.0, alpha=0.78, zorder=2)
    idx = np.linspace(2, len(x) - 3, 2, dtype=int)
    for j in idx:
        ax.annotate(
            "",
            xy=(x[j + 1], y[j + 1]),
            xytext=(x[j], y[j]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, alpha=0.8, shrinkA=0, shrinkB=0),
            zorder=3,
        )
    ax.scatter([x[0]], [y[0]], s=20, marker=MARKERS[rid], facecolor="white", edgecolor=color, linewidth=0.8, zorder=4)
    ax.scatter([x[-1]], [y[-1]], s=24, marker=MARKERS[rid], facecolor=color, edgecolor="white", linewidth=0.55, zorder=4)
    ax.axhline(0, color="#B9C2CD", lw=0.55, ls=(0, (3, 3)))
    ax.axvline(0, color="#B9C2CD", lw=0.55, ls=(0, (3, 3)))
    ax.text(0.03, 0.88, rid, color=color, transform=ax.transAxes, fontsize=6.6, fontweight="bold", va="top")
    ax.set_xlim(min(x) - 0.15 * (max(x) - min(x) + 1e-9), max(x) + 0.15 * (max(x) - min(x) + 1e-9))
    ax.set_ylim(min(y) - 0.16 * (max(y) - min(y) + 1e-9), max(y) + 0.16 * (max(y) - min(y) + 1e-9))
    ax.grid(True, color=GRID, lw=0.35, zorder=0)
    ax.tick_params(labelsize=5.2, length=1.8, width=0.5)
    if not show_xlabel:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("M", labelpad=1)
    ax.set_ylabel(r"$\Psi$", labelpad=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_trajectory_strip(fig: plt.Figure, spec, segments: pd.DataFrame) -> list[plt.Axes]:
    sub = spec.subgridspec(5, 1, hspace=0.12)
    axes: list[plt.Axes] = []
    for i, rid in enumerate(ROUTES):
        ax = fig.add_subplot(sub[i, 0])
        g = segments[segments["regime_id"] == rid].sort_values("cycle")
        draw_single_trajectory(ax, g, rid, show_xlabel=i == len(ROUTES) - 1)
        axes.append(ax)
    axes[0].set_title("full route trajectories", loc="left", pad=4)
    axes[0].text(
        0.55,
        0.88,
        "open=start, filled=end",
        transform=axes[0].transAxes,
        fontsize=5.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    return axes


def draw_rank(ax: plt.Axes, route: pd.DataFrame, rel: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str) -> None:
    for _, row in route.iterrows():
        rid = row["regime_id"]
        ax.scatter(row[x], row[y], s=42, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(row[x] + 0.06 * (route[x].max() - route[x].min() + 1e-9), row[y], rid, color=COLORS[rid], fontsize=6.3, va="center")
    order = np.argsort(route[x].to_numpy(float))
    ax.plot(route[x].to_numpy(float)[order], route[y].to_numpy(float)[order], color="#AEB6C0", lw=0.75, zorder=2)
    stat = rel[(rel["x"] == x) & (rel["y"] == y)].iloc[0]
    ax.text(
        0.96,
        0.06,
        rf"$\rho={stat.spearman:.2f}$, exact $P={stat.exact_p_two_sided:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=INK,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    finish(ax)


def draw_mechanism_grid(ax: plt.Axes, route: pd.DataFrame) -> None:
    d = route.sort_values("mean_overload").copy()
    rows = [
        ("clockwise area", "clockwise_circulation"),
        ("signed area", "signed_circulation_area"),
        ("arc length", "trajectory_arc_length"),
        ("downward bias", "downward_bias"),
        ("loop gain", "loop_gain"),
        ("mean overload", "mean_overload"),
    ]
    mat = []
    for _, col in rows:
        vals = scaled(d[col])
        mat.append(vals.to_numpy(float))
    mat = np.vstack(mat)
    im = ax.imshow(mat, cmap="cividis", aspect="auto", vmin=0, vmax=1)
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
    ax.set_title("orientation, not path length, follows overload", loc="left", pad=4)


def build_figure(route: pd.DataFrame, segments: pd.DataFrame, rel: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.0, 1.0])
    axes_a = draw_trajectory_strip(fig, gs[:, 0], segments)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])
    panel(axes_a[0], "a", x=-0.16, y=1.20)
    draw_rank(
        ax_b,
        route,
        rel,
        "signed_circulation_area",
        "mean_overload",
        "signed circulation area",
        "mean asinh overload",
        "overload follows circulation orientation",
    )
    panel(ax_b, "b")
    draw_rank(
        ax_c,
        route,
        rel,
        "signed_circulation_area",
        "loop_gain",
        "signed circulation area",
        r"loop-to-overload gain $G(S)$",
        "susceptibility follows the same orientation",
    )
    panel(ax_c, "c")
    draw_mechanism_grid(ax_d, route)
    panel(ax_d, "d", x=-0.08)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig53_phase_circulation_geometry.{ext}", **kwargs)
    plt.close(fig)


def write_report(route: pd.DataFrame, rel: pd.DataFrame) -> None:
    over = rel[(rel["x"] == "signed_circulation_area") & (rel["y"] == "mean_overload")].iloc[0]
    gain = rel[(rel["x"] == "signed_circulation_area") & (rel["y"] == "loop_gain")].iloc[0]
    arc = rel[(rel["x"] == "trajectory_arc_length") & (rel["y"] == "mean_overload")].iloc[0]
    out = ROOT / "nature_physics_phase_circulation_geometry.md"
    lines = [
        "# Phase-circulation geometry audit",
        "",
        "## Question",
        "",
        "Does the full cycle-by-cycle trajectory in the reduced cold-memory/hot-excitation plane support the breathing-geometry interpretation, or is the Fig. 5a displacement signal only a three-segment summary?",
        "",
        "## Main result",
        "",
        f"The signed circulation area of the full route trajectory co-orders mean overload with Spearman rho = {over.spearman:.3f} (exact P = {over.exact_p_two_sided:.3f}) and loop-to-overload gain with rho = {gain.spearman:.3f} (exact P = {gain.exact_p_two_sided:.3f}). The trajectory arc length does not show the same ordering (rho = {arc.spearman:.3f}), so the signal is orientation/rectification rather than simply larger motion in the reduced plane.",
        "",
        "This supports a cautious non-equilibrium reading: dangerous routes do not merely breathe more; they circulate through the reduced memory-excitation plane with the opposite handedness from the buffered route.",
        "",
        "## Route geometry table",
        "",
        route.round(4).to_markdown(index=False),
        "",
        "## Exact Spearman checks",
        "",
        rel.round(4).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: signed phase-space circulation is a route-level diagnostic of breathing rectification and co-orders overload/gain in the measured five-route ensemble.",
        "",
        "Not allowed: this is not a geometric phase law, entropy production, or evidence for a universal attractor. The area is a reduced-coordinate audit and depends on the chosen memory/excitation projection.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig53_phase_circulation_geometry.*`",
        "- `source_data/nphys_phase_circulation_geometry.csv`",
        "- `source_data/nphys_phase_circulation_geometry_segments.csv`",
        "- `source_data/nphys_phase_circulation_geometry_correlations.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    route, segments = route_geometry()
    rel = correlation_table(route)
    route.to_csv(SRC / "nphys_phase_circulation_geometry.csv", index=False)
    segments.to_csv(SRC / "nphys_phase_circulation_geometry_segments.csv", index=False)
    rel.to_csv(SRC / "nphys_phase_circulation_geometry_correlations.csv", index=False)
    build_figure(route, segments, rel)
    write_report(route, rel)
    print("Wrote phase-circulation geometry audit")
    print(rel[["relationship", "spearman", "exact_p_two_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
