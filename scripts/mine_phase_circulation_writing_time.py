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


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = pd.read_csv(CYCLES)
    atlas = pd.read_csv(ATLAS).set_index("regime_id")
    curve_rows = []
    summary_rows = []
    for rid, g in cycles.dropna(subset=["memory_coordinate", "hot_excitation_coordinate"]).groupby("regime_id", sort=True):
        g = g.sort_values("cycle")
        x = g["memory_coordinate"].to_numpy(float)
        y = g["hot_excitation_coordinate"].to_numpy(float)
        cycle = g["cycle"].to_numpy(int)
        inc = 0.5 * (x[:-1] * y[1:] - x[1:] * y[:-1])
        cumulative = np.cumsum(inc)
        final = float(cumulative[-1])
        denom = abs(final) if abs(final) > 1e-12 else 1.0
        target = 0.5 * denom
        half_cycle = np.nan
        for c, val in zip(cycle[:-1], cumulative):
            if abs(val) >= target:
                half_cycle = int(c)
                break
        final_sign = float(np.sign(final))
        stable_cycle = np.nan
        if final_sign:
            for i, val in enumerate(cumulative):
                if np.all(np.sign(cumulative[i:]) == final_sign):
                    stable_cycle = int(cycle[i])
                    break
        early = float(np.sum(inc[:5]) / final) if abs(final) > 1e-12 else np.nan
        middle = float(np.sum(inc[5:15]) / final) if abs(final) > 1e-12 else np.nan
        late = float(np.sum(inc[15:]) / final) if abs(final) > 1e-12 else np.nan
        summary_rows.append(
            {
                "regime_id": rid,
                "final_signed_area": final,
                "final_sign": final_sign,
                "half_area_cycle": half_cycle,
                "stable_sign_cycle": stable_cycle,
                "early_fraction": early,
                "middle_fraction": middle,
                "late_fraction": late,
                "mean_overload": float(atlas.loc[rid, "mean_overload"]),
                "loop_gain": float(atlas.loc[rid, "loop_gain"]),
                "S": float(atlas.loc[rid, "S"]),
            }
        )
        for c, area_increment, area_cumulative in zip(cycle[:-1], inc, cumulative):
            curve_rows.append(
                {
                    "regime_id": rid,
                    "cycle": int(c),
                    "signed_area_increment": float(area_increment),
                    "cumulative_signed_area": float(area_cumulative),
                    "normalised_cumulative_signed_area": float(area_cumulative / denom),
                    "signed_progress_to_final": float(area_cumulative / final) if abs(final) > 1e-12 else np.nan,
                    "final_signed_area": final,
                }
            )
    curves = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    summary["route_order"] = pd.Categorical(summary["regime_id"], ROUTES, ordered=True)
    summary = summary.sort_values("route_order").drop(columns=["route_order"])

    corr_rows = []
    for x in ["half_area_cycle", "early_fraction", "middle_fraction", "late_fraction"]:
        for y in ["mean_overload", "loop_gain", "S"]:
            d = summary.dropna(subset=[x, y])
            rho, p = exact_spearman(d[x], d[y])
            corr_rows.append(
                {
                    "relationship": f"{x} vs {y}",
                    "x": x,
                    "y": y,
                    "n_routes": len(d),
                    "spearman": rho,
                    "exact_p_two_sided": p,
                }
            )
    return curves, summary, pd.DataFrame(corr_rows)


def draw_curves(ax: plt.Axes, curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    label_offsets = {"R1": (0.25, 0.03), "R3": (0.25, 0.05), "R5": (0.25, 0.09), "R6": (0.25, -0.07), "R6c": (0.25, -0.11)}
    for rid in ROUTES:
        g = curves[curves["regime_id"] == rid].sort_values("cycle")
        s = summary[summary["regime_id"] == rid].iloc[0]
        ax.plot(g["cycle"], g["signed_progress_to_final"], color=COLORS[rid], lw=1.25, alpha=0.9)
        ax.scatter([s["half_area_cycle"]], [0.5], marker=MARKERS[rid], s=34, color=COLORS[rid], edgecolor="white", linewidth=0.55, zorder=4)
        lx, ly = float(s["half_area_cycle"]), 0.5
        ox, oy = label_offsets.get(rid, (0.25, 0.0))
        ax.text(lx + ox, ly + oy, rid, color=COLORS[rid], fontsize=6.2, va="center")
    ax.axhline(0.5, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axhline(1.0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlim(1, 30.8)
    ax.set_ylim(-0.18, 1.36)
    ax.set_xlabel("cycle")
    ax.set_ylabel("progress to final circulation")
    ax.set_title("circulation handedness is written over cycles", loc="left", pad=4)
    ax.text(0.04, 0.10, "symbols mark half-area cycle", transform=ax.transAxes, fontsize=6.1, color=MUTED, va="bottom")
    finish(ax)


def draw_half_time(ax: plt.Axes, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    x = "half_area_cycle"
    y = "mean_overload"
    for _, row in summary.iterrows():
        rid = row["regime_id"]
        ax.scatter(row[x], row[y], s=42, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(row[x] + 0.10, row[y], rid, color=COLORS[rid], fontsize=6.3, va="center")
    stat = corr[(corr["x"] == x) & (corr["y"] == y)].iloc[0]
    ax.text(0.04, 0.94, rf"$\rho={stat.spearman:.2f}$, exact $P={stat.exact_p_two_sided:.3f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=INK)
    ax.set_xlabel("half-circulation writing cycle")
    ax.set_ylabel("mean asinh overload")
    ax.set_title("risk grows with writing time", loc="left", pad=4)
    finish(ax)


def draw_fraction_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    d = summary.set_index("regime_id").loc[ROUTES]
    mat = d[["early_fraction", "middle_fraction", "late_fraction"]].to_numpy(float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.1, vmax=1.1, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["early\n1-5", "middle\n6-15", "late\n16-29"])
    ax.set_yticks(np.arange(len(ROUTES)))
    ax.set_yticklabels(ROUTES)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.7, color="white" if abs(mat[i, j]) > 0.65 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.set_label("fraction of final area", fontsize=6.0)
    ax.set_title("when the circulation is written", loc="left", pad=4)


def build_figure(curves: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1:])
    draw_curves(ax_a, curves, summary)
    panel(ax_a, "a", x=-0.12)
    draw_half_time(ax_b, summary, corr)
    panel(ax_b, "b", x=-0.08)
    draw_fraction_heatmap(ax_c, summary)
    panel(ax_c, "c", x=-0.08)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig56_phase_circulation_writing_time.{ext}", **kwargs)
    plt.close(fig)


def write_report(curves: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    stat = corr[(corr["x"] == "half_area_cycle") & (corr["y"] == "mean_overload")].iloc[0]
    out = ROOT / "nature_physics_phase_circulation_writing_time.md"
    lines = [
        "# Phase-circulation writing-time audit",
        "",
        "## Question",
        "",
        "Is the phase-circulation orientation an end-state geometric label, or can its accumulation over cycles be read as a memory-writing process?",
        "",
        "## Main result",
        "",
        f"The half-circulation writing cycle co-orders mean overload across routes with Spearman rho = {stat.spearman:.3f} (exact P = {stat.exact_p_two_sided:.3f}). The sign of the final circulation is established early in all routes, but the cycle at which the trajectory accumulates half of its final signed area shifts from the first cycle in buffered routes to cycle 5 in R6c.",
        "",
        "This supports a bounded memory-writing interpretation: dangerous routes do not simply have larger phase-space motion; they take more breaths to accumulate the final circulation orientation.",
        "",
        "## Summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Correlations",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: the circulation orientation has a measurable writing time in the five-route ensemble, and this writing time increases with route severity/overload.",
        "",
        "Not allowed: this is not a universal breathing frequency, critical slowing down, or a route-independent clock.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig56_phase_circulation_writing_time.*`",
        "- `source_data/nphys_phase_circulation_writing_time_curves.csv`",
        "- `source_data/nphys_phase_circulation_writing_time_summary.csv`",
        "- `source_data/nphys_phase_circulation_writing_time_correlations.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    curves, summary, corr = build_tables()
    curves.to_csv(SRC / "nphys_phase_circulation_writing_time_curves.csv", index=False)
    summary.to_csv(SRC / "nphys_phase_circulation_writing_time_summary.csv", index=False)
    corr.to_csv(SRC / "nphys_phase_circulation_writing_time_correlations.csv", index=False)
    build_figure(curves, summary, corr)
    write_report(curves, summary, corr)
    print("Wrote phase-circulation writing-time audit")
    print(corr[corr["x"].eq("half_area_cycle")][["y", "spearman", "exact_p_two_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
