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

INFILE = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"

COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#242A31"
GRID = "#E7EAEE"
MUTED = "#7B8490"
ACCENT = "#B6423E"
COOL = "#3D6B9C"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["dM"] = df["next_memory_coordinate"] - df["memory_coordinate"]
    df["dPsi"] = df["next_hot_excitation_coordinate"] - df["hot_excitation_coordinate"]
    df["radial_before"] = np.hypot(df["memory_coordinate"], df["hot_excitation_coordinate"])
    df["radial_after"] = np.hypot(df["next_memory_coordinate"], df["next_hot_excitation_coordinate"])
    df["radial_contraction"] = df["radial_before"] - df["radial_after"]
    df["phase_speed"] = np.hypot(df["dM"], df["dPsi"])
    df["signed_circulation"] = 0.5 * (
        df["memory_coordinate"] * df["next_hot_excitation_coordinate"]
        - df["next_memory_coordinate"] * df["hot_excitation_coordinate"]
    )
    df["absolute_circulation"] = df["signed_circulation"].abs()
    df["tangential_fraction"] = df["absolute_circulation"] / (df["radial_before"] * df["phase_speed"] + 1e-12)
    for col in ["overload_number", "dimensionless_loop_number"]:
        df[f"{col}_within_route"] = df[col] - df.groupby("regime_id")[col].transform("mean")
    df["overload_number_within_route_asinh"] = np.arcsinh(df["overload_number_within_route"] / 2.0)

    pairs = [
        ("absolute_circulation", "overload_number_within_route_asinh", "geometric_area_to_overload"),
        ("phase_speed", "overload_number_within_route_asinh", "phase_speed_to_overload"),
        ("tangential_fraction", "dimensionless_loop_number", "tangential_fraction_to_loop_number"),
        ("radial_contraction", "overload_number_within_route", "contraction_to_overload_boundary"),
    ]
    corr_rows = []
    for x, y, name in pairs:
        d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        stat = spearmanr(d[x], d[y])
        corr_rows.append(
            {
                "relationship": name,
                "predictor": x,
                "target": y,
                "spearman": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "n": int(len(d)),
            }
        )
    corr = pd.DataFrame(corr_rows)
    summary = (
        df.groupby("regime_id", sort=True)
        .agg(
            n=("cycle", "count"),
            signed_circulation_mean=("signed_circulation", "mean"),
            absolute_circulation_mean=("absolute_circulation", "mean"),
            radial_contraction_mean=("radial_contraction", "mean"),
            phase_speed_mean=("phase_speed", "mean"),
            tangential_fraction_mean=("tangential_fraction", "mean"),
            overload_number_mean=("overload_number", "mean"),
            dimensionless_loop_number_mean=("dimensionless_loop_number", "mean"),
        )
        .reset_index()
    )
    return df, corr, summary


def scatter_by_regime(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, *, s: float = 18.0, alpha: float = 0.78) -> None:
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g[x],
            g[y],
            s=s,
            color=COLORS.get(rid, MUTED),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            linewidth=0.35,
            alpha=alpha,
            zorder=3,
        )


def draw_figure(df: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel(ax_a, "a")
    for rid, g in df.dropna(subset=["memory_coordinate", "hot_excitation_coordinate"]).groupby("regime_id", sort=True):
        color = COLORS.get(rid, MUTED)
        g = g.sort_values("cycle")
        ax_a.plot(g["memory_coordinate"], g["hot_excitation_coordinate"], color=color, lw=0.8, alpha=0.55)
        ax_a.scatter(
            g["memory_coordinate"],
            g["hot_excitation_coordinate"],
            c=g["signed_circulation"],
            cmap="RdBu_r",
            vmin=-0.8,
            vmax=0.8,
            s=14,
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            linewidth=0.25,
            alpha=0.84,
            zorder=3,
        )
        ax_a.text(g["memory_coordinate"].tail(5).median() + 0.07, g["hot_excitation_coordinate"].tail(5).median(), rid, color=color, fontsize=6.3)
    ax_a.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_a.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_a.set_xlabel("cold loop-memory coordinate")
    ax_a.set_ylabel(r"hot loop-excitation coordinate $\Psi$")
    ax_a.set_title("signed geometric flow in the breathing plane", loc="left", pad=4)
    finish(ax_a)

    panel(ax_b, "b")
    scatter_by_regime(ax_b, df, "absolute_circulation", "overload_number_within_route_asinh")
    row = corr[corr["relationship"] == "geometric_area_to_overload"].iloc[0]
    ax_b.text(0.05, 0.96, rf"$\rho={row.spearman:.2f}$" + f"\nP={row.p_value:.1e}", transform=ax_b.transAxes, ha="left", va="top")
    ax_b.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_b.set_xlabel("absolute circulation")
    ax_b.set_ylabel(r"asinh route-centred overload")
    ax_b.set_title("circulation is supporting, not sufficient", loc="left", pad=4)
    finish(ax_b)

    panel(ax_c, "c")
    scatter_by_regime(ax_c, df, "phase_speed", "overload_number_within_route_asinh")
    row = corr[corr["relationship"] == "phase_speed_to_overload"].iloc[0]
    ax_c.text(0.05, 0.96, rf"$\rho={row.spearman:.2f}$" + f"\nP={row.p_value:.1e}", transform=ax_c.transAxes, ha="left", va="top")
    ax_c.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_c.set_xlabel("phase-flow speed")
    ax_c.set_ylabel(r"asinh route-centred overload")
    ax_c.set_title("faster phase flow weakly raises overload", loc="left", pad=4)
    finish(ax_c)

    panel(ax_d, "d")
    order = summary.sort_values("dimensionless_loop_number_mean")["regime_id"].tolist()
    s = summary.set_index("regime_id").loc[order]
    x = np.arange(len(order))
    ax_d.bar(x - 0.18, s["absolute_circulation_mean"], width=0.34, color="#AAB4C0", label="|circulation|")
    ax_d.bar(x + 0.18, s["tangential_fraction_mean"], width=0.34, color=ACCENT, label="tangential fraction")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(order)
    ax_d.set_ylabel("route mean")
    ax_d.set_title("route ordering separates geometry and force loops", loc="left", pad=4)
    ax_d.legend(loc="upper left", fontsize=5.8, handlelength=1.1)
    finish(ax_d, "y")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig19_geometric_phase_flow.{ext}", **kwargs)
    plt.close(fig)


def write_report(corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Geometric phase-flow audit",
        "",
        "This audit asks whether the reduced breathing plane contains a geometric circulation signal beyond the force-loop number.",
        "",
        "## Correlations",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Manuscript-safe interpretation",
        "",
        "- Absolute circulation and phase-flow speed weakly correlate with route-centred overload, but the effects are much smaller than the dimensionless loop-number collapse.",
        "- The result is useful as a boundary: generic geometric cycling is not sufficient for overload; the dangerous response requires force-loop embedding and route controls.",
        "- The figure should be Extended Data unless the manuscript needs an explicit negative-control figure for the breathing analogy.",
    ]
    (ROOT / "nature_physics_geometric_phase_flow_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df, corr, summary = build_metrics()
    df.to_csv(SRC / "nphys_geometric_phase_flow_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_geometric_phase_flow_correlations.csv", index=False)
    summary.to_csv(SRC / "nphys_geometric_phase_flow_route_summary.csv", index=False)
    draw_figure(df, corr, summary)
    write_report(corr, summary)
    print("Wrote geometric phase-flow audit")
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
