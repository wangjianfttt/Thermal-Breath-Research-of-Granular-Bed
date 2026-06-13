#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
SRC = ROOT / "source_data"

SUMMARY = SRC / "nature_physics_two_channel_summary_source.csv"
ORTHO_REGRESSION = SRC / "nphys_readout_orthogonality_regression.csv"
ORTHO_BOOTSTRAP = SRC / "nphys_readout_orthogonality_bootstrap.csv"

REGIME_ORDER = [
    "a050_mu010_g002",
    "a050_mu060_g002",
    "a100_mu030_g002",
    "a100_mu060_g000",
    "a150_mu010_g002",
    "a150_mu060_g020",
]

REGIME_IDS = {tag: f"R{i + 1}" for i, tag in enumerate(REGIME_ORDER)}
REGIME_LABELS = {
    "a050_mu010_g002": r"$0.5,0.1$",
    "a050_mu060_g002": r"$0.5,0.6$",
    "a100_mu030_g002": r"$1.0,0.3$",
    "a100_mu060_g000": r"$1.0,0.6$",
    "a150_mu010_g002": r"$1.5,0.1$",
    "a150_mu060_g020": r"$1.5,0.6$",
}

COLORS = {
    "a050_mu010_g002": "#345995",
    "a050_mu060_g002": "#2F7F6F",
    "a100_mu030_g002": "#D98C3A",
    "a100_mu060_g000": "#7F5AA2",
    "a150_mu010_g002": "#6BAFB0",
    "a150_mu060_g020": "#C84E4E",
}

COLD = "#345995"
HOT = "#C95F3F"
NEUTRAL = "#30343B"
GRID = "#E9ECEF"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
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


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.075, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 else float("nan")


def linfit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x) & np.isfinite(y)
    p = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(x[ok].min(), x[ok].max(), 120)
    return xx, p[0] * xx + p[1]


def load_stats() -> pd.DataFrame:
    stats = pd.read_csv(SUMMARY)
    stats["tag"] = pd.Categorical(stats["tag"], categories=REGIME_ORDER, ordered=True)
    return stats.sort_values("tag").reset_index(drop=True)


def draw_state_space(ax: plt.Axes, stats: pd.DataFrame, fig: plt.Figure) -> None:
    cold = stats["cold_bottom_pN_mean"]
    sizes = 72 + 470 * (cold - cold.min()) / (cold.max() - cold.min())
    cmap = LinearSegmentedColormap.from_list("hotload", ["#F5E7AD", "#E88A4A", "#A53E76", "#2B2356"])
    sc = ax.scatter(
        stats["Z_cold_N_mean"],
        stats["hot_tail_p99_over_mean"],
        s=sizes,
        c=stats["hot_bottom_pN_mean"],
        cmap=cmap,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    offsets = {
        "a050_mu010_g002": (-0.08, 0.14, "right"),
        "a050_mu060_g002": (0.04, -0.13, "left"),
        "a100_mu030_g002": (0.04, 0.06, "left"),
        "a100_mu060_g000": (0.04, 0.06, "left"),
        "a150_mu010_g002": (0.05, 0.18, "left"),
        "a150_mu060_g020": (0.06, 0.00, "left"),
    }
    for _, row in stats.iterrows():
        tag = str(row["tag"])
        dx, dy, ha = offsets[tag]
        ax.text(
            row["Z_cold_N_mean"] + dx,
            row["hot_tail_p99_over_mean"] + dy,
            f"{REGIME_IDS[tag]}  {REGIME_LABELS[tag]}",
            ha=ha,
            va="center",
            fontsize=6.1,
            color="#20242B",
        )
    ax.annotate(
        "cold reservoir",
        xy=(5.35, 6.63),
        xytext=(3.25, 6.63),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLD},
        color=COLD,
        fontsize=7,
        ha="left",
        va="center",
    )
    ax.annotate(
        "hot susceptibility",
        xy=(2.95, 7.72),
        xytext=(2.95, 6.10),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": HOT},
        color=HOT,
        fontsize=7,
        ha="center",
        va="bottom",
        rotation=90,
    )
    ax.text(0.04, 0.96, "area: residual cold load", transform=ax.transAxes, fontsize=6.2, color="#4A4F57", va="top")
    ax.set_xlim(2.82, 5.65)
    ax.set_ylim(5.75, 8.04)
    ax.set_xlabel("cold reservoir coordinate, $Z_c$")
    ax.set_ylabel(r"hot susceptibility coordinate, $f_{99}/\langle f\rangle_h$")
    finish(ax)
    panel(ax, "a")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label("hot load proxy (Pa)", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8, width=0.55)


def draw_relation(
    ax: plt.Axes,
    stats: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str,
    xlabel: str,
    ylabel: str,
    label: str,
    err_color: str,
) -> None:
    x = stats[x_col].to_numpy(float)
    y = stats[y_col].to_numpy(float)
    xx, yy = linfit(x, y)
    ax.plot(xx, yy, color=NEUTRAL, lw=1.05, alpha=0.72, zorder=1)
    for _, row in stats.iterrows():
        tag = str(row["tag"])
        ax.errorbar(
            row[x_col],
            row[y_col],
            yerr=row[yerr_col],
            fmt="o",
            ms=5.0,
            color=COLORS[tag],
            mec="white",
            mew=0.6,
            ecolor=err_color,
            elinewidth=0.75,
            capsize=2,
            zorder=3,
        )
    ax.text(0.05, 0.92, rf"$r={pearson(x, y):.2f}$", transform=ax.transAxes, fontsize=7.2, color=NEUTRAL, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label, fontsize=7.4, pad=5)
    finish(ax)


def draw_force_tail(ax: plt.Axes, stats: pd.DataFrame) -> None:
    x = np.arange(len(stats))
    for i, row in stats.iterrows():
        ax.plot([i, i], [row["cold_tail_p99_over_mean"], row["hot_tail_p99_over_mean"]], color="#B9C0C8", lw=1.0, zorder=1)
    ax.scatter(x, stats["cold_tail_p99_over_mean"], s=30, color=COLD, edgecolor="white", linewidth=0.55, label="cold", zorder=3)
    ax.scatter(x, stats["hot_tail_p99_over_mean"], s=32, color=HOT, edgecolor="white", linewidth=0.55, label="hot", zorder=4)
    ax.set_xticks(x, [REGIME_IDS[str(t)] for t in stats["tag"]])
    ax.set_xlabel("regime")
    ax.set_ylabel(r"$f_{99}/\langle f\rangle$")
    ax.set_ylim(3.0, 8.25)
    ax.set_title("force-tail broadening", fontsize=7.4, pad=5)
    ax.legend(loc="upper left", ncol=2, fontsize=5.9, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.15)
    finish(ax, axis="y")


def draw_variability(ax: plt.Axes, stats: pd.DataFrame) -> None:
    y = np.arange(len(stats))[::-1]
    cold = stats["cold_bottom_pN_cv"].to_numpy(float)
    hot = stats["hot_bottom_pN_cv"].to_numpy(float)
    for yi, c, h in zip(y, cold, hot):
        ax.plot([c, h], [yi, yi], color="#B9C0C8", lw=1.0, zorder=1)
    ax.scatter(cold, y, s=30, color=COLD, edgecolor="white", linewidth=0.55, label="cold load", zorder=3)
    ax.scatter(hot, y, s=34, color=HOT, edgecolor="white", linewidth=0.55, label="hot load", zorder=4)
    ax.set_yticks(y, [REGIME_IDS[str(t)] for t in stats["tag"]])
    ax.set_xlabel("coefficient of variation")
    ax.set_ylabel("regime")
    ax.set_xlim(0, 0.28)
    ax.set_title("route sensitivity", fontsize=7.4, pad=5)
    ax.legend(loc="upper right", fontsize=5.7, handletextpad=0.3, borderaxespad=0.15)
    finish(ax, axis="x")


def draw_readout_gradients(ax: plt.Axes) -> None:
    reg = pd.read_csv(ORTHO_REGRESSION)
    boot = pd.read_csv(ORTHO_BOOTSTRAP)
    cold = reg[reg["target"] == "cold_load"].iloc[0]
    hot = reg[reg["target"] == "hot_overload_force_delta"].iloc[0]
    angle = float(reg[reg["target"] == "cold_hot_gradient_angle"]["r2"].iloc[0])
    angle_ci = np.quantile(boot["angle_degrees"], [0.025, 0.5, 0.975])

    cold_vec = np.array([float(cold["beta_fabric"]), float(cold["beta_loop"])])
    hot_vec = np.array([float(hot["beta_fabric"]), float(hot["beta_loop"])])
    cold_vec = cold_vec / np.linalg.norm(cold_vec)
    hot_vec = hot_vec / np.linalg.norm(hot_vec)

    ax.axhline(0, color="#C7CED6", lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.axvline(0, color="#C7CED6", lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.annotate(
        "",
        xy=cold_vec,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.25, "color": COLD, "mutation_scale": 10},
        zorder=4,
    )
    ax.annotate(
        "",
        xy=hot_vec,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.25, "color": HOT, "mutation_scale": 10},
        zorder=4,
    )
    ax.scatter([cold_vec[0], hot_vec[0]], [cold_vec[1], hot_vec[1]], s=36, color=[COLD, HOT], edgecolor="white", lw=0.6, zorder=5)
    ax.text(0.66, -0.06, "cold load", color=COLD, fontsize=6.2, ha="left", va="top")
    ax.text(0.13, 0.88, "hot overload", color=HOT, fontsize=6.2, ha="left", va="center")

    arc = Arc((0, 0), 0.58, 0.58, theta1=2, theta2=88, color=NEUTRAL, lw=0.8)
    ax.add_patch(arc)
    ax.text(0.29, 0.31, rf"{angle:.0f}$^\circ$", fontsize=8.2, fontweight="bold", color=NEUTRAL, ha="center")
    ax.text(0.06, 0.72, rf"bootstrap median {angle_ci[1]:.0f}$^\circ$", transform=ax.transAxes, fontsize=5.9, color=NEUTRAL, ha="left")
    ax.text(0.06, 0.64, rf"95% CI {angle_ci[0]:.0f}--{angle_ci[2]:.0f}$^\circ$", transform=ax.transAxes, fontsize=5.9, color=NEUTRAL, ha="left")
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-0.25, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("fabric-memory axis")
    ax.set_ylabel("loop-activation axis")
    ax.set_title("readout gradients", fontsize=7.4, pad=5)
    finish(ax)


def main() -> None:
    stats = load_stats()
    setup_style()

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.25, 0.98, 0.98], height_ratios=[1.03, 0.97])

    ax = fig.add_subplot(gs[:, 0:2])
    draw_state_space(ax, stats, fig)

    ax = fig.add_subplot(gs[0, 2])
    draw_relation(ax, stats, "Z_cold_N_mean", "cold_bottom_pN_mean", "cold_bottom_pN_sd", "$Z_c$", "cold load (Pa)", "cold reservoir readout", COLD)
    panel(ax, "b")

    stats = stats.copy()
    stats["hot_force_mean_mN"] = 1e3 * stats["hot_force_mean"]
    ax = fig.add_subplot(gs[0, 3])
    draw_relation(ax, stats, "hot_force_mean_mN", "hot_bottom_pN_mean", "hot_bottom_pN_sd", r"$\langle f\rangle_h$ (mN)", "hot load (Pa)", "hot susceptibility readout", HOT)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    draw_force_tail(ax, stats)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 3])
    draw_readout_gradients(ax)
    panel(ax, "e")

    for ext in ["svg", "pdf", "png", "tiff"]:
        path = OUT / f"nphys_fig3_two_channel.{ext}"
        if ext in {"png", "tiff"}:
            fig.savefig(path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
