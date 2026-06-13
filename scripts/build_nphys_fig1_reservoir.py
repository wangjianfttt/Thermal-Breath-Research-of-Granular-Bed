#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
SRC = ROOT / "source_data"
DATA = SRC / "nphys_fig1_memory_source.csv"
TRAINING_DATA = SRC / "nphys_training_order_parameter_flow_cycle_metrics.csv"
TRAINING_CORR = SRC / "nphys_training_order_parameter_flow_correlations.csv"

COLD = "#355F91"
HOT = "#C65F42"
FABRIC = "#2F7F6F"
AGED = "#B89645"
PURPLE = "#7A6A9B"
INK = "#272C33"
MUTED = "#7C858F"
EDGE = "#9EA7B2"
GRID = "#E8EBEF"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.2,
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


def panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color="black")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def network_points(cx: float, cy: float, w: float, h: float, seed: int, jitter: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols, rows = 7, 4
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx - 0.43 * w + c * 0.145 * w + (0.065 * w if r % 2 else 0)
            y = cy - 0.34 * h + r * 0.23 * h
            pts.append((x + rng.normal(0, jitter * w), y + rng.normal(0, jitter * h)))
    return np.asarray(pts)


def draw_network(ax: plt.Axes, cx: float, cy: float, w: float, h: float, *, seed: int, node: str, label: str, highlight: str | None = None, compact: float = 0.0) -> None:
    pts = network_points(cx, cy - compact * h, w, h * (1 - 0.10 * compact), seed, 0.012 + 0.006 * compact)
    for i, p in enumerate(pts):
        for j in range(i + 1, len(pts)):
            q = pts[j]
            d = np.hypot((p[0] - q[0]) / w, (p[1] - q[1]) / h)
            if d < 0.17:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=0.55, alpha=0.78, zorder=1)
    if highlight:
        for i, j in [(9, 10), (10, 17), (17, 18), (18, 11), (11, 10)]:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=highlight, lw=1.35, alpha=0.95, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=20, color=node, edgecolor="white", lw=0.45, zorder=3)
    ax.text(cx, cy - 0.62 * h, label, ha="center", va="top", fontsize=7.1, fontweight="bold", color=node)


def draw_training_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "a", x=0.0, y=1.02)
    ax.text(0.50, 0.94, "thermal cycling trains a contact-network reservoir", ha="center", va="top", fontsize=9.8, fontweight="bold", color=INK)
    draw_network(ax, 0.19, 0.52, 0.26, 0.58, seed=10, node="#92ABD1", label="loose cold")
    draw_network(ax, 0.50, 0.52, 0.26, 0.58, seed=13, node="#D88A6A", label="hot inhale", highlight=HOT, compact=-0.1)
    draw_network(ax, 0.81, 0.52, 0.26, 0.58, seed=16, node="#D4B96A", label="aged cold reservoir", compact=0.12)
    for x0, x1 in [(0.33, 0.38), (0.64, 0.69)]:
        ax.add_patch(FancyArrowPatch((x0, 0.52), (x1, 0.52), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED))


def draw_readouts(ax: plt.Axes, df: pd.DataFrame) -> None:
    first, last = df.iloc[0], df.iloc[-1]
    labels = ["height", r"$k_{\rm eff}$", r"$Z_c$", "survival", r"$\sigma_{\phi}$"]
    ratios = [
        last.BedHeight_m / first.BedHeight_m,
        last.k_eff_W_mK / first.k_eff_W_mK,
        last.z_mean / first.z_mean,
        last.contact_survival / first.contact_survival,
        last.phi_voro_std / first.phi_voro_std,
    ]
    colors = [COLD, FABRIC, COLD, HOT, PURPLE]
    y = np.arange(len(labels))[::-1]
    for yi, r, col in zip(y, ratios, colors):
        ax.plot([1, r], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
        ax.scatter([1, r], [yi, yi], s=26, color=col, edgecolor="white", linewidth=0.45, zorder=3)
        dy = -0.24 if yi == y.max() and r < 1 else 0.13
        ax.text(r + (0.013 if r >= 1 else -0.013), yi + dy, f"{r:.2f}x", color=col, ha="left" if r >= 1 else "right", fontsize=6.2)
    ax.axvline(1, color="#AEB6C2", lw=0.75, ls=(0, (3, 3)))
    ax.set_yticks(y, labels)
    ax.set_xlim(0.86, 1.34)
    ax.set_xlabel("cycle 20 / cycle 1")
    ax.set_title("reservoir readouts", loc="left")
    finish(ax, axis="x")
    panel(ax, "b")


def draw_aging(ax: plt.Axes, df: pd.DataFrame) -> None:
    z_norm = df.z_mean / df.z_mean.iloc[0]
    survival_norm = df.contact_survival / df.contact_survival.iloc[0]
    ax.plot(df.cycle, z_norm, color=COLD, marker="o", ms=2.9, lw=1.05, label=r"$Z_c/Z_c(1)$")
    ax.plot(df.cycle, survival_norm, color=HOT, marker="s", ms=2.7, lw=1.0, label=r"$S/S(1)$")
    ax.set_xlabel("cycle")
    ax.set_ylabel("cycle-normalised reservoir metric")
    ax.legend(loc="lower right", fontsize=5.7, handlelength=1.0)
    ax.set_title("contact graph ages", loc="left")
    finish(ax)
    panel(ax, "c")


def draw_rearrangement(ax: plt.Axes, df: pd.DataFrame) -> None:
    created = df.contacts_created / df.contacts_created.iloc[0]
    broken = df.contacts_broken / df.contacts_broken.iloc[0]
    rms = df.rms_displacement_m / df.rms_displacement_m.iloc[0]
    ax.plot(df.cycle, created, color=FABRIC, marker="o", ms=2.6, lw=0.95, label="created contacts")
    ax.plot(df.cycle, broken, color=HOT, marker="s", ms=2.6, lw=0.95, label="broken contacts")
    ax.plot(df.cycle, rms, color=PURPLE, marker="D", ms=2.4, lw=0.9, label=r"$\Delta r_{\rm rms}$")
    ax.set_yscale("log")
    ax.set_xlabel("cycle")
    ax.set_ylabel("cycle / cycle 1")
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.0)
    ax.set_title("exploration damps", loc="left")
    finish(ax)
    panel(ax, "d")


def draw_free_volume(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.plot(df.cycle, df.phi_voro_mean / df.phi_voro_mean.iloc[0], color=FABRIC, marker="o", ms=2.6, lw=0.95, label=r"$\langle\phi_V\rangle$")
    ax.plot(df.cycle, df.free_volume_entropy / df.free_volume_entropy.iloc[0], color=INK, marker="s", ms=2.6, lw=0.95, label=r"$H_V$")
    ax.plot(df.cycle, df.phi_voro_std / df.phi_voro_std.iloc[0], color=PURPLE, marker="D", ms=2.4, lw=0.9, label=r"$\sigma(\phi_V)$")
    ax.set_xlabel("cycle")
    ax.set_ylabel("normalised value")
    ax.set_title("free-volume distribution narrows", loc="left")
    ax.legend(loc="lower left", fontsize=5.5, handlelength=1.0)
    finish(ax)
    panel(ax, "e")


def draw_training_flow(ax: plt.Axes, training: pd.DataFrame, corr: pd.DataFrame) -> None:
    sc = ax.scatter(
        training["activity_index"],
        training["memory_index"],
        c=training["cycle"],
        cmap="viridis",
        s=24 + 30 * training["cycle_fraction"],
        edgecolor="white",
        lw=0.5,
        zorder=4,
    )
    ax.plot(training["activity_index"], training["memory_index"], color="#AEB5BE", lw=0.95, zorder=2)
    offsets = {1: (0.018, -0.035), 2: (0.028, -0.025), 5: (0.024, -0.005), 10: (0.024, 0.012), 20: (0.030, 0.018)}
    for cyc in [1, 2, 5, 10, 20]:
        row = training.loc[training["cycle"] == cyc].iloc[0]
        dx, dy = offsets[cyc]
        ax.text(row["activity_index"] + dx, row["memory_index"] + dy, str(cyc), fontsize=5.9, va="center")
    ax.annotate(
        "training flow",
        xy=(training["activity_index"].iloc[7], training["memory_index"].iloc[7]),
        xytext=(0.42, 0.34),
        arrowprops={"arrowstyle": "->", "lw": 0.75, "color": INK, "connectionstyle": "arc3,rad=-0.12"},
        fontsize=6.6,
        color=INK,
    )
    rho = corr.query("x == 'activity_index' and y == 'memory_index'")["spearman_rho"].iloc[0]
    ax.text(
        0.98,
        0.93,
        rf"$\rho={rho:.2f}$" + "\n" + r"$t_{1/2}^A\ll t_{1/2}^M$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color=INK,
        bbox={"fc": "white", "ec": "none", "alpha": 0.82, "pad": 1.2},
    )
    ax.set_xlim(-0.035, 1.08)
    ax.set_ylim(-0.045, 1.065)
    ax.set_xlabel("activity order parameter, $A$")
    ax.set_ylabel("memory order parameter, $M$")
    ax.set_title("activity extinguishes as memory is written", loc="left")
    finish(ax)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(size=2, width=0.5, labelsize=5.6)
    cbar.set_label("cycle", labelpad=2, fontsize=5.9)
    panel(ax, "e")


def main() -> None:
    setup()
    df = pd.read_csv(DATA)
    training = pd.read_csv(TRAINING_DATA)
    corr = pd.read_csv(TRAINING_CORR)
    fig = plt.figure(figsize=(7.2, 5.9))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.08, top=0.96, wspace=0.38, hspace=0.50)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 1.0])
    draw_training_schematic(fig.add_subplot(gs[0, :]))
    draw_readouts(fig.add_subplot(gs[1, 0]), df)
    draw_aging(fig.add_subplot(gs[1, 1]), df)
    draw_rearrangement(fig.add_subplot(gs[2, 0]), df)
    draw_training_flow(fig.add_subplot(gs[2, 1]), training, corr)
    for ext in ["svg", "pdf", "png", "tiff"]:
        path = FIG / f"nphys_fig1_memory.{ext}"
        fig.savefig(path, dpi=600 if ext in {"png", "tiff"} else None, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
