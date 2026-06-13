#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
SRC = ROOT / "source_data"

SEMI = SRC / "nphys_fig2_boundary_semi_confined_source.csv"
PRE = SRC / "nphys_fig2_boundary_precompressed_source.csv"
WALL = SRC / "nphys_fig2_boundary_wall_profile_source.csv"

COLD = "#355F91"
HOT = "#C65F42"
PRELOAD = "#C78A2D"
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


def hot_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Phase"].eq("Heating")].copy()


def packed_network(ax: plt.Axes, cx: float, cy: float, w: float, h: float, *, seed: int, node: str, label: str, accent: str | None = None) -> None:
    rng = np.random.default_rng(seed)
    cols, rows = 6, 4
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append(
                (
                    cx - 0.42 * w + c * 0.16 * w + (0.07 * w if r % 2 else 0) + rng.normal(0, 0.012 * w),
                    cy - 0.30 * h + r * 0.20 * h + rng.normal(0, 0.012 * h),
                )
            )
    pts = np.asarray(pts)
    for i, p in enumerate(pts):
        for j in range(i + 1, len(pts)):
            q = pts[j]
            if np.hypot((p[0] - q[0]) / w, (p[1] - q[1]) / h) < 0.19:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=0.5, alpha=0.70, zorder=1)
    if accent:
        for i, j in [(8, 9), (9, 15), (15, 16), (14, 20)]:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=accent, lw=1.25, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=18, color=node, edgecolor="white", lw=0.45, zorder=3)
    ax.text(cx, cy + 0.43 * h, label, ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=INK)


def draw_schematic(ax: plt.Axes, label: str = "a", compact: bool = False) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, label, x=0.0, y=1.02)
    if compact:
        ax.text(0.50, 0.97, "same reservoir, different readout", ha="center", va="top", fontsize=7.5, fontweight="bold", color=INK)
        packed_network(ax, 0.50, 0.61, 0.48, 0.43, seed=5, node="#C2B4D7", label=r"state $\mathbf{X}$")
        ax.add_patch(FancyArrowPatch((0.30, 0.23), (0.12, 0.23), arrowstyle="-|>", mutation_scale=9, lw=0.95, color=COLD))
        ax.add_patch(FancyArrowPatch((0.70, 0.23), (0.88, 0.23), arrowstyle="-|>", mutation_scale=9, lw=0.95, color=HOT))
        ax.text(0.12, 0.13, "release", ha="center", fontsize=6.2, color=COLD)
        ax.text(0.88, 0.13, "amplify", ha="center", fontsize=6.2, color=HOT)
        ax.text(0.50, 0.22, r"$\mathcal{P}_w$", ha="center", va="center", fontsize=7.0, color=INK)
        return
    ax.text(0.50, 0.96, "one trained state, two boundary projections", ha="center", va="top", fontsize=9.6, fontweight="bold", color=INK)
    packed_network(ax, 0.18, 0.52, 0.24, 0.54, seed=5, node="#C2B4D7", label="fabric reservoir")
    packed_network(ax, 0.50, 0.52, 0.24, 0.54, seed=8, node="#D88A6A", label="free path", accent=HOT)
    packed_network(ax, 0.82, 0.52, 0.24, 0.54, seed=11, node="#D8C176", label="preloaded cage", accent=PRELOAD)
    ax.add_patch(FancyArrowPatch((0.31, 0.55), (0.38, 0.55), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=HOT))
    ax.add_patch(FancyArrowPatch((0.69, 0.55), (0.62, 0.55), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=COLD))
    ax.text(0.50, 0.15, r"$\mathcal{P}_{w}$ amplifies hot side/bottom load", ha="center", fontsize=6.6, color=HOT)
    ax.text(0.82, 0.15, r"$\mathcal{P}_{w}$ releases stored preload", ha="center", fontsize=6.6, color=COLD)
    ax.add_patch(Rectangle((0.74, 0.70), 0.16, 0.012, fc=PRELOAD, ec="none", alpha=0.9))
    ax.text(0.18, 0.15, r"state $\mathbf{X}$", ha="center", fontsize=6.6, color=MUTED)


def draw_ratio(ax: plt.Axes, semi: pd.DataFrame, pre: pd.DataFrame, label: str = "b") -> None:
    semi_h = hot_rows(semi)
    pre_h = hot_rows(pre)
    semi_vec = np.array(
        [
            semi_h.SidePressure_Pa.iloc[-1] / semi_h.SidePressure_Pa.iloc[0],
            semi_h.BottomPressure_Pa.iloc[-1] / semi_h.BottomPressure_Pa.iloc[0],
        ]
    )
    pre_vec = np.array(
        [
            pre_h.LidPressure_Pa.iloc[-1] / pre_h.LidPressure_Pa.iloc[0],
            pre_h.BottomPressure_Pa.iloc[-1] / pre_h.BottomPressure_Pa.iloc[0],
        ]
    )
    ax.axvspan(1, 12, ymin=0.50, ymax=1.0, color=HOT, alpha=0.08, lw=0)
    ax.axvspan(0.002, 1, ymin=0.0, ymax=0.50, color=COLD, alpha=0.08, lw=0)
    ax.axhline(1, color="#AEB6C2", lw=0.75, ls=(0, (3, 3)))
    ax.axvline(1, color="#AEB6C2", lw=0.75, ls=(0, (3, 3)))
    for vec, color, endpoint_label, text_xy in [
        (semi_vec, HOT, "semi-confined", (2.35, 3.2)),
        (pre_vec, PRELOAD, "precompressed", (0.018, 0.055)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                (1, 1),
                (vec[0], vec[1]),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.35,
                color=color,
                shrinkA=2,
                shrinkB=2,
                zorder=3,
            )
        )
        ax.scatter(vec[0], vec[1], s=34, color=color, edgecolor="white", linewidth=0.5, zorder=4)
        ax.text(text_xy[0], text_xy[1], endpoint_label, color=color, fontsize=6.3, ha="center", va="center", clip_on=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0015, 12)
    ax.set_ylim(0.003, 8)
    ax.set_xlabel("lateral/lid readout ratio")
    ax.set_ylabel("bottom readout ratio")
    ax.set_title("boundary projection vector", loc="left", pad=8)
    finish(ax)
    panel(ax, label, x=-0.055, y=1.12)


def draw_semi_gain(ax: plt.Axes, semi: pd.DataFrame) -> None:
    heat = hot_rows(semi)
    cool = semi[semi["Phase"].eq("Cooling")]
    side_hot = heat.SidePressure_Pa / heat.SidePressure_Pa.iloc[0]
    bottom_hot = heat.BottomPressure_Pa / heat.BottomPressure_Pa.iloc[0]
    side_cold = cool.SidePressure_Pa / heat.SidePressure_Pa.iloc[0]
    bottom_cold = cool.BottomPressure_Pa / heat.BottomPressure_Pa.iloc[0]
    ax.axhline(1, color="#AEB6C2", lw=0.75, ls=(0, (3, 3)), zorder=1)
    ax.plot(heat.Cycle, side_hot, color=COLD, marker="o", ms=2.8, lw=1.15, zorder=4)
    ax.plot(heat.Cycle, bottom_hot, color=HOT, marker="o", ms=2.8, lw=1.15, zorder=4)
    ax.plot(cool.Cycle, side_cold, color=COLD, marker="s", ms=2.1, lw=0.75, alpha=0.28, zorder=2)
    ax.plot(cool.Cycle, bottom_cold, color=HOT, marker="s", ms=2.1, lw=0.75, alpha=0.28, zorder=2)
    ax.text(20.35, side_hot.iloc[-1], rf"side hot $\times${side_hot.iloc[-1]:.1f}", color=COLD, fontsize=5.9, va="center", ha="right")
    ax.text(20.35, bottom_hot.iloc[-1], rf"bottom hot $\times${bottom_hot.iloc[-1]:.1f}", color=HOT, fontsize=5.9, va="center", ha="right")
    ax.text(2.0, 5.25, "cooling returns\nremain phase-shifted", color=MUTED, fontsize=5.5, ha="left", va="center")
    ax.set_xlabel("thermal cycle")
    ax.set_ylabel(r"wall-readout gain $p_w(n)/p_w(1)$")
    ax.set_ylim(0, 7.8)
    ax.set_title("semi-confined gain", loc="left")
    finish(ax)
    panel(ax, "c")


def draw_wall(ax: plt.Axes, wall: pd.DataFrame) -> None:
    ax.fill_betweenx(1e3 * wall.z_mid_m, 0, wall.pressure_Pa, color=COLD, alpha=0.16)
    ax.plot(wall.pressure_Pa, 1e3 * wall.z_mid_m, color=COLD, lw=1.2)
    ax.set_xlabel(r"side-wall load $p_s(z)$ (Pa)")
    ax.set_ylabel(r"$z$ (mm)")
    ax.set_title("localized force path", loc="left")
    finish(ax, axis="x")
    panel(ax, "d")


def draw_release(ax: plt.Axes, pre: pd.DataFrame) -> None:
    heat = hot_rows(pre)
    cool = pre[pre["Phase"].eq("Cooling")]
    ax.plot(heat.Cycle, heat.BottomPressure_Pa / heat.BottomPressure_Pa.iloc[0], color=INK, marker="o", ms=2.8, lw=1.0, label="hot bottom")
    ax.plot(heat.Cycle, heat.LidPressure_Pa / heat.LidPressure_Pa.iloc[0], color=PRELOAD, marker="o", ms=2.8, lw=1.0, label="hot lid")
    ax.plot(cool.Cycle, cool.BottomPressure_Pa / heat.BottomPressure_Pa.iloc[0], color="#68707A", marker="s", ms=2.5, lw=0.85, label="cold bottom")
    ax.axhline(1, color="#AEB6C2", lw=0.75, ls=(0, (3, 3)))
    ax.set_xlabel("thermal cycle")
    ax.set_ylabel(r"$F/F_0$")
    ax.set_title("precompressed release", loc="left")
    ax.legend(loc="upper right", fontsize=5.4, handlelength=1.0)
    finish(ax)
    panel(ax, "e")


def main() -> None:
    setup()
    semi = pd.read_csv(SEMI)
    pre = pd.read_csv(PRE)
    wall = pd.read_csv(WALL)
    fig = plt.figure(figsize=(7.2, 4.9))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.10, top=0.95, wspace=0.48, hspace=0.58)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 1.0], width_ratios=[1.18, 1.0, 1.0])
    draw_ratio(fig.add_subplot(gs[0, 0:2]), semi, pre, label="a")
    draw_schematic(fig.add_subplot(gs[0, 2]), label="b", compact=True)
    draw_semi_gain(fig.add_subplot(gs[1, 0]), semi)
    draw_wall(fig.add_subplot(gs[1, 1]), wall)
    draw_release(fig.add_subplot(gs[1, 2]), pre)
    for ext in ["svg", "pdf", "png", "tiff"]:
        path = FIG / f"nphys_fig2_boundary.{ext}"
        fig.savefig(path, dpi=600 if ext in {"png", "tiff"} else None, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
