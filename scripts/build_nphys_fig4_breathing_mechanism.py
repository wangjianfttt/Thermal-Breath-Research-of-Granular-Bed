#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#9E3D34"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
GRID = "#E8EBEF"
INK = "#252A31"
MUTED = "#8A929C"
ACCENT = "#B6423E"
COLD = "#3D6B9C"
HOT = "#C95F3F"
OVERLOAD_SCALE = 0.003


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 7.3,
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


def panel(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def within_center(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_wc"] = out[col] - out.groupby("regime_id")[col].transform("mean")
    return out


def rho_text(df: pd.DataFrame, x: str, y: str, *, within: bool = True) -> str:
    d = df[["regime_id", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if within:
        d = within_center(d, [x, y])
        x = f"{x}_wc"
        y = f"{y}_wc"
    s = spearmanr(d[x], d[y], nan_policy="omit")
    return rf"$\rho_{{within}}={s.statistic:.2f}$" + f"\nP={s.pvalue:.1e}"


def scaled_overload(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(values, dtype=float) / OVERLOAD_SCALE)


def draw_cycle_map(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "a", x=-0.05, y=1.02)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str, rad: float = 0.0, lw: float = 1.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                connectionstyle=f"arc3,rad={rad}",
                lw=lw,
                color=color,
                shrinkA=3,
                shrinkB=3,
            )
        )

    def lattice_network(
        cx: float,
        cy: float,
        w: float,
        h: float,
        *,
        node: str,
        edge: str,
        signal: str | None = None,
        seed: int = 0,
        label: str,
        label_color: str,
    ) -> None:
        rng = np.random.default_rng(seed)
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="#FBFCFE", ec="none", zorder=0))
        cols, rows = 5, 4
        pts: list[tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                x = cx - 0.40 * w + c * 0.20 * w + (0.07 * w if r % 2 else 0)
                y = cy - 0.33 * h + r * 0.22 * h
                x += rng.normal(0, 0.012 * w)
                y += rng.normal(0, 0.015 * h)
                pts.append((x, y))
        for i, (x1, y1) in enumerate(pts):
            for j, (x2, y2) in enumerate(pts):
                if j <= i:
                    continue
                d = np.hypot((x1 - x2) / w, (y1 - y2) / h)
                if d < 0.23:
                    ax.plot([x1, x2], [y1, y2], color=edge, lw=0.55, alpha=0.72, zorder=1)
        if signal:
            signal_edges = [(6, 7), (7, 12), (12, 13), (13, 8), (8, 7), (14, 18)]
            for i, j in signal_edges:
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                ax.plot([x1, x2], [y1, y2], color=signal, lw=1.45, alpha=0.92, zorder=2)
        xy = np.array(pts)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, color=node, edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(cx, cy + 0.45 * h, label, ha="center", va="bottom", fontsize=6.15, fontweight="bold", color=label_color)

    ax.text(0.50, 0.96, "memory-induced thermal breathing as a network return map", ha="center", va="top", fontsize=8.9, fontweight="bold", color=INK)
    lattice_network(0.18, 0.56, 0.23, 0.45, node="#9DB8DC", edge="#6F7B88", seed=2, label="cold reservoir", label_color=COLD)
    lattice_network(0.50, 0.56, 0.23, 0.45, node="#E9B37C", edge="#6F7B88", signal=HOT, seed=5, label="hot loop activation", label_color=HOT)
    lattice_network(0.82, 0.56, 0.23, 0.45, node="#B8A6CF", edge="#6F7B88", signal="#7A6A9B", seed=8, label="next-cold imprint", label_color="#7A6A9B")

    arrow((0.31, 0.56), (0.37, 0.56), HOT, lw=0.95)
    arrow((0.63, 0.56), (0.69, 0.56), "#B39AC6", lw=0.95)
    arrow((0.78, 0.33), (0.22, 0.33), COLD, rad=-0.18, lw=0.9)
    ax.text(0.34, 0.67, "trained\namplitude", ha="center", va="center", fontsize=6.0, color=COLD)
    ax.text(0.66, 0.67, "loop cost\nsets overload", ha="center", va="center", fontsize=6.0, color=HOT)
    ax.text(0.50, 0.18, "imprint efficiency stores the next cold memory", ha="center", va="center", fontsize=6.2, color="#7A6A9B")


def plot_amplitude(ax: plt.Axes, metrics: pd.DataFrame, parameter_summary: pd.DataFrame, label: str = "a") -> None:
    panel(ax, label, x=-0.16)
    half_life_lines = []
    for rid, g in metrics.groupby("regime_id", sort=True):
        color = COLORS.get(rid, "#6F7C8A")
        ax.plot(g["cycle"], g["breath_amplitude"], color=color, lw=1.05)
        ax.scatter(g["cycle"], g["breath_amplitude"], s=9.5, color=color, edgecolor="white", lw=0.25, zorder=3, label=rid)
        half = parameter_summary.loc[parameter_summary["regime_id"] == rid, "amplitude_half_life_cycles"].iloc[0]
        half_life_lines.append((rid, half))
    ax.set_xlim(0.5, 30.5)
    ax.set_xlabel("cycle")
    ax.set_ylabel("breathing amplitude")
    ax.set_title("trained breathing amplitude", loc="left", pad=2)
    ax.legend(ncol=1, loc="upper right", handlelength=1.0, borderaxespad=0.15)
    text = "\n".join([f"{rid} $t_{{1/2}}$={half:.0f} cycles" for rid, half in half_life_lines])
    ax.text(0.05, 0.94, text, transform=ax.transAxes, ha="left", va="top", fontsize=5.7, color=INK)
    finish(ax)


def plot_memory_overload(ax: plt.Axes, metrics: pd.DataFrame, label: str = "b") -> None:
    panel(ax, label, x=-0.22)
    x = "fabric_reservoir_index"
    y = "force_p99_hot_minus_cold"
    d = within_center(metrics, [x, y])
    d[f"{y}_wc_scaled"] = scaled_overload(d[f"{y}_wc"])
    for rid, g in d.groupby("regime_id", sort=True):
        ax.scatter(g[f"{x}_wc"], g[f"{y}_wc_scaled"], s=20, marker=MARKERS.get(rid, "o"), color=COLORS.get(rid, "#6F7C8A"), edgecolor="white", lw=0.35, alpha=0.88)
    xx = d[f"{x}_wc"].to_numpy(float)
    yy = d[f"{y}_wc_scaled"].to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    coef = np.polyfit(xx[ok], yy[ok], 1)
    line_x = np.linspace(xx[ok].min(), xx[ok].max(), 100)
    ax.plot(line_x, coef[0] * line_x + coef[1], color=INK, lw=0.75)
    ax.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax.axvline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax.text(0.05, 0.95, rho_text(metrics, x, y), transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("regime-centred fabric reservoir")
    ax.set_ylabel("scaled regime-centred\nhot overload")
    ax.set_title("cold memory sets susceptibility", loc="left", pad=2)
    finish(ax)


def plot_loop_overload(ax: plt.Axes, force_delta: pd.DataFrame, label: str = "c") -> None:
    panel(ax, label, x=-0.18)
    x = "force_h1_birth_force_share_hot_minus_cold"
    y = "force_p99_hot_minus_cold"
    d = within_center(force_delta, [x, y])
    d[f"{y}_wc_scaled"] = scaled_overload(d[f"{y}_wc"])
    for rid, g in d.groupby("regime_id", sort=True):
        ax.scatter(
            g[f"{x}_wc"],
            g[f"{y}_wc_scaled"],
            s=20,
            marker=MARKERS.get(rid, "o"),
            color=COLORS.get(rid, "#6F7C8A"),
            edgecolor="white",
            lw=0.35,
            alpha=0.88,
            label=rid,
        )
    xx = d[f"{x}_wc"].to_numpy(float)
    yy = d[f"{y}_wc_scaled"].to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    coef = np.polyfit(xx[ok], yy[ok], 1)
    line_x = np.linspace(xx[ok].min(), xx[ok].max(), 100)
    ax.plot(line_x, coef[0] * line_x + coef[1], color=INK, lw=0.75)
    ax.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax.axvline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax.text(0.05, 0.95, rho_text(force_delta, x, y), transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("regime-centred loop activation")
    ax.set_ylabel("scaled regime-centred\nhot overload")
    ax.set_title("inhale becomes dangerous via loops", loc="left", pad=2)
    ax.legend(loc="lower right", ncol=2, fontsize=5.8, handletextpad=0.25, columnspacing=0.55)
    finish(ax)


def plot_segment_buffering(ax: plt.Axes, quality_segments: pd.DataFrame, label: str = "e") -> None:
    panel(ax, label, x=-0.18)
    segment_order = {"early": 0, "middle": 1, "late": 2}
    d = quality_segments.copy()
    d["seg_order"] = d["segment"].map(segment_order)
    d = d.sort_values(["regime_id", "seg_order"])
    norm = Normalize(
        vmin=float(d["response_amplitude"].quantile(0.05)),
        vmax=float(d["response_amplitude"].quantile(0.95)),
    )
    for rid, g in d.groupby("regime_id", sort=True):
        g = g.sort_values("seg_order")
        color = COLORS.get(rid, "#6F7C8A")
        ax.plot(g["buffer_efficiency"], g["overload_asinh"], color=color, lw=0.75, alpha=0.58)
        ax.scatter(
            g["buffer_efficiency"],
            g["overload_asinh"],
            s=28 + 32 * norm(g["response_amplitude"]).clip(0, 1),
            marker=MARKERS.get(rid, "o"),
            color=color,
            edgecolor="white",
            lw=0.45,
            zorder=3,
            label=rid,
        )
        first = g.iloc[0]
        last = g.iloc[-1]
        ax.annotate(
            "",
            xy=(last["buffer_efficiency"], last["overload_asinh"]),
            xytext=(first["buffer_efficiency"], first["overload_asinh"]),
            arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": color, "alpha": 0.72, "mutation_scale": 7},
        )
    ax.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    r6e = d[(d["regime_id"] == "R6") & (d["segment"] == "early")]
    if not r6e.empty:
        row = r6e.iloc[0]
        ax.annotate(
            "early R6",
            xy=(row["buffer_efficiency"], row["overload_asinh"]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=5.8,
            color=ACCENT,
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": ACCENT},
        )
    ax.set_xlabel("buffer efficiency $\\eta$")
    ax.set_ylabel("scaled overload")
    ax.set_title("buffering controls cost", loc="left", pad=2)
    ax.legend(loc="upper right", fontsize=5.8, handletextpad=0.25, borderaxespad=0.15)
    finish(ax)


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * 0
    return (values - values.mean()) / std


def plot_breathing_hazard_state(ax: plt.Axes, quality_segments: pd.DataFrame, label: str = "a") -> None:
    panel(ax, label, x=-0.07)
    segment_order = {"early": 0, "middle": 1, "late": 2}
    d = quality_segments.copy()
    d["seg_order"] = d["segment"].map(segment_order)
    d = d.sort_values(["regime_id", "seg_order"]).reset_index(drop=True)
    cols = [
        "response_amplitude",
        "loop_activation_positive",
        "buffer_efficiency",
        "breathing_hazard_number",
        "overload_asinh",
    ]
    image = d[cols].apply(zscore, axis=0).to_numpy(float)
    im = ax.imshow(image, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto", zorder=1)
    ax.set_yticks(np.arange(len(d)), d["regime_id"] + "-" + d["segment"].str[0])
    ax.set_xticks(
        np.arange(len(cols)),
        ["inhale\nA", "loop\nL+", "buffer\n$\\eta$", "hazard\n${\\cal H}_b$", "overload"],
    )
    ax.set_title("breathing state separates memory, loop activation and overload", loc="left", pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    r6e = d[(d["regime_id"] == "R6") & (d["segment"] == "early")]
    if not r6e.empty:
        y = int(r6e.index[0])
        ax.add_patch(Rectangle((-0.48, y - 0.48), len(cols) - 0.04, 0.96, fc="none", ec=INK, lw=0.75, zorder=4))
        ax.text(4.42, y, "early R6", ha="right", va="center", fontsize=5.8, color="white", fontweight="bold")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(size=2, width=0.5)
    cbar.set_label("z-score", labelpad=2)


def main() -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    metrics = pd.read_csv(SRC / "nphys_breathing_parameter_effects_cycle_metrics.csv")
    mode_metrics = pd.read_csv(SRC / "nphys_memory_induced_breathing_cycle_metrics.csv")
    parameter_summary = pd.read_csv(SRC / "nphys_breathing_parameter_effects_summary.csv")
    force_delta = pd.read_csv(SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv")
    quality_segments = pd.read_csv(SRC / "nphys_breathing_quality_factor_route_segments.csv")
    metrics = metrics.merge(
        mode_metrics[["regime_id", "cycle", "fabric_reservoir_index"]],
        on=["regime_id", "cycle"],
        how="left",
        suffixes=("", "_mode"),
    )

    fig = plt.figure(figsize=(7.2, 4.85))
    fig.subplots_adjust(left=0.08, right=0.965, bottom=0.10, top=0.945, wspace=0.42, hspace=0.50)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 1.0], height_ratios=[1.08, 1.0])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    plot_breathing_hazard_state(ax_a, quality_segments, label="a")
    plot_amplitude(ax_b, metrics, parameter_summary, label="b")
    plot_memory_overload(ax_c, metrics, label="c")
    plot_loop_overload(ax_d, force_delta, label="d")
    plot_segment_buffering(ax_e, quality_segments, label="e")

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig4_breathing_mechanism.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/nphys_fig4_breathing_mechanism.*")


if __name__ == "__main__":
    main()
