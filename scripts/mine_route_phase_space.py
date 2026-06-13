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

MATRIX = SRC / "fig4_regime_matrix_source.csv"
TARGETED = SRC / "nature_physics_two_channel_summary_source.csv"
LONG = SRC / "fig5_long_cycle_source.csv"

OUT_CASE = SRC / "nphys_route_phase_space_27case.csv"
OUT_SUMMARY = SRC / "nphys_route_phase_space_summary.csv"
OUT_REPORT = ROOT / "nature_physics_route_phase_space_report.md"

COLD = "#345995"
HOT = "#C95F3F"
LOSSY = "#D98C3A"
BUFFER = "#2F7F6F"
NEUTRAL = "#2B3036"
GRID = "#E8EBEF"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def route_label(row: pd.Series) -> str:
    c = row["cold_memory_index"]
    h = row["hot_susceptibility_index"]
    l = row["loss_index"]
    if h >= 0.75 and c <= 0.25:
        return "excitable overload"
    if c >= 0.75 and h < 0.75:
        return "buffered reservoir"
    if l >= 0.70 or (h >= 0.15 and c >= 0.15):
        return "lossy transitional"
    return "weakly coupled"


def build_phase_space() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(MATRIX)
    m["cold_memory_index"] = (
        zscore(m["Z_cold_N"])
        + zscore(m["survival_N"])
        - zscore(m["force_gini_cold_N"])
        - 0.35 * zscore(m["cold_porosity_N"])
    ) / 3.35
    m["hot_susceptibility_index"] = (
        zscore(m["hot_bottom_pN"])
        + zscore(m["hot_side_pN_abs"])
        + zscore(m["work_proxy_total"])
    ) / 3.0
    m["loss_index"] = (
        zscore(m["work_proxy_total"])
        + zscore(m["force_gini_cold_N"])
        - zscore(m["survival_N"])
    ) / 3.0
    m["projection_angle_rad"] = np.arctan2(m["hot_susceptibility_index"], m["cold_memory_index"])
    m["projection_norm"] = np.hypot(m["hot_susceptibility_index"], m["cold_memory_index"])
    m["route_class"] = m.apply(route_label, axis=1)
    m["selected_for_targeted_ensemble"] = m["tag"].isin(pd.read_csv(TARGETED)["tag"])
    long_tags = pd.read_csv(LONG)["tag"].str.replace("_c30", "", regex=False)
    m["long_cycle_true_force_route"] = m["tag"].isin(set(long_tags))
    summary = (
        m.groupby(["alpha_mult", "friction", "lid_gap_radii"], as_index=False)
        .agg(
            cold_memory_index=("cold_memory_index", "mean"),
            hot_susceptibility_index=("hot_susceptibility_index", "mean"),
            loss_index=("loss_index", "mean"),
            projection_norm=("projection_norm", "mean"),
            route_class=("route_class", lambda s: s.mode().iloc[0]),
        )
        .sort_values(["alpha_mult", "friction", "lid_gap_radii"])
    )
    return m, summary


def draw_heatmaps(fig: plt.Figure, gs, data: pd.DataFrame, value: str, title: str, cmap: str, vlim: float, panel_label: str) -> None:
    gaps = sorted(data["lid_gap_radii"].unique())
    axes = []
    for i, gap in enumerate(gaps):
        ax = fig.add_subplot(gs[i])
        sub = data[data["lid_gap_radii"].eq(gap)]
        piv = sub.pivot(index="friction", columns="alpha_mult", values=value).sort_index(ascending=False)
        im = ax.imshow(piv.to_numpy(float), cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_xticks(np.arange(len(piv.columns)), [f"{x:.1f}" for x in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)), [f"{x:.1f}" for x in piv.index])
        ax.set_xlabel(r"expansion multiplier $\alpha/\alpha_0$")
        if i == 0:
            ax.set_ylabel(r"friction $\mu$")
            panel(ax, panel_label)
        else:
            ax.set_ylabel("")
        ax.set_title(f"{title}, gap={gap:.2f}R", loc="left", pad=2)
        for y in range(piv.shape[0]):
            for x in range(piv.shape[1]):
                val = piv.to_numpy(float)[y, x]
                ax.text(x, y, f"{val:.1f}", ha="center", va="center", fontsize=5.8, color="black")
        axes.append(ax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.012)
    cbar.ax.tick_params(size=2, width=0.5)


def draw_projection(ax: plt.Axes, data: pd.DataFrame) -> None:
    panel(ax, "c")
    colors = {
        "buffered reservoir": BUFFER,
        "lossy transitional": LOSSY,
        "excitable overload": HOT,
        "weakly coupled": "#87919D",
    }
    markers = {0.0: "o", 0.02: "s", 0.2: "^"}
    for klass, g in data.groupby("route_class"):
        for gap, gg in g.groupby("lid_gap_radii"):
            ax.scatter(
                gg["cold_memory_index"],
                gg["hot_susceptibility_index"],
                s=34 + 42 * gg["friction"],
                marker=markers[gap],
                color=colors[klass],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.9,
                label=klass if gap == sorted(data["lid_gap_radii"].unique())[0] else None,
            )
    selected = data[data["selected_for_targeted_ensemble"]]
    ax.scatter(selected["cold_memory_index"], selected["hot_susceptibility_index"], s=92, facecolor="none", edgecolor=NEUTRAL, linewidth=0.85, label="targeted")
    long = data[data["long_cycle_true_force_route"]]
    for _, row in long.iterrows():
        ax.text(row["cold_memory_index"] + 0.05, row["hot_susceptibility_index"], row["tag"].split("_g")[0].replace("a", "A").replace("_mu", ", M"), fontsize=5.8, va="center")
    ax.axhline(0, color="#B7BDC5", lw=0.6, ls=(0, (3, 3)))
    ax.axvline(0, color="#B7BDC5", lw=0.6, ls=(0, (3, 3)))
    ax.set_xlabel("cold-memory coordinate")
    ax.set_ylabel("hot-susceptibility coordinate")
    ax.set_title("27-case route phase space", loc="left", pad=2)
    ax.grid(True, color=GRID, lw=0.45)
    ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.2, borderaxespad=0.2)


def draw_route_counts(ax: plt.Axes, data: pd.DataFrame) -> None:
    panel(ax, "d")
    order = ["buffered reservoir", "lossy transitional", "excitable overload", "weakly coupled"]
    counts = pd.crosstab(data["alpha_mult"], data["route_class"]).reindex(columns=order, fill_value=0)
    bottom = np.zeros(len(counts))
    colors = {
        "buffered reservoir": BUFFER,
        "lossy transitional": LOSSY,
        "excitable overload": HOT,
        "weakly coupled": "#87919D",
    }
    x = np.arange(len(counts))
    for klass in order:
        vals = counts[klass].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=colors[klass], width=0.62, label=klass)
        bottom += vals
    ax.set_xticks(x, [f"{v:.1f}" for v in counts.index])
    ax.set_xlabel(r"expansion multiplier $\alpha/\alpha_0$")
    ax.set_ylabel("number of cases")
    ax.set_title("route classes shift with expansion", loc="left", pad=2)
    ax.legend(loc="upper left", fontsize=5.4, handlelength=0.8)
    ax.grid(True, axis="y", color=GRID, lw=0.45)


def plot(data: pd.DataFrame) -> None:
    setup()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.25))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.95, wspace=0.42, hspace=0.62)
    outer = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.08])
    draw_heatmaps(fig, [outer[0, i] for i in range(3)], data, "cold_memory_index", "cold memory", "PuBuGn", 1.6, "a")
    draw_heatmaps(fig, [outer[1, i] for i in range(3)], data, "hot_susceptibility_index", "hot susceptibility", "OrRd", 1.6, "b")
    draw_projection(fig.add_subplot(outer[2, :2]), data)
    draw_route_counts(fig.add_subplot(outer[2, 2]), data)
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig15_route_phase_space.{ext}", dpi=450 if ext in {"png", "tiff"} else None, bbox_inches="tight")
    plt.close(fig)


def write_report(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    corr = data[["cold_memory_index", "hot_susceptibility_index", "loss_index", "hot_bottom_pN", "cold_bottom_pN", "hot_side_pN_abs", "Z_cold_N", "survival_N", "force_gini_cold_N"]].corr(method="spearman")
    top_hot = data.sort_values("hot_susceptibility_index", ascending=False).head(5)
    top_cold = data.sort_values("cold_memory_index", ascending=False).head(5)
    counts = data["route_class"].value_counts().rename_axis("route_class").reset_index(name="n")
    lines = [
        "# Route phase-space audit",
        "",
        "This audit asks whether the route-conditioned breathing mechanism has a wider parameter-space envelope in the existing 27-case regime matrix. It does not use force-loop variables and therefore should not be used as independent proof of the loop mechanism. Its role is to map where the macroscopic reservoir and hot-susceptibility readouts live.",
        "",
        "## Definitions",
        "",
        "- Cold-memory coordinate: standardized cold coordination plus contact survival, minus cold force heterogeneity and a smaller porosity penalty.",
        "- Hot-susceptibility coordinate: standardized final hot bottom load, hot side load and total work proxy.",
        "- Loss coordinate: standardized work and cold force heterogeneity, minus contact survival.",
        "",
        "## Route counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## Strongest hot-susceptibility cases",
        "",
        top_hot[["tag", "alpha_mult", "friction", "lid_gap_radii", "cold_memory_index", "hot_susceptibility_index", "loss_index", "route_class"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Strongest cold-memory cases",
        "",
        top_cold[["tag", "alpha_mult", "friction", "lid_gap_radii", "cold_memory_index", "hot_susceptibility_index", "loss_index", "route_class"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Spearman correlations among route coordinates and observables",
        "",
        corr.to_markdown(floatfmt=".3f"),
        "",
        "## Manuscript-safe interpretation",
        "",
        "The 27-case matrix supports a parameter-space envelope for the route-conditioned mechanism. Low-expansion cases occupy memory-rich, buffered regions with high cold coordination and survival. High-expansion/high-friction cases move toward a hot-susceptibility/loss sector, consistent with the long-cycle R6 route. The matrix does not by itself prove force-loop causality, because true pair-force loop activation was only measured in selected long-cycle routes. It should be presented as a route map that motivates where full force-network reruns are needed.",
        "",
        "Recommended use: Extended Data or supplementary mechanism audit. A single sentence can be added to the main text: the selected long-cycle routes span the reservoir-rich, transitional and hot-susceptible sectors of the broader 27-case matrix.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data, summary = build_phase_space()
    data.to_csv(OUT_CASE, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(data)
    write_report(data, summary)
    print(OUT_CASE)
    print(OUT_SUMMARY)
    print(FIG / "nphys_fig15_route_phase_space.*")
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
