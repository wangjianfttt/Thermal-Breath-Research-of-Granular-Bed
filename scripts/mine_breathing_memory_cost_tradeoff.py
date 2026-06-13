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
INFILE = SRC / "nphys_breathing_response_function_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
MUTED = "#8D99A6"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
RED = "#B6423E"
VIOLET = "#7E6AAE"
GREEN = "#4F8B67"
ROUTE_COLORS = {"R1": BLUE, "R3": GOLD, "R6": RED}
ROUTE_MARKERS = {"R1": "o", "R3": "s", "R6": "^"}
SEGMENT_MARKERS = {"early": "o", "middle": "s", "late": "^"}


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def segment_label(cycle: int) -> str:
    if cycle <= 10:
        return "early"
    if cycle <= 20:
        return "middle"
    return "late"


def route_local_pareto(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pts = g[["overload_cost_positive", "memory_benefit"]].to_numpy(float)
    efficient: list[bool] = []
    deficits: list[float] = []
    for cost, benefit in pts:
        dominated = np.any(
            (pts[:, 0] <= cost)
            & (pts[:, 1] >= benefit)
            & ((pts[:, 0] < cost) | (pts[:, 1] > benefit))
        )
        best_benefit = float(pts[pts[:, 0] <= cost + 1e-12, 1].max())
        efficient.append(not dominated)
        deficits.append(best_benefit - benefit)
    return np.asarray(efficient, dtype=bool), np.asarray(deficits, dtype=float)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["segment"] = df["cycle"].astype(int).map(segment_label)
    df["memory_benefit"] = df["exhalation_imprint_norm"].astype(float)
    df["memory_benefit_plot"] = np.log1p(df["memory_benefit"])
    df["overload_cost_positive"] = np.maximum(df["response_overload_asinh"].astype(float), 0.0)
    df["imprint_efficiency"] = df["response_imprint_efficiency"].astype(float)
    df["loop_activation"] = df["response_loop_activation"].astype(float)
    df["hazard_number"] = df["response_hazard_number"].astype(float)

    parts = []
    for _, g in df.groupby("regime_id", sort=True):
        d = g.copy()
        d["benefit_rank"] = d["memory_benefit"].rank(pct=True)
        d["cost_rank"] = d["overload_cost_positive"].rank(pct=True)
        d["route_efficiency_score"] = d["benefit_rank"] - d["cost_rank"]
        eff, deficit = route_local_pareto(d)
        d["route_pareto_efficient"] = eff
        d["route_frontier_deficit"] = deficit
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    summary = (
        df.groupby(["regime_id", "segment"], observed=True)
        .agg(
            n=("cycle", "count"),
            mean_overload_cost=("overload_cost_positive", "mean"),
            mean_memory_benefit=("memory_benefit", "mean"),
            mean_memory_benefit_plot=("memory_benefit_plot", "mean"),
            mean_efficiency_score=("route_efficiency_score", "mean"),
            pareto_fraction=("route_pareto_efficient", "mean"),
            mean_frontier_deficit=("route_frontier_deficit", "mean"),
            mean_hazard_number=("hazard_number", "mean"),
            mean_loop_activation=("loop_activation", "mean"),
            mean_imprint_efficiency=("imprint_efficiency", "mean"),
            mean_overload_asinh=("response_overload_asinh", "mean"),
        )
        .reset_index()
    )

    pairs = [
        ("route_efficiency_score", "response_overload_asinh"),
        ("route_efficiency_score", "loop_activation"),
        ("route_efficiency_score", "hazard_number"),
        ("route_efficiency_score", "imprint_efficiency"),
        ("route_frontier_deficit", "hazard_number"),
        ("route_frontier_deficit", "loop_activation"),
    ]
    rows = []
    for x, y in pairs:
        d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        stat = spearmanr(d[x], d[y]) if len(d) >= 6 else None
        rows.append(
            {
                "predictor": x,
                "target": y,
                "n": int(len(d)),
                "spearman": float(stat.statistic) if stat else np.nan,
                "p_value": float(stat.pvalue) if stat else np.nan,
            }
        )
    corr = pd.DataFrame(rows)
    return df, summary, corr


def draw_arrow(ax: plt.Axes, a: pd.Series, b: pd.Series, color: str) -> None:
    ax.annotate(
        "",
        xy=(b["mean_overload_cost"], b["mean_memory_benefit_plot"]),
        xytext=(a["mean_overload_cost"], a["mean_memory_benefit_plot"]),
        arrowprops=dict(arrowstyle="-|>", lw=0.75, color=color, shrinkA=4, shrinkB=4),
        zorder=2,
    )


def make_figure(df: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.35, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 0.92], height_ratios=[1.08, 0.92])

    ax = fig.add_subplot(gs[:, 0])
    for rid, g in df.groupby("regime_id", sort=True):
        color = ROUTE_COLORS[rid]
        for seg, h in g.groupby("segment", sort=False):
            ax.scatter(
                h["overload_cost_positive"],
                h["memory_benefit_plot"],
                s=np.where(h["route_pareto_efficient"], 42, 22),
                marker=SEGMENT_MARKERS[seg],
                color=color,
                edgecolor="white",
                lw=np.where(h["route_pareto_efficient"], 0.65, 0.35),
                alpha=0.86,
                zorder=3,
            )
        front = g[g["route_pareto_efficient"]].sort_values("overload_cost_positive")
        if len(front) >= 2:
            ax.plot(front["overload_cost_positive"], front["memory_benefit_plot"], color=color, lw=0.85, alpha=0.9)
        last = g.iloc[-1]
        ax.text(
            last["overload_cost_positive"] + 0.035,
            last["memory_benefit_plot"],
            rid,
            color=color,
            fontsize=6.5,
            va="center",
        )
    ax.text(0.03, 0.95, "efficient imprint\nlow overload", transform=ax.transAxes, color=GREEN, fontsize=6.3, va="top")
    ax.text(0.66, 0.09, "costly inhale", transform=ax.transAxes, color=RED, fontsize=6.3, va="bottom")
    ax.set_xlabel(r"positive overload cost, $\max[\operatorname{asinh}(\Omega/2),0]$")
    ax.set_ylabel(r"$\log(1+\mathrm{next\ cold\ memory\ imprint})$")
    ax.set_title("route-local memory--cost landscape", loc="left", pad=4)
    finish(ax)
    panel(ax, "a", x=-0.10, y=1.04)

    ax = fig.add_subplot(gs[0, 1])
    seg_order = ["early", "middle", "late"]
    for rid, g in summary.groupby("regime_id", sort=True):
        color = ROUTE_COLORS[rid]
        g = g.set_index("segment").loc[seg_order].reset_index()
        ax.plot(g["mean_overload_cost"], g["mean_memory_benefit_plot"], color=color, lw=0.8, alpha=0.8)
        for _, row in g.iterrows():
            ax.scatter(
                row["mean_overload_cost"],
                row["mean_memory_benefit_plot"],
                s=42,
                marker=SEGMENT_MARKERS[row["segment"]],
                color=color,
                edgecolor="white",
                lw=0.45,
                zorder=3,
            )
            ax.text(row["mean_overload_cost"] + 0.018, row["mean_memory_benefit_plot"], row["segment"][0], fontsize=5.8, color=color, va="center")
        draw_arrow(ax, g.iloc[0], g.iloc[1], color)
        draw_arrow(ax, g.iloc[1], g.iloc[2], color)
    ax.set_xlabel("mean positive overload cost")
    ax.set_ylabel(r"mean $\log(1+\mathrm{imprint})$")
    ax.set_title("segment drift", loc="left", pad=4)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    g = summary.copy()
    g["route_segment"] = g["regime_id"] + " " + g["segment"].str[0]
    order = g.sort_values("mean_efficiency_score")["route_segment"].tolist()
    g = g.set_index("route_segment").loc[order].reset_index()
    colors = [ROUTE_COLORS[r.split()[0]] for r in g["route_segment"]]
    y = np.arange(len(g))
    ax.barh(y, g["mean_efficiency_score"], color=colors, alpha=0.86)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(g["route_segment"], fontsize=5.9)
    ax.set_xlabel("route-local efficiency score")
    ax.set_title("benefit rank minus cost rank", loc="left", pad=4)
    finish(ax, axis="x")
    panel(ax, "c")

    ax = fig.add_subplot(gs[:, 2])
    rows = [
        ("score vs hazard", "route_efficiency_score", "hazard_number", GREEN),
        ("score vs loop", "route_efficiency_score", "loop_activation", GREEN),
        ("score vs overload", "route_efficiency_score", "response_overload_asinh", GREEN),
        ("deficit vs hazard", "route_frontier_deficit", "hazard_number", RED),
        ("deficit vs loop", "route_frontier_deficit", "loop_activation", RED),
    ]
    labels = []
    vals = []
    pvals = []
    colors = []
    for label, pred, target, color in rows:
        row = corr[(corr["predictor"] == pred) & (corr["target"] == target)].iloc[0]
        labels.append(label)
        vals.append(row["spearman"])
        pvals.append(row["p_value"])
        colors.append(color)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, alpha=0.86)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.7)
    for yi, value, p in zip(y, vals, pvals):
        xpos = 0.36 if value >= 0 else -0.54
        ha = "left" if value >= 0 else "left"
        ax.text(xpos, yi, f"P={p:.1e}", va="center", ha=ha, fontsize=5.1, color=MUTED)
    ax.set_xlim(-0.62, 0.62)
    ax.set_xlabel("Spearman")
    ax.set_title("trade-off diagnostics", loc="left", pad=4)
    finish(ax, axis="x")
    panel(ax, "d", x=-0.24, y=1.03)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig44_breathing_memory_cost_tradeoff.{ext}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    safe = corr.copy()
    lines = [
        "# Breathing memory-cost trade-off",
        "",
        "This reserve audit asks whether the operational breathing variables contain a memory-benefit versus overload-cost structure. It uses the existing three-route breathing response table; no additional DEM result is introduced.",
        "",
        "## Definitions",
        "",
        "- Memory benefit: norm of the next-cold exhalation imprint, `exhalation_imprint_norm`.",
        "- Positive overload cost: `max(asinh(overload/2), 0)`, using the response-function overload number.",
        "- Route-local efficiency score: percentile rank of memory benefit minus percentile rank of positive overload cost within each route.",
        "- Route-local Pareto points: cycles not dominated by another cycle in the same route with both lower overload cost and higher memory benefit.",
        "",
        "## Correlation diagnostics",
        "",
        safe.round(4).to_markdown(index=False),
        "",
        "## Route/segment summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Mechanistic reading",
        "",
        "The trade-off is route-local rather than global: one early low-cost imprint point dominates the pooled Pareto envelope, so the safe inference is not a universal efficiency frontier. Within routes, high efficiency means a breath writes relatively more next-cold memory for less hot overload cost. This score is positively tied to imprint efficiency and negatively tied to loop activation, hazard and overload. R6 moves from an early costly inhale toward a later, more efficient sector, whereas R1 remains low-cost but weakly writing after early training.",
        "",
        "Interpretation boundary: this is an operational memory-cost audit, not a thermodynamic entropy-production or universal optimization law. It supports the language that breathing quality depends on both memory imprint and overload cost.",
    ]
    (ROOT / "nature_physics_breathing_memory_cost_tradeoff.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df, summary, corr = prepare()
    df.to_csv(SRC / "nphys_breathing_memory_cost_tradeoff_cycle_metrics.csv", index=False)
    summary.to_csv(SRC / "nphys_breathing_memory_cost_tradeoff_segment_summary.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_memory_cost_tradeoff_correlations.csv", index=False)
    make_figure(df, summary, corr)
    write_report(df, summary, corr)
    print("Wrote breathing memory-cost trade-off products")
    print(corr.round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
