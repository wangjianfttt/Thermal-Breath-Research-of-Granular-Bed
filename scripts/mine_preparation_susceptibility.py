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

SEED_FILE = SRC / "nature_physics_two_channel_seed_source.csv"
SUMMARY_FILE = SRC / "nature_physics_two_channel_summary_source.csv"

REGIME_ORDER = [
    "a050_mu010_g002",
    "a050_mu060_g002",
    "a100_mu030_g002",
    "a100_mu060_g000",
    "a150_mu010_g002",
    "a150_mu060_g020",
]
REGIME_ID = {tag: f"R{i + 1}" for i, tag in enumerate(REGIME_ORDER)}
REGIME_LABEL = {
    "a050_mu010_g002": r"$0.5,0.1$",
    "a050_mu060_g002": r"$0.5,0.6$",
    "a100_mu030_g002": r"$1.0,0.3$",
    "a100_mu060_g000": r"$1.0,0.6$",
    "a150_mu010_g002": r"$1.5,0.1$",
    "a150_mu060_g020": r"$1.5,0.6$",
}

COLORS = {
    "a050_mu010_g002": "#3D6B9C",
    "a050_mu060_g002": "#2F7F6F",
    "a100_mu030_g002": "#D98C3A",
    "a100_mu060_g000": "#7E6AAE",
    "a150_mu010_g002": "#6BAFB0",
    "a150_mu060_g020": "#B6423E",
}
INK = "#252A31"
GRID = "#E7EAEE"
COLD = "#3D6B9C"
HOT = "#B6423E"
MUTED = "#8B929A"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 7.5,
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


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    vals = []
    for perm in permutations(y):
        vals.append(abs(float(spearmanr(x, np.asarray(perm)).statistic)))
    p = float(np.mean(np.asarray(vals) >= abs(rho) - 1e-12))
    return rho, p


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(SEED_FILE).copy()
    summary = pd.read_csv(SUMMARY_FILE).copy()
    for d in (seed, summary):
        d["tag"] = pd.Categorical(d["tag"], categories=REGIME_ORDER, ordered=True)
        d["regime_id"] = d["tag"].astype(str).map(REGIME_ID)
    seed = seed.sort_values(["tag", "route"]).reset_index(drop=True)
    summary = summary.sort_values("tag").reset_index(drop=True)
    summary["hot_cold_cv_ratio"] = summary["hot_bottom_pN_cv"] / (summary["cold_bottom_pN_cv"] + 1e-12)
    structure_cv = summary[["Z_cold_N_cv", "Gini_cold_N_cv", "survival_N_cv"]].mean(axis=1)
    summary["structural_cv_mean"] = structure_cv
    summary["hot_structural_cv_ratio"] = summary["hot_bottom_pN_cv"] / (structure_cv + 1e-12)
    summary["log_hot_cold_cv_ratio"] = np.log1p(summary["hot_cold_cv_ratio"])
    summary["hot_tail_excess"] = summary["hot_tail_p99_over_mean"] - summary["cold_tail_p99_over_mean"]
    summary["preparation_class"] = pd.cut(
        summary["hot_cold_cv_ratio"],
        bins=[-np.inf, 1.0, 3.0, np.inf],
        labels=["cold-dominated", "warm", "preparation-sensitive"],
    )

    tests = []
    pairs = [
        ("hot_tail_p99_over_mean", "hot_bottom_pN_cv", "hot force-tail vs hot-load CV"),
        ("hot_force_gini_direct", "hot_bottom_pN_cv", "hot force Gini vs hot-load CV"),
        ("hot_bottom_pN_mean", "hot_cold_cv_ratio", "hot-load mean vs hot/cold CV ratio"),
        ("hot_tail_p99_over_mean", "hot_cold_cv_ratio", "hot force-tail vs hot/cold CV ratio"),
        ("hot_bottom_pN_mean", "hot_structural_cv_ratio", "hot-load mean vs hot/structural CV ratio"),
    ]
    for x, y, label in pairs:
        d = summary[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p_exact = exact_spearman(d[x].to_numpy(float), d[y].to_numpy(float))
        tests.append({"relationship": label, "predictor": x, "target": y, "n": len(d), "spearman": rho, "exact_p": p_exact})
    return seed, summary, pd.DataFrame(tests)


def plot_figure(seed: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.02, 0.98])

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", x=-0.11)
    xloc = np.arange(len(REGIME_ORDER))
    jitter = {"s31001": -0.12, "s31002": 0.0, "s31003": 0.12}
    for _, row in seed.iterrows():
        x = REGIME_ORDER.index(str(row["tag"]))
        j = jitter.get(str(row["route"]), 0.0)
        ax.plot([x + j, x + j], [row["cold_bottom_pN"], row["hot_bottom_pN"]], color="#D0D6DD", lw=0.75, zorder=1)
        ax.scatter(x + j, row["cold_bottom_pN"], s=18, color=COLD, edgecolor="white", lw=0.35, alpha=0.88, zorder=3)
        ax.scatter(x + j, row["hot_bottom_pN"], s=20, color=HOT, edgecolor="white", lw=0.35, alpha=0.88, zorder=4)
    ax.set_xticks(xloc, [REGIME_ID[tag] for tag in REGIME_ORDER])
    ax.set_yscale("log")
    ax.set_xlabel("targeted regime")
    ax.set_ylabel("bottom-load proxy (Pa)")
    ax.set_title("preparation-to-preparation load scatter", loc="left", pad=4)
    ax.text(0.04, 0.96, "blue: cold\nred: hot", transform=ax.transAxes, va="top", ha="left", fontsize=6.2)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    y = np.arange(len(summary))[::-1]
    cold = summary["cold_bottom_pN_cv"].to_numpy(float)
    hot = summary["hot_bottom_pN_cv"].to_numpy(float)
    for yi, c, h in zip(y, cold, hot):
        ax.plot([c, h], [yi, yi], color="#B9C0C8", lw=0.9, zorder=1)
    ax.scatter(cold, y, s=30, color=COLD, edgecolor="white", lw=0.45, label="cold", zorder=3)
    ax.scatter(hot, y, s=34, color=HOT, edgecolor="white", lw=0.45, label="hot", zorder=4)
    ax.set_yticks(y, summary["regime_id"])
    ax.set_xlabel("coefficient of variation")
    ax.set_title("hot readout has larger preparation scatter", loc="left", pad=4)
    ax.legend(loc="lower right", ncol=2, fontsize=5.8, handletextpad=0.25, columnspacing=0.65)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    for _, row in summary.iterrows():
        tag = str(row["tag"])
        ax.scatter(
            row["hot_bottom_pN_mean"],
            row["hot_cold_cv_ratio"],
            s=44,
            color=COLORS[tag],
            edgecolor="white",
            lw=0.5,
            zorder=3,
        )
        ax.text(row["hot_bottom_pN_mean"] * 1.03, row["hot_cold_cv_ratio"], row["regime_id"], color=COLORS[tag], fontsize=6.2, va="center")
    row = tests.query("relationship == 'hot-load mean vs hot/cold CV ratio'").iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman:.2f}$" + f"\nexact P={row.exact_p:.3f}", transform=ax.transAxes, va="top", ha="left")
    ax.axhline(1, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("mean hot-load proxy (Pa)")
    ax.set_ylabel("hot/cold CV ratio")
    ax.set_title("largest loads are not most fluctuating", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    for _, row in summary.iterrows():
        tag = str(row["tag"])
        ax.scatter(
            row["hot_tail_p99_over_mean"],
            row["hot_bottom_pN_cv"],
            s=44,
            color=COLORS[tag],
            edgecolor="white",
            lw=0.5,
            zorder=3,
        )
        ax.text(row["hot_tail_p99_over_mean"] + 0.035, row["hot_bottom_pN_cv"], row["regime_id"], color=COLORS[tag], fontsize=6.2, va="center")
    row = tests.query("relationship == 'hot force-tail vs hot-load CV'").iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman:.2f}$" + f"\nexact P={row.exact_p:.3f}", transform=ax.transAxes, va="top", ha="left")
    ax.set_xlabel(r"hot force-tail width, $f_{99}/\langle f\rangle$")
    ax.set_ylabel("hot-load CV")
    ax.set_title("force-tail width weakly tracks fluctuation", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    values = summary["hot_structural_cv_ratio"].to_numpy(float)
    bars = ax.bar(np.arange(len(summary)), values, color=[COLORS[str(t)] for t in summary["tag"]], width=0.62)
    ax.set_xticks(np.arange(len(summary)), summary["regime_id"])
    ax.set_yscale("log")
    ax.set_xlabel("regime")
    ax.set_ylabel("hot-load CV / structural CV")
    ax.set_title("load scatter can exceed fabric scatter", loc="left", pad=4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.1f}", ha="center", va="bottom", fontsize=5.8)
    finish(ax, axis="y")

    out = FIG / "nphys_fig33_preparation_susceptibility"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    top_ratio = summary.sort_values("hot_cold_cv_ratio", ascending=False).head(2)
    top_struct = summary.sort_values("hot_structural_cv_ratio", ascending=False).head(1).iloc[0]
    hot_gt_cold = int((summary["hot_bottom_pN_cv"] > summary["cold_bottom_pN_cv"]).sum())
    report = f"""# Preparation-susceptibility audit

Date: 2026-06-12

## Question

The six-regime ensemble has three independently settled packings per regime. This audit asks whether preparation-to-preparation fluctuations are read out differently in the cold and hot phases. It is a fluctuation/susceptibility diagnostic, not a direct experiment and not evidence for a critical point.

## Main result

Hot-load coefficient of variation exceeds cold-load coefficient of variation in {hot_gt_cold} of 6 regimes. The hot/cold CV ratio peaks in the intermediate regimes {', '.join(top_ratio['regime_id'].astype(str))}, with ratios {', '.join(f'{v:.2f}' for v in top_ratio['hot_cold_cv_ratio'])}. The largest hot-load route is not the most fluctuating route: the exact Spearman test between mean hot load and hot/cold CV ratio is rho = {tests.query("relationship == 'hot-load mean vs hot/cold CV ratio'")['spearman'].iloc[0]:.3f}, exact P = {tests.query("relationship == 'hot-load mean vs hot/cold CV ratio'")['exact_p'].iloc[0]:.3f}.

The strongest load-versus-fabric fluctuation separation occurs in {top_struct['regime_id']}, where hot-load CV is {top_struct['hot_structural_cv_ratio']:.1f} times the mean CV of cold coordination, cold force-Gini and contact survival. This supports phase-selective preparation susceptibility: some routes convert small preparation differences into much larger hot-load differences while keeping cold fabric metrics comparatively stable.

## Tests

{tests.round(4).to_markdown(index=False)}

## Regime summary

{summary[['regime_id', 'hot_bottom_pN_mean', 'cold_bottom_pN_cv', 'hot_bottom_pN_cv', 'hot_cold_cv_ratio', 'structural_cv_mean', 'hot_structural_cv_ratio', 'hot_tail_p99_over_mean', 'hot_tail_excess']].round(4).to_markdown(index=False)}

## Interpretation allowed in the manuscript

Allowed: preparation sensitivity is phase selective and peaks in intermediate routes, consistent with an excitable rather than purely monotonic load law.

Not allowed: with only six regimes and three seeds per regime, this is not finite-size scaling, not a critical fluctuation measurement and not a universal susceptibility law.

## Generated files

- `figures/nphys_fig33_preparation_susceptibility.*`
- `source_data/nphys_preparation_susceptibility_regime_summary.csv`
- `source_data/nphys_preparation_susceptibility_tests.csv`
"""
    (ROOT / "nature_physics_preparation_susceptibility.md").write_text(report, encoding="utf-8")


def main() -> None:
    seed, summary, tests = load_tables()
    summary.to_csv(SRC / "nphys_preparation_susceptibility_regime_summary.csv", index=False)
    tests.to_csv(SRC / "nphys_preparation_susceptibility_tests.csv", index=False)
    plot_figure(seed, summary, tests)
    write_report(summary, tests)
    print("Preparation-susceptibility audit complete.")
    print(summary[["regime_id", "cold_bottom_pN_cv", "hot_bottom_pN_cv", "hot_cold_cv_ratio", "hot_structural_cv_ratio"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
