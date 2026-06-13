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

REGIME_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R6": "#C95F3F"}
REGIME_MARKERS = {"R1": "o", "R3": "s", "R6": "^"}
GRID = "#E8EBEF"
INK = "#252A31"
MUTED = "#8A929C"

STATE_FEATURES = [
    "Z_geom",
    "force_h1_birth_force_share",
    "force_p99",
    "cycle_birth_positive_fraction",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
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


def zscore(values: pd.Series) -> pd.Series:
    sigma = values.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return values * 0
    return (values - values.mean()) / sigma


def within_spearman(df: pd.DataFrame, predictor: str, target: str) -> dict[str, float | int | str]:
    d = df[["regime_id", predictor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    raw = spearmanr(d[predictor], d[target], nan_policy="omit") if len(d) >= 6 else (np.nan, np.nan)
    dc = d.copy()
    dc[predictor] = dc[predictor] - dc.groupby("regime_id")[predictor].transform("mean")
    dc[target] = dc[target] - dc.groupby("regime_id")[target].transform("mean")
    wc = spearmanr(dc[predictor], dc[target], nan_policy="omit") if len(dc) >= 6 else (np.nan, np.nan)
    return {
        "predictor": predictor,
        "target": target,
        "spearman_raw": float(raw.statistic),
        "p_raw": float(raw.pvalue),
        "spearman_within_regime": float(wc.statistic),
        "p_within_regime": float(wc.pvalue),
        "n": int(len(d)),
    }


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(SRC / "nphys_breathing_cycle_metrics.csv")
    m = m.copy()

    for feature in STATE_FEATURES:
        values = pd.concat(
            [
                m[f"{feature}_cold"],
                m[f"{feature}_hot"],
                m[f"{feature}_cold"] + m[f"{feature}_next_cold_minus_current"],
            ]
        ).replace([np.inf, -np.inf], np.nan).dropna()
        mu = values.mean()
        sigma = values.std(ddof=0)
        for phase in ["cold", "hot"]:
            m[f"{feature}_{phase}_z"] = (m[f"{feature}_{phase}"] - mu) / sigma
        m[f"{feature}_nextcold_z"] = (m[f"{feature}_cold"] + m[f"{feature}_next_cold_minus_current"] - mu) / sigma

    cold = np.column_stack([m[f"{feature}_cold_z"] for feature in STATE_FEATURES])
    hot = np.column_stack([m[f"{feature}_hot_z"] for feature in STATE_FEATURES])
    nextcold = np.column_stack([m[f"{feature}_nextcold_z"] for feature in STATE_FEATURES])

    m["inhalation_norm"] = np.linalg.norm(hot - cold, axis=1)
    m["exhalation_imprint_norm"] = np.linalg.norm(nextcold - cold, axis=1)
    m["closure_ratio"] = m["exhalation_imprint_norm"] / (m["inhalation_norm"] + 1e-12)

    x0 = m["Z_geom_cold_z"]
    y0 = m["force_h1_birth_force_share_cold_z"]
    x1 = m["Z_geom_hot_z"]
    y1 = m["force_h1_birth_force_share_hot_z"]
    x2 = m["Z_geom_nextcold_z"]
    y2 = m["force_h1_birth_force_share_nextcold_z"]
    signed_area = 0.5 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    m["signed_breathing_area"] = signed_area
    m["breathing_triangle_area"] = signed_area.abs()

    for col in ["Z_geom_cold", "cold_cycle_jaccard", "next_cold_cycle_jaccard"]:
        m[f"{col}_z"] = zscore(m[col])
    m["fabric_reservoir_index"] = m[
        ["Z_geom_cold_z", "cold_cycle_jaccard_z", "next_cold_cycle_jaccard_z"]
    ].mean(axis=1, skipna=True)

    m["segment"] = pd.cut(m["cycle"], bins=[0, 10, 20, 29], labels=["early", "middle", "late"])

    corr_pairs = [
        ("cycle", "inhalation_norm"),
        ("cycle", "closure_ratio"),
        ("fabric_reservoir_index", "inhalation_norm"),
        ("fabric_reservoir_index", "force_h1_birth_force_share_hot_minus_cold"),
        ("fabric_reservoir_index", "force_p99_hot_minus_cold"),
        ("Z_geom_cold", "force_h1_birth_force_share_hot_minus_cold"),
        ("next_cold_cycle_jaccard", "force_p99_hot_minus_cold"),
        ("breathing_triangle_area", "force_p99_hot_minus_cold"),
        ("closure_ratio", "force_p99_hot_minus_cold"),
    ]
    corr = pd.DataFrame([within_spearman(m, predictor, target) for predictor, target in corr_pairs])

    summary = (
        m.groupby(["regime_id", "segment"], observed=True)
        .agg(
            n=("cycle", "count"),
            inhalation_norm_mean=("inhalation_norm", "mean"),
            exhalation_imprint_norm_mean=("exhalation_imprint_norm", "mean"),
            closure_ratio_mean=("closure_ratio", "mean"),
            breathing_area_mean=("breathing_triangle_area", "mean"),
            loop_activation_mean=("force_h1_birth_force_share_hot_minus_cold", "mean"),
            overload_mean=("force_p99_hot_minus_cold", "mean"),
            fabric_reservoir_mean=("fabric_reservoir_index", "mean"),
        )
        .reset_index()
    )
    return m, corr, summary


def plot_figure(metrics: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.43)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel(ax_a, "a")
    for rid, g in metrics.groupby("regime_id", sort=True):
        ax_a.plot(g["cycle"], g["inhalation_norm"], color=REGIME_COLORS[rid], lw=1.05, alpha=0.88)
        ax_a.scatter(g["cycle"], g["inhalation_norm"], s=12, color=REGIME_COLORS[rid], edgecolor="white", lw=0.25, zorder=3)
        ax_a.text(g["cycle"].max() + 0.3, g["inhalation_norm"].iloc[-1], rid, color=REGIME_COLORS[rid], fontsize=6.6, va="center")
    rho = corr[(corr["predictor"] == "cycle") & (corr["target"] == "inhalation_norm")]["spearman_within_regime"].iloc[0]
    p = corr[(corr["predictor"] == "cycle") & (corr["target"] == "inhalation_norm")]["p_within_regime"].iloc[0]
    ax_a.text(0.70, 0.90, rf"$\rho_{{within}}={rho:.2f}$" + f"\nP={p:.1e}", transform=ax_a.transAxes, ha="left", va="top")
    ax_a.set_xlabel("cycle")
    ax_a.set_ylabel("inhalation amplitude\nstandardized state distance")
    ax_a.set_title("ratcheting trains the breathing amplitude", loc="left", pad=2)
    ax_a.set_xlim(0.5, 31.2)
    finish(ax_a)

    panel(ax_b, "b")
    plot_data = metrics.copy()
    x = "fabric_reservoir_index"
    y = "force_p99_hot_minus_cold"
    plot_data[f"{x}_wc"] = plot_data[x] - plot_data.groupby("regime_id")[x].transform("mean")
    plot_data[f"{y}_wc"] = plot_data[y] - plot_data.groupby("regime_id")[y].transform("mean")
    for rid, g in plot_data.groupby("regime_id", sort=True):
        ax_b.scatter(
            g[f"{x}_wc"],
            g[f"{y}_wc"],
            s=23,
            marker=REGIME_MARKERS[rid],
            color=REGIME_COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.86,
            label=rid,
        )
    xx = plot_data[f"{x}_wc"].to_numpy(float)
    yy = plot_data[f"{y}_wc"].to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    coef = np.polyfit(xx[ok], yy[ok], 1)
    line_x = np.linspace(xx[ok].min(), xx[ok].max(), 100)
    ax_b.plot(line_x, coef[0] * line_x + coef[1], color=INK, lw=0.75)
    row = corr[(corr["predictor"] == x) & (corr["target"] == y)].iloc[0]
    ax_b.text(
        0.04,
        0.96,
        rf"$\rho_{{within}}={row.spearman_within_regime:.2f}$" + f"\nP={row.p_within_regime:.1e}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
    )
    ax_b.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax_b.axvline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax_b.set_xlabel("regime-centred fabric reservoir")
    ax_b.set_ylabel("regime-centred hot overload")
    ax_b.set_title("cold memory sets hot susceptibility", loc="left", pad=2)
    finish(ax_b)

    panel(ax_c, "c")
    segments = ["early", "middle", "late"]
    for rid, g in summary.groupby("regime_id", sort=True):
        g = g.set_index("segment").loc[segments].reset_index()
        ax_c.plot(
            g["inhalation_norm_mean"],
            g["closure_ratio_mean"],
            marker=REGIME_MARKERS[rid],
            ms=5,
            lw=1.05,
            color=REGIME_COLORS[rid],
        )
        for _, row in g.iterrows():
            ax_c.text(row["inhalation_norm_mean"] + 0.035, row["closure_ratio_mean"], str(row["segment"])[0], color=REGIME_COLORS[rid], fontsize=6.2)
        ax_c.text(g["inhalation_norm_mean"].iloc[-1] + 0.05, g["closure_ratio_mean"].iloc[-1], rid, color=REGIME_COLORS[rid], fontsize=6.6, va="center")
    ax_c.set_xlabel("mean inhalation amplitude")
    ax_c.set_ylabel("imprint / inhalation ratio")
    ax_c.set_title("three breathing modes, not one limit cycle", loc="left", pad=2)
    finish(ax_c)

    panel(ax_d, "d")
    matrix_cols = ["inhalation_norm_mean", "closure_ratio_mean", "loop_activation_mean", "overload_mean"]
    mat = summary.copy()
    for col in matrix_cols:
        mat[col] = (mat[col] - mat[col].mean()) / mat[col].std(ddof=0)
    mat["row"] = mat["regime_id"] + "-" + mat["segment"].astype(str).str[0]
    image = mat[matrix_cols].to_numpy(float)
    im = ax_d.imshow(image, aspect="auto", cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    ax_d.set_yticks(np.arange(len(mat)))
    ax_d.set_yticklabels(mat["row"])
    ax_d.set_xticks(np.arange(len(matrix_cols)))
    ax_d.set_xticklabels(["inhale", "imprint\nratio", "loop", "overload"], rotation=0)
    ax_d.set_title("route-specific breathing signature", loc="left", pad=2)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(size=2, width=0.5)
    cbar.set_label("z-score", labelpad=2)

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig13_memory_induced_breathing_modes.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Memory-induced breathing modes",
        "",
        "This audit asks whether the breathing-like picture is more than wording. The analysis defines breathing as a cycle map from cold state to hot state to next cold state in a standardized fabric-loop-force space.",
        "",
        "## Operational metrics",
        "",
        "- Inhalation amplitude: standardized state distance from cold to hot using fabric, force-loop share, force p99 and positive-cycle fraction.",
        "- Exhalation imprint: standardized state distance from current cold to next cold.",
        "- Closure ratio: exhalation imprint divided by inhalation amplitude.",
        "- Breathing area: triangle area in fabric-loop coordinates for cold, hot and next-cold states.",
        "- Fabric reservoir index: standardized cold coordination plus cold-to-cold contact-retention diagnostics.",
        "",
        "## Correlation audit",
        "",
        corr.to_markdown(index=False),
        "",
        "## Early-middle-late breathing modes",
        "",
        summary.to_markdown(index=False),
        "",
        "## Main interpretation",
        "",
        "The data support a memory-induced breathing picture, but not a single universal limit cycle. The common feature is training: the cold-to-hot inhalation amplitude decreases with cycle within regimes. The cold fabric reservoir also predicts hot loop activation and hot overload after regime centering, which supports the idea that memory controls thermal susceptibility. The routes then split into breathing modes: R1 is buffered and closes rapidly, R3 is a sustained lossy breather, and R6 shows a violent early inhalation with large overload followed by relaxation and a larger late imprint fraction.",
        "",
        "## Manuscript use",
        "",
        "Use this as the story line: thermal ratcheting trains a memory reservoir; subsequent cycles breathe through that reservoir; the hot inhale is dangerous when it activates force loops; the cold exhale retains a route-dependent imprint. Avoid claiming a universal limit cycle or that contact turnover alone causes overload.",
    ]
    (ROOT / "nature_physics_memory_induced_breathing_modes_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metrics, corr, summary = build_metrics()
    metrics.to_csv(SRC / "nphys_memory_induced_breathing_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_memory_induced_breathing_correlations.csv", index=False)
    summary.to_csv(SRC / "nphys_memory_induced_breathing_segment_summary.csv", index=False)
    plot_figure(metrics, corr, summary)
    write_report(metrics, corr, summary)
    print(corr[["predictor", "target", "spearman_within_regime", "p_within_regime", "n"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
