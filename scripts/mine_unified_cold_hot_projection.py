#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INK = "#252A31"
COLD = "#345995"
HOT = "#C95F3F"
NEUTRAL = "#8B929A"
GRID = "#E7EAEE"


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
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: plt.Axes, label: str, x: float = -0.13) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype=float)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def corr_row(scope: str, x_name: str, y_name: str, x: pd.Series, y: pd.Series) -> dict[str, float | str | int]:
    xx = pd.Series(x, dtype=float)
    yy = pd.Series(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if len(xx) < 4:
        return {
            "scope": scope,
            "predictor": x_name,
            "target": y_name,
            "n": int(len(xx)),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }
    sp = spearmanr(xx, yy)
    pr = pearsonr(xx, yy)
    return {
        "scope": scope,
        "predictor": x_name,
        "target": y_name,
        "n": int(len(xx)),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }


def load_six_regime_projection() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SRC / "nature_physics_two_channel_summary_source.csv")
    force = pd.read_csv(SRC / "nphys_true_force_percolation_hot_cold_delta.csv")
    six = summary.merge(force, on="tag", how="left", suffixes=("", "_force"))
    six["fabric_memory_index"] = (
        zscore(six["Z_cold_N_mean"])
        + zscore(six["survival_N_mean"])
        - zscore(six["cold_force_gini_direct"])
    )
    six["loop_activation_index"] = (
        zscore(six["force_h1_birth_force_share_hot_minus_cold"])
        + zscore(six["force_p99_hot_minus_cold"])
    )
    six["cold_load"] = six["cold_bottom_pN_mean"]
    six["hot_load"] = six["hot_bottom_pN_mean"]
    six["hot_overload_force_delta"] = six["force_p99_hot_minus_cold"]

    rows = []
    for target in ["cold_load", "hot_load", "hot_overload_force_delta"]:
        for predictor in [
            "fabric_memory_index",
            "loop_activation_index",
            "Z_cold_N_mean",
            "survival_N_mean",
            "force_h1_birth_force_share_hot_minus_cold",
            "force_share_top5_edges_hot_minus_cold",
        ]:
            rows.append(corr_row("six_regime", predictor, target, six[predictor], six[target]))
    corr = pd.DataFrame(rows)
    return six, corr


def load_long_cycle_projection() -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = pd.read_csv(SRC / "nphys_long_cycle_true_force_geometry_join.csv")
    wide = joined.pivot_table(
        index=["run", "tag", "regime_id", "cycle"],
        columns="phase",
        values=[
            "force_p99",
            "force_share_top5_edges",
            "force_h1_birth_force_share",
            "Z_geom",
            "orientation_entropy",
            "cycle_birth_positive_fraction",
            "force_proxy_gini",
            "force_proxy_q99_q50",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{name}_{phase}" for name, phase in wide.columns]
    wide = wide.reset_index()
    for col in [
        "force_p99",
        "force_share_top5_edges",
        "force_h1_birth_force_share",
        "Z_geom",
        "orientation_entropy",
        "cycle_birth_positive_fraction",
        "force_proxy_gini",
        "force_proxy_q99_q50",
    ]:
        wide[f"{col}_delta"] = wide[f"{col}_hot"] - wide[f"{col}_cold"]
    wide["fabric_reservoir"] = (
        zscore(wide["Z_geom_cold"])
        + zscore(wide["force_h1_birth_force_share_cold"])
        - zscore(wide["orientation_entropy_cold"])
    )
    wide["thermal_rewrite"] = (
        zscore(wide["Z_geom_delta"].abs())
        + zscore(wide["cycle_birth_positive_fraction_delta"].abs())
    )
    wide["loop_activation"] = wide["force_h1_birth_force_share_delta"]
    wide["overload_delta"] = wide["force_p99_delta"]

    centered = wide.copy()
    numeric_cols = [c for c in centered.columns if centered[c].dtype.kind in "fc"]
    for col in numeric_cols:
        centered[col] = centered[col] - centered.groupby("regime_id")[col].transform("mean")

    rows = []
    for df, scope in [(wide, "long_cycle_pooled"), (centered, "long_cycle_within_regime")]:
        for target in ["force_p99_cold", "force_p99_hot", "overload_delta", "loop_activation"]:
            for predictor in [
                "fabric_reservoir",
                "thermal_rewrite",
                "Z_geom_cold",
                "force_h1_birth_force_share_cold",
                "cycle_birth_positive_fraction_delta",
                "force_h1_birth_force_share_delta",
                "force_share_top5_edges_delta",
            ]:
                rows.append(corr_row(scope, predictor, target, df[predictor], df[target]))
    corr = pd.DataFrame(rows)
    return wide, corr


def build_figure(six: pd.DataFrame, wide: pd.DataFrame, six_corr: pd.DataFrame, long_corr: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color="#D6DADF", lw=0.7)
    ax.axvline(0, color="#D6DADF", lw=0.7)
    sizes = 28 + 150 * (six["cold_load"] - six["cold_load"].min()) / (six["cold_load"].max() - six["cold_load"].min())
    sc = ax.scatter(
        six["fabric_memory_index"],
        six["loop_activation_index"],
        s=sizes,
        c=six["hot_load"],
        cmap="OrRd",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for _, row in six.iterrows():
        ax.text(row["fabric_memory_index"] + 0.08, row["loop_activation_index"], row["regime_id"], fontsize=5.9, va="center")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("hot load proxy", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8)
    ax.set_xlabel("fabric-memory coordinate")
    ax.set_ylabel("loop-activation coordinate")
    ax.set_title("one state space, two projections", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(six["fabric_memory_index"], six["cold_load"], s=38, color=COLD, edgecolor="white", linewidth=0.5)
    rho = six_corr.query("predictor == 'fabric_memory_index' and target == 'cold_load'")["spearman_rho"].iloc[0]
    ax.text(0.05, 0.92, rf"$\rho={rho:.2f}$", transform=ax.transAxes, color=COLD, fontsize=6.4, va="top")
    ax.set_xlabel("fabric-memory coordinate")
    ax.set_ylabel("cold load proxy")
    ax.set_title("cold readout", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(six["loop_activation_index"], six["hot_overload_force_delta"], s=38, color=HOT, edgecolor="white", linewidth=0.5)
    rho = six_corr.query("predictor == 'loop_activation_index' and target == 'hot_overload_force_delta'")["spearman_rho"].iloc[0]
    ax.text(0.05, 0.92, rf"$\rho={rho:.2f}$", transform=ax.transAxes, color=HOT, fontsize=6.4, va="top")
    ax.set_xlabel("loop-activation coordinate")
    ax.set_ylabel(r"$\Delta f_{99}$")
    ax.set_title("hot overload readout", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    for rid, group in wide.groupby("regime_id"):
        ax.plot(group["cycle"], group["fabric_reservoir"], marker="o", ms=2.1, lw=1.0, label=rid)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel("fabric reservoir")
    ax.set_title("state reservoir evolves by route", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.5, ncol=3)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    for rid, group in wide.groupby("regime_id"):
        ax.plot(group["cycle"], group["thermal_rewrite"], marker="o", ms=2.1, lw=1.0, label=rid)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel("thermal rewrite")
    ax.set_title("heating perturbation", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    rows = [
        ("rewrite -> loops", "thermal_rewrite", "loop_activation", HOT),
        ("loops -> overload", "force_h1_birth_force_share_delta", "overload_delta", HOT),
        ("top-5% -> overload", "force_share_top5_edges_delta", "overload_delta", NEUTRAL),
        ("cold loops -> cold f99", "force_h1_birth_force_share_cold", "force_p99_cold", COLD),
    ]
    values = []
    labels = []
    colors = []
    for label, pred, target, color in rows:
        match = long_corr.query(
            "scope == 'long_cycle_within_regime' and predictor == @pred and target == @target"
        )
        values.append(match["spearman_rho"].iloc[0])
        labels.append(label)
        colors.append(color)
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.axvline(0, color="#9AA1A9", lw=0.7)
    ax.set_yticks(np.arange(len(values)), labels)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlabel("within-route Spearman rho")
    ax.set_title("projection tests", fontsize=7.5, pad=5)
    finish(ax, "x")
    panel(ax, "f")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig10_unified_projection.{ext}", **kwargs)
    plt.close(fig)


def write_report(six_corr: pd.DataFrame, long_corr: pd.DataFrame) -> None:
    def grab(scope: str, predictor: str, target: str, table: pd.DataFrame) -> float:
        return float(
            table.query("scope == @scope and predictor == @predictor and target == @target")[
                "spearman_rho"
            ].iloc[0]
        )

    lines = [
        "# Unified cold-hot projection mechanism",
        "",
        "## Working theory",
        "",
        "Cold memory and hot overload can be treated as two phase-dependent projections of the same thermally rewritten contact-network state, rather than as two unrelated mechanisms.",
        "",
        "The state has at least two useful coordinates:",
        "",
        "- A fabric-memory reservoir, dominated by coordination, contact survival and low force heterogeneity in the cold aged state.",
        "- A loop-activation susceptibility, dominated by the force carried by cycle-closing graph edges during hot expansion.",
        "",
        "Cold readout is approximately a projection onto the fabric reservoir. Hot overload is approximately a projection of the heating perturbation onto loop activation.",
        "",
        "## Evidence from the six-regime ensemble",
        "",
        f"- Fabric-memory coordinate vs cold load: Spearman rho = {grab('six_regime', 'fabric_memory_index', 'cold_load', six_corr):.2f}.",
        f"- Loop-activation coordinate vs true-force overload delta: Spearman rho = {grab('six_regime', 'loop_activation_index', 'hot_overload_force_delta', six_corr):.2f}.",
        f"- Loop-activation coordinate vs hot wall-load proxy: Spearman rho = {grab('six_regime', 'loop_activation_index', 'hot_load', six_corr):.2f}; this is positive but small-n and should be treated as supportive, not decisive.",
        "",
        "## Evidence from the long-cycle true-force rerun",
        "",
        f"- Within-route thermal rewrite vs loop activation: Spearman rho = {grab('long_cycle_within_regime', 'thermal_rewrite', 'loop_activation', long_corr):.2f}.",
        f"- Within-route force-loop activation vs overload delta: Spearman rho = {grab('long_cycle_within_regime', 'force_h1_birth_force_share_delta', 'overload_delta', long_corr):.2f}.",
        f"- Within-route top-5% force-share delta vs overload delta: Spearman rho = {grab('long_cycle_within_regime', 'force_share_top5_edges_delta', 'overload_delta', long_corr):.2f}.",
        f"- Within-route cold force-loop share vs cold force p99: Spearman rho = {grab('long_cycle_within_regime', 'force_h1_birth_force_share_cold', 'force_p99_cold', long_corr):.2f}.",
        "",
        "## Mechanistic interpretation",
        "",
        "The same contact network has a stable sector and an excitable sector. During cooling, thermal expansion is removed and the load is read mainly through the stable fabric reservoir. During heating, the imposed expansion perturbs the same network; if the perturbation closes high-force redundant graph loops, the hot state produces overload. Thus cold and hot readouts are different projections of one history-dependent network, not separate phenomena.",
        "",
        "## Conservative boundary",
        "",
        "The six-regime projection uses only six regime means and should be used as an organizing diagnostic. The long-cycle test is stronger for the hot overload pathway because it contains 90 paired hot-cold states. The unified theory should be phrased as a reduced state-space interpretation, not as a universal constitutive law.",
        "",
    ]
    (ROOT / "nature_physics_unified_projection_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    six, six_corr = load_six_regime_projection()
    wide, long_corr = load_long_cycle_projection()
    six.to_csv(SRC / "nphys_unified_projection_six_regime.csv", index=False)
    wide.to_csv(SRC / "nphys_unified_projection_long_cycle.csv", index=False)
    pd.concat([six_corr, long_corr], ignore_index=True).to_csv(
        SRC / "nphys_unified_projection_correlations.csv", index=False
    )
    build_figure(six, wide, six_corr, long_corr)
    write_report(six_corr, long_corr)
    print("six-regime projection")
    print(six[["tag", "fabric_memory_index", "loop_activation_index", "cold_load", "hot_load", "hot_overload_force_delta"]])
    print("selected correlations")
    print(pd.concat([six_corr, long_corr], ignore_index=True).query("abs(spearman_rho) > 0.75"))


if __name__ == "__main__":
    main()
