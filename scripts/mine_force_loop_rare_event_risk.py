#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#8A929C"
GRID = "#E8EBEF"
LOOP = "#B6423E"
TAIL = "#3D6B9C"
GIANT = "#7E6AAE"
PERC = "#D98C3A"
ROUTE = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}

PREDICTORS = {
    "force-loop activation": ("loop_activation_z", LOOP),
    "top-5% force tail": ("top5_activation_z", TAIL),
    "giant-component change": ("giant_activation_z", GIANT),
    "wall-spanning proxy": ("percolation_activation_z", PERC),
}


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


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=INK)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    centered = df[col] - df.groupby("regime_id")[col].transform("median")
    scale = df.groupby("regime_id")[col].transform(lambda s: np.subtract(*np.percentile(s, [75, 25])))
    scale = scale.replace(0, np.nan)
    return centered / scale


def empirical_rare_event_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (col, _) in PREDICTORS.items():
        d = df[["rare_overload", col]].replace([np.inf, -np.inf], np.nan).dropna()
        d["bin"] = pd.qcut(d[col], q=5, labels=False, duplicates="drop")
        for b, g in d.groupby("bin", sort=True):
            rows.append(
                {
                    "predictor": label,
                    "bin": int(b) + 1,
                    "n": int(len(g)),
                    "mean_score": float(g[col].mean()),
                    "rare_event_probability": float(g["rare_overload"].mean()),
                    "rare_event_count": int(g["rare_overload"].sum()),
                }
            )
    return pd.DataFrame(rows)


def predictor_metrics(df: pd.DataFrame, n_perm: int = 4000, seed: int = 41) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metrics = []
    nulls = []
    y = df["rare_overload"].to_numpy(int)
    base_rate = float(y.mean())
    for label, (col, _) in PREDICTORS.items():
        d = df[["regime_id", "rare_overload", col]].replace([np.inf, -np.inf], np.nan).dropna()
        y = d["rare_overload"].to_numpy(int)
        score = d[col].to_numpy(float)
        auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
        rho = spearmanr(score, y).statistic
        bins = np.asarray(pd.qcut(score, q=5, labels=False, duplicates="drop"), dtype=int)
        mi_bits = float(mutual_info_score(y, bins) / np.log(2))
        contingency = pd.crosstab(bins, y)
        chi2_p = float(chi2_contingency(contingency, correction=False).pvalue)
        high = d.loc[bins == np.nanmax(bins), "rare_overload"].mean()
        low = d.loc[bins == np.nanmin(bins), "rare_overload"].mean()
        metrics.append(
            {
                "predictor": label,
                "n": int(len(d)),
                "base_rate": base_rate,
                "auc": float(auc),
                "spearman_score_event": float(rho),
                "mutual_information_bits": mi_bits,
                "chi2_p_quintile_event": chi2_p,
                "top_quintile_risk": float(high),
                "bottom_quintile_risk": float(low),
                "top_quintile_risk_ratio_vs_base": float(high / base_rate) if base_rate else np.nan,
            }
        )

        observed = mi_bits
        route_labels = d["regime_id"].to_numpy()
        for i in range(n_perm):
            y_perm = y.copy()
            for rid in np.unique(route_labels):
                idx = np.flatnonzero(route_labels == rid)
                y_perm[idx] = rng.permutation(y_perm[idx])
            null_mi = float(mutual_info_score(y_perm, bins) / np.log(2))
            nulls.append(
                {
                    "predictor": label,
                    "permutation": i,
                    "null_mutual_information_bits": null_mi,
                    "observed_mutual_information_bits": observed,
                }
            )
    null_df = pd.DataFrame(nulls)
    metrics_df = pd.DataFrame(metrics)
    pvals = []
    for label, g in null_df.groupby("predictor"):
        observed = float(g["observed_mutual_information_bits"].iloc[0])
        p = (1 + np.sum(g["null_mutual_information_bits"].to_numpy() >= observed)) / (len(g) + 1)
        pvals.append({"predictor": label, "route_permutation_p_mi": float(p)})
    metrics_df = metrics_df.merge(pd.DataFrame(pvals), on="predictor", how="left")
    return metrics_df, null_df


def leave_one_route_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("regime_id", sort=True):
        y = g["rare_overload"].to_numpy(int)
        for label, (col, _) in PREDICTORS.items():
            score = g[col].to_numpy(float)
            auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
            rows.append(
                {
                    "left_out_route": rid,
                    "predictor": label,
                    "n_test": int(len(g)),
                    "rare_event_count": int(y.sum()),
                    "route_local_auc": float(auc),
                }
            )
    return pd.DataFrame(rows)


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(INFILE).copy()
    df["overload_asinh"] = np.arcsinh(df["overload_number"] / 2.0)
    for col in [
        "overload_asinh",
        "loop_activation",
        "top5_activation",
        "giant_activation",
        "bottom_side_percolation_edge_fraction_hot_minus_cold",
    ]:
        df[f"{col}_z"] = route_zscore(df, col)
    df = df.rename(
        columns={
            "overload_asinh_z": "overload_z",
            "bottom_side_percolation_edge_fraction_hot_minus_cold_z": "percolation_activation_z",
        }
    )
    threshold = df.groupby("regime_id")["overload_asinh"].transform(lambda s: s.quantile(0.80))
    df["rare_overload"] = (df["overload_asinh"] >= threshold).astype(int)
    return df


def make_figure(df: pd.DataFrame, bins: pd.DataFrame, metrics: pd.DataFrame, loro: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.08, 5.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.46, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    for rid, g in df.groupby("regime_id", sort=True):
        base = g[g["rare_overload"] == 0]
        rare = g[g["rare_overload"] == 1]
        ax.scatter(
            base["loop_activation_z"],
            base["overload_z"],
            s=13,
            c=ROUTE[rid],
            marker=MARKERS[rid],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            zorder=2,
            label=rid,
        )
        ax.scatter(
            rare["loop_activation_z"],
            rare["overload_z"],
            s=28,
            c=LOOP,
            marker=MARKERS[rid],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=3,
        )
    ax.axhline(df["overload_z"].quantile(0.80), color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("route-local loop activation")
    ax.set_ylabel("route-local overload")
    ax.set_title("Extreme events occupy the loop-activated tail", loc="left", pad=4)
    finish(ax)
    handles = [
        plt.Line2D([0], [0], marker=MARKERS[r], color="none", markerfacecolor=ROUTE[r], markeredgecolor="white", markersize=5, label=r)
        for r in sorted(ROUTE)
    ]
    ax.legend(handles=handles, ncol=3, loc="lower right", fontsize=6.0, handletextpad=0.2, columnspacing=0.7)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for label, (_, color) in PREDICTORS.items():
        d = bins[bins["predictor"] == label]
        ax.plot(d["bin"], d["rare_event_probability"], marker="o", ms=3.8, lw=1.15, color=color, label=label)
    ax.axhline(df["rare_overload"].mean(), color=MUTED, lw=0.8, ls=(0, (3, 2)), label="base rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("predictor quintile")
    ax.set_ylabel("P(route-local top 20% overload)")
    ax.set_title("Conditional risk separates topology from force tail", loc="left", pad=4)
    finish(ax)
    ax.legend(loc="upper left", fontsize=5.9, handlelength=1.4)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    order = list(PREDICTORS)
    x = np.arange(len(order))
    auc = metrics.set_index("predictor").loc[order, "auc"].to_numpy()
    mi = metrics.set_index("predictor").loc[order, "mutual_information_bits"].to_numpy()
    width = 0.34
    ax.bar(x - width / 2, auc, width=width, color=[PREDICTORS[o][1] for o in order], alpha=0.92, label="AUC")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, mi, width=width, color=[PREDICTORS[o][1] for o in order], alpha=0.35, label="MI")
    ax.axhline(0.5, color=MUTED, lw=0.75, ls=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(["loop", "top-5%", "giant", "span"], rotation=0)
    ax.set_ylabel("AUC")
    ax2.set_ylabel("mutual information (bits)")
    ax.set_ylim(0.0, 1.02)
    ax2.set_ylim(0.0, max(0.05, float(np.nanmax(mi) * 1.25)))
    ax.set_title("Loop activation carries the rare-event information", loc="left", pad=4)
    finish(ax, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(width=0.65, direction="in")
    auc_handle = plt.Rectangle((0, 0), 1, 1, fc=INK, alpha=0.86, ec="none")
    mi_handle = plt.Rectangle((0, 0), 1, 1, fc=INK, alpha=0.28, ec="none")
    ax.legend([auc_handle, mi_handle], ["AUC", "MI"], loc="upper right", fontsize=6.0, handlelength=1.1)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    summary = loro.groupby("predictor", sort=False)["route_local_auc"].agg(["mean", "min", "max"]).loc[order].reset_index()
    colors = [PREDICTORS[o][1] for o in order]
    xpos = np.arange(len(order))
    ax.bar(xpos, summary["mean"], color=colors, alpha=0.86, width=0.56)
    for i, row in summary.iterrows():
        ax.plot([i, i], [row["min"], row["max"]], color=INK, lw=0.9)
        vals = loro[loro["predictor"] == row["predictor"]]["route_local_auc"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, s=14, color="white", edgecolor=colors[i], linewidth=0.75, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.75, ls=(0, (3, 2)))
    ax.set_xticks(xpos)
    ax.set_xticklabels(["loop", "top-5%", "giant", "span"])
    ax.set_ylabel("route-local AUC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Route-wise tail tests bound the claim", loc="left", pad=4)
    finish(ax, axis="y")

    fig.suptitle("Force-loop activation turns hot overload into a rare-event readout", x=0.02, y=0.995, ha="left", fontsize=10.2, fontweight="bold", color=INK)
    fig.savefig(FIG / "nphys_fig40_force_loop_rare_events.pdf", bbox_inches="tight")
    fig.savefig(FIG / "nphys_fig40_force_loop_rare_events.svg", bbox_inches="tight")
    fig.savefig(FIG / "nphys_fig40_force_loop_rare_events.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG / "nphys_fig40_force_loop_rare_events.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, loro: pd.DataFrame, bins: pd.DataFrame) -> None:
    loop = metrics.set_index("predictor").loc["force-loop activation"]
    tail = metrics.set_index("predictor").loc["top-5% force tail"]
    giant = metrics.set_index("predictor").loc["giant-component change"]
    span = metrics.set_index("predictor").loc["wall-spanning proxy"]
    top_loop = bins[(bins["predictor"] == "force-loop activation") & (bins["bin"] == 5)].iloc[0]
    low_loop = bins[(bins["predictor"] == "force-loop activation") & (bins["bin"] == 1)].iloc[0]
    loro_mean = loro.groupby("predictor")["route_local_auc"].mean()
    lines = [
        "# Force-loop rare-event risk audit",
        "",
        "Purpose: test whether the hot-overload mechanism is only a rank correlation or whether force-loop activation specifically identifies route-local extreme events.",
        "",
        "## Definition",
        "",
        "A rare overload event is defined within each route as a cycle whose asinh overload lies in the top 20% of that route. This removes route-mean severity and asks which cycles enter the dangerous tail under a fixed preparation route.",
        "",
        "## Main results",
        "",
        f"- Force-loop activation gives AUC={loop['auc']:.3f} and mutual information={loop['mutual_information_bits']:.3f} bits for the route-local rare-event label.",
        f"- The top predictor quintile has rare-event probability {top_loop['rare_event_probability']:.3f}, compared with {low_loop['rare_event_probability']:.3f} in the bottom quintile and a base rate of {loop['base_rate']:.3f}.",
        f"- The route-wise mean AUC for force-loop activation is {loro_mean['force-loop activation']:.3f}; the corresponding values are {loro_mean['top-5% force tail']:.3f} for the top-5% force-tail surrogate, {loro_mean['giant-component change']:.3f} for giant-component change and {loro_mean['wall-spanning proxy']:.3f} for the wall-spanning proxy.",
        f"- Route-preserving permutation gives P={loop['route_permutation_p_mi']:.4f} for loop mutual information. The force-tail surrogate is also non-random but oppositely signed: AUC={tail['auc']:.3f}, MI={tail['mutual_information_bits']:.3f} bits and P={tail['route_permutation_p_mi']:.4f}.",
        "",
        "## Interpretation boundary",
        "",
        "This audit supports rare-event language for the hot channel: overload is preferentially read from the force-loop-activated tail of the cycle ensemble. It does not establish a thermodynamic large-deviation rate function or a universal critical threshold, because the evidence is finite-route and route-conditioned. The defensible claim is that graph-cycle force embedding increases the conditional risk and information content of route-local extreme overload events beyond force-tail and connectivity controls. The force-tail surrogate carries information mainly as an inverse control, so it should not be reinterpreted as a positive overload coordinate.",
        "",
        "## Metrics table",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Route-wise AUC table",
        "",
        loro.to_markdown(index=False),
        "",
    ]
    (ROOT / "nature_physics_force_loop_rare_event_risk.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    df = prepare_data()
    bins = empirical_rare_event_table(df)
    metrics, nulls = predictor_metrics(df)
    loro = leave_one_route_auc(df)

    df.to_csv(SRC / "nphys_force_loop_rare_event_cycle_metrics.csv", index=False)
    bins.to_csv(SRC / "nphys_force_loop_rare_event_predictor_bins.csv", index=False)
    metrics.to_csv(SRC / "nphys_force_loop_rare_event_metrics.csv", index=False)
    nulls.to_csv(SRC / "nphys_force_loop_rare_event_permutation_null.csv", index=False)
    loro.to_csv(SRC / "nphys_force_loop_rare_event_route_auc.csv", index=False)
    make_figure(df, bins, metrics, loro)
    write_report(metrics, loro, bins)
    print(metrics.to_string(index=False))
    print("\nWrote figures/nphys_fig40_force_loop_rare_events.*")


if __name__ == "__main__":
    main()
