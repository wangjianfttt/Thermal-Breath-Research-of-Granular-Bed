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

ROUTES = ["R1", "R3", "R5", "R6", "R6c"]
COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
MUTED = "#7F8790"
GRID = "#E7EAEE"
LOOP = "#B6423E"
TAIL = "#3D6B9C"


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=INK)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    obs = abs(rho)
    null = []
    for perm in permutations(np.asarray(y, dtype=float)):
        null.append(abs(float(spearmanr(x, np.asarray(perm)).statistic)))
    p = float(np.mean(np.asarray(null) >= obs - 1e-12))
    return rho, p


def minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return series * 0
    return (series - lo) / (hi - lo)


def load_atlas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gain = pd.read_csv(SRC / "nphys_two_scale_variance_partition_route_gain.csv")
    kernels = pd.read_csv(SRC / "nphys_two_scale_response_collapse_route_kernels.csv")
    maps = pd.read_csv(SRC / "nphys_return_map_normal_form_route_metrics.csv")
    storage = pd.read_csv(SRC / "nphys_lyapunov_storage_route_metrics.csv")
    rare = pd.read_csv(SRC / "nphys_force_loop_rare_event_route_auc.csv")
    precursor = pd.read_csv(SRC / "nphys_force_loop_rare_event_precursor_route_auc.csv")
    memory = pd.read_csv(SRC / "nphys_five_route_breath_memory_spectrum_route_summary.csv")

    rare_loop = rare[rare["predictor"] == "force-loop activation"][
        ["left_out_route", "route_local_auc"]
    ].rename(columns={"left_out_route": "regime_id", "route_local_auc": "rare_loop_auc"})
    rare_tail = rare[rare["predictor"] == "top-5% force tail"][
        ["left_out_route", "route_local_auc"]
    ].rename(columns={"left_out_route": "regime_id", "route_local_auc": "rare_tail_auc"})
    prec_loop = precursor[precursor["predictor"] == "lagged loop"][
        ["regime_id", "route_local_auc"]
    ].rename(columns={"route_local_auc": "lag2_loop_auc"})
    prec_tail = precursor[precursor["predictor"] == "lagged top-5% tail"][
        ["regime_id", "route_local_auc"]
    ].rename(columns={"route_local_auc": "lag2_tail_auc"})

    atlas = (
        gain[
            [
                "regime_id",
                "S",
                "loop_gain",
                "intercept",
                "spearman_loop_overload",
                "mean_overload",
                "mean_loop_activation",
            ]
        ]
        .merge(kernels[["regime_id", "slope_ci_low", "slope_ci_high"]], on="regime_id", how="left")
        .merge(
            maps[
                [
                    "regime_id",
                    "spectral_radius",
                    "stability_margin",
                    "one_step_gain",
                    "peak_normalized_gain",
                    "nonnormality",
                    "alternating_mode_strength",
                    "two_cycle_persistence",
                ]
            ],
            on="regime_id",
            how="left",
        )
        .merge(storage[["regime_id", "storage_anisotropy", "hot_axis_storage"]], on="regime_id", how="left")
        .merge(rare_loop, on="regime_id", how="left")
        .merge(rare_tail, on="regime_id", how="left")
        .merge(prec_loop, on="regime_id", how="left")
        .merge(prec_tail, on="regime_id", how="left")
        .merge(memory[["regime_id", "lag0_loop_rho", "lag2_loop_rho"]], on="regime_id", how="left")
    )
    atlas["rare_loop_minus_tail_auc"] = atlas["rare_loop_auc"] - atlas["rare_tail_auc"]
    atlas["lag2_loop_minus_tail_auc"] = atlas["lag2_loop_auc"] - atlas["lag2_tail_auc"]
    atlas["route_label"] = pd.Categorical(atlas["regime_id"], ROUTES, ordered=True)
    atlas = atlas.sort_values("route_label").drop(columns=["route_label"]).reset_index(drop=True)

    metric_cols = [
        "S",
        "loop_gain",
        "mean_loop_activation",
        "mean_overload",
        "spectral_radius",
        "peak_normalized_gain",
        "storage_anisotropy",
        "rare_loop_auc",
        "lag2_loop_auc",
    ]
    heat = atlas[["regime_id", *metric_cols]].copy()
    for col in metric_cols:
        heat[f"{col}_scaled"] = minmax(heat[col])

    pairs = [
        ("S", "loop_gain", "slow severity vs loop gain"),
        ("S", "mean_overload", "slow severity vs mean overload"),
        ("loop_gain", "mean_overload", "loop gain vs mean overload"),
        ("spectral_radius", "peak_normalized_gain", "stability vs transient gain"),
        ("rare_loop_minus_tail_auc", "mean_overload", "loop-tail rare-event advantage vs overload"),
        ("lag2_loop_minus_tail_auc", "mean_overload", "lagged loop-tail advantage vs overload"),
    ]
    corr_rows = []
    for xcol, ycol, label in pairs:
        rho, p = exact_spearman(atlas[xcol].to_numpy(float), atlas[ycol].to_numpy(float))
        corr_rows.append({"relationship": label, "x": xcol, "y": ycol, "n": len(atlas), "spearman": rho, "exact_p": p})
    corr = pd.DataFrame(corr_rows)
    return atlas, heat, corr


def build_figure(atlas: pd.DataFrame, heat: pd.DataFrame, corr: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.25, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.38, 1.0, 1.0], height_ratios=[1.05, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", x=-0.08)
    labels = [
        ("S", "slow S"),
        ("loop_gain", "loop gain"),
        ("mean_loop_activation", "mean loop"),
        ("mean_overload", "mean load"),
        ("spectral_radius", "map radius"),
        ("peak_normalized_gain", "peak gain"),
        ("storage_anisotropy", "storage anis."),
        ("rare_loop_auc", "rare AUC"),
        ("lag2_loop_auc", "lag-2 AUC"),
    ]
    mat = heat[[f"{col}_scaled" for col, _ in labels]].to_numpy(float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "atlas",
        ["#F7F8FA", "#D9E7EE", "#8EB6C9", "#3D6B9C", "#252A31"],
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels(heat["regime_id"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([lab for _, lab in labels], rotation=38, ha="right", fontsize=5.8, rotation_mode="anchor")
    ax.tick_params(length=0)
    for y in np.arange(0.5, len(heat) - 0.5, 1):
        ax.axhline(y, color="white", lw=0.8)
    for x in np.arange(0.5, len(labels) - 0.5, 1):
        ax.axvline(x, color="white", lw=0.65)
    for i, row in atlas.iterrows():
        for j, (col, _) in enumerate(labels):
            value = row[col]
            text = f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
            color = "white" if mat[i, j] > 0.62 else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=4.8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("within-column scaled value", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    ax.set_title("route susceptibility atlas", loc="left", pad=5)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    order = atlas.sort_values("S")
    yerr = np.vstack([order["loop_gain"] - order["slope_ci_low"], order["slope_ci_high"] - order["loop_gain"]])
    ax.errorbar(order["S"], order["loop_gain"], yerr=yerr, fmt="none", ecolor="#C7CDD4", elinewidth=0.9, capsize=2.0, zorder=1)
    for _, row in order.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["S"], row["loop_gain"], s=42, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["S"] + 0.018, row["loop_gain"], rid, color=COLORS[rid], fontsize=6.1, va="center")
    rel = corr[corr["relationship"] == "slow severity vs loop gain"].iloc[0]
    ax.text(0.04, 0.95, rf"$\rho={rel.spearman:.2f}$" + f"\nexact P={rel.exact_p:.3f}", transform=ax.transAxes, va="top", color=MUTED, fontsize=6.0)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("slow route severity, S")
    ax.set_ylabel("loop-to-overload gain")
    ax.set_title("slow state sets loop gain", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    for _, row in atlas.iterrows():
        rid = row["regime_id"]
        size = 38 + 46 * max(row["mean_overload"], 0)
        ax.scatter(row["spectral_radius"], row["peak_normalized_gain"], s=size, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.5, alpha=0.95)
        ax.text(row["spectral_radius"] + 0.01, row["peak_normalized_gain"], rid, color=COLORS[rid], fontsize=6.0, va="center")
    ax.axvline(1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axhline(1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlim(0.52, 1.02)
    ax.set_xlabel(r"spectral radius, $\rho(A)$")
    ax.set_ylabel("peak normalised gain")
    ax.set_title("stable maps can still amplify", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    y = np.arange(len(atlas))
    for i, row in atlas.iterrows():
        rid = row["regime_id"]
        ax.plot([row["rare_tail_auc"], row["rare_loop_auc"]], [i, i], color="#C9CED6", lw=1.2, zorder=1)
        ax.scatter(row["rare_tail_auc"], i, s=30, color=TAIL, marker="o", edgecolor="white", lw=0.4, zorder=2)
        ax.scatter(row["rare_loop_auc"], i, s=34, color=LOOP, marker="s", edgecolor="white", lw=0.4, zorder=3)
        ax.text(1.015, i, rid, color=COLORS[rid], fontsize=5.9, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(-0.02, 1.08)
    ax.set_xlabel("route-local rare-event AUC")
    ax.set_title("loop sector beats force tail", loc="left", pad=4)
    ax.text(0.05, 0.08, "blue: top-5% tail\nred: force-loop", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    x = np.arange(len(atlas))
    ax.bar(x - 0.17, atlas["rare_loop_minus_tail_auc"], width=0.30, color=LOOP, alpha=0.86, label="same cycle")
    ax.bar(x + 0.17, atlas["lag2_loop_minus_tail_auc"], width=0.30, color="#D98C3A", alpha=0.86, label="lag 2")
    ax.set_xticks(x)
    ax.set_xticklabels(atlas["regime_id"])
    ax.set_ylabel(r"$\Delta$AUC, loop - tail")
    ax.set_title("loop advantage persists with memory", loc="left", pad=4)
    ax.legend(fontsize=5.8, handlelength=1.1, loc="upper left")
    finish(ax, axis="y")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig51_route_susceptibility_atlas.{ext}", **kwargs)
    plt.close(fig)


def write_report(atlas: pd.DataFrame, corr: pd.DataFrame) -> None:
    out = ROOT / "nature_physics_route_susceptibility_atlas.md"
    c = corr.set_index("relationship")
    lines = [
        "# Route susceptibility atlas",
        "",
        "## Question",
        "",
        "Can the scattered mechanism audits be read as one route-level physics picture: a slow trained route state sets the gain, a fast force-loop coordinate supplies the drive and a stable return map still permits transient amplification?",
        "",
        "## Main result",
        "",
        f"Across the five true-force routes, slow route severity orders the fitted loop-to-overload gain with Spearman rho = {c.loc['slow severity vs loop gain','spearman']:.3f} (exact P = {c.loc['slow severity vs loop gain','exact_p']:.3f}). The same severity also orders mean overload with rho = {c.loc['slow severity vs mean overload','spearman']:.3f}.",
        "",
        f"The return maps remain stable, with spectral radii from {atlas['spectral_radius'].min():.3f} to {atlas['spectral_radius'].max():.3f}, but their peak normalised gains range from {atlas['peak_normalized_gain'].min():.3f} to {atlas['peak_normalized_gain'].max():.3f}. This is the route-level signature of stable-but-excitable breathing.",
        "",
        f"The force-loop sector beats the top-5 percent force-tail surrogate in rare-event readout for four of five routes and in lag-2 precursor readout for all five routes. The strongest susceptible route, {atlas.sort_values('mean_overload').iloc[-1]['regime_id']}, has mean asinh overload {atlas['mean_overload'].max():.3f}, loop gain {atlas.sort_values('mean_overload').iloc[-1]['loop_gain']:.3f} and lag-2 loop AUC {atlas.sort_values('mean_overload').iloc[-1]['lag2_loop_auc']:.3f}.",
        "",
        "## Interpretation boundary",
        "",
        "Allowed: the atlas is a synthesis figure showing that route severity, loop gain, return-map stability, transient amplification and rare-event risk are consistent with a slow-susceptibility times fast-loop-drive mechanism.",
        "",
        "Not allowed: this is not an independent causal proof, not a universal material law and not a new main-figure claim. It should be used as reserve evidence or as a compact reviewer-facing map of the mechanism.",
        "",
        "## Route atlas",
        "",
        atlas.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Route-level exact Spearman checks",
        "",
        corr.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig51_route_susceptibility_atlas.*`",
        "- `source_data/nphys_route_susceptibility_atlas.csv`",
        "- `source_data/nphys_route_susceptibility_atlas_heatmap.csv`",
        "- `source_data/nphys_route_susceptibility_atlas_correlations.csv`",
        "",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    atlas, heat, corr = load_atlas()
    atlas.to_csv(SRC / "nphys_route_susceptibility_atlas.csv", index=False)
    heat.to_csv(SRC / "nphys_route_susceptibility_atlas_heatmap.csv", index=False)
    corr.to_csv(SRC / "nphys_route_susceptibility_atlas_correlations.csv", index=False)
    build_figure(atlas, heat, corr)
    write_report(atlas, corr)
    print(atlas[["regime_id", "S", "loop_gain", "mean_overload", "spectral_radius", "peak_normalized_gain", "rare_loop_auc", "lag2_loop_auc"]].to_string(index=False))
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
