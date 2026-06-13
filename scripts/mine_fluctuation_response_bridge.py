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

MATRIX_FILE = SRC / "nphys_route_phase_space_27case.csv"
PREP_FILE = SRC / "nphys_preparation_susceptibility_regime_summary.csv"
KERNEL_FILE = SRC / "nphys_loop_susceptibility_kernel_route_kernels.csv"

REGIME_ORDER = [
    "a050_mu010_g002",
    "a050_mu060_g002",
    "a100_mu030_g002",
    "a100_mu060_g000",
    "a150_mu010_g002",
    "a150_mu060_g020",
]
REGIME_ID = {tag: f"R{i + 1}" for i, tag in enumerate(REGIME_ORDER)}
COLORS = {
    "R1": "#3D6B9C",
    "R2": "#2F7F6F",
    "R3": "#D98C3A",
    "R4": "#7E6AAE",
    "R5": "#6BAFB0",
    "R6": "#B6423E",
}
INK = "#252A31"
GRID = "#E7EAEE"
MUTED = "#8B929A"


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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = float(spearmanr(x, y).statistic)
    vals = []
    for perm in permutations(y):
        vals.append(abs(float(spearmanr(x, np.asarray(perm, dtype=float)).statistic)))
    p = float(np.mean(np.asarray(vals) >= abs(rho) - 1e-12))
    return rho, p


def zscore(s: pd.Series) -> pd.Series:
    sd = float(s.std(ddof=0))
    if sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / sd


def local_derivative(table: pd.DataFrame, row: pd.Series, axis: str, target: str) -> float:
    fixed = {
        "alpha_mult": row["alpha_mult"],
        "friction": row["friction"],
        "lid_gap_radii": row["lid_gap_radii"],
    }
    mask = np.ones(len(table), dtype=bool)
    for key, val in fixed.items():
        if key != axis:
            mask &= np.isclose(table[key].to_numpy(float), float(val), atol=1e-12)
    line = table.loc[mask, [axis, target]].dropna().sort_values(axis)
    if len(line) < 2:
        return np.nan
    x = line[axis].to_numpy(float)
    y = line[target].to_numpy(float)
    x0 = float(row[axis])
    if np.any(np.isclose(x, x0, atol=1e-12)):
        idx = int(np.where(np.isclose(x, x0, atol=1e-12))[0][0])
        if 0 < idx < len(x) - 1:
            return float((y[idx + 1] - y[idx - 1]) / (x[idx + 1] - x[idx - 1]))
        if idx == 0:
            return float((y[1] - y[0]) / (x[1] - x[0]))
        return float((y[-1] - y[-2]) / (x[-1] - x[-2]))
    return float(np.interp(x0, x, np.gradient(y, x)))


def build_bridge() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(MATRIX_FILE).copy()
    prep = pd.read_csv(PREP_FILE).copy()
    kernels = pd.read_csv(KERNEL_FILE).copy()

    matrix["regime_id"] = matrix["tag"].map(REGIME_ID)
    selected = matrix[matrix["tag"].isin(REGIME_ORDER)].copy()
    selected["tag"] = pd.Categorical(selected["tag"], categories=REGIME_ORDER, ordered=True)
    selected = selected.sort_values("tag").reset_index(drop=True)

    target = "hot_susceptibility_index"
    rows = []
    for _, row in selected.iterrows():
        d_alpha = local_derivative(matrix, row, "alpha_mult", target)
        d_mu = local_derivative(matrix, row, "friction", target)
        d_gap = local_derivative(matrix, row, "lid_gap_radii", target)
        gradient_norm = float(np.sqrt(np.nansum(np.asarray([d_alpha, d_mu, d_gap], dtype=float) ** 2)))
        dominant = ["alpha", "friction", "gap"][int(np.nanargmax(np.abs([d_alpha, d_mu, d_gap])))]
        rows.append(
            {
                "tag": row["tag"],
                "regime_id": row["regime_id"],
                "alpha_mult": row["alpha_mult"],
                "friction": row["friction"],
                "lid_gap_radii": row["lid_gap_radii"],
                "hot_susceptibility_index": row[target],
                "hot_load_risk": row["hot_load_risk"],
                "d_hot_susceptibility_d_alpha": d_alpha,
                "d_hot_susceptibility_d_friction": d_mu,
                "d_hot_susceptibility_d_gap": d_gap,
                "local_response_norm": gradient_norm,
                "dominant_response_axis": dominant,
            }
        )
    bridge = pd.DataFrame(rows)
    keep = [
        "tag",
        "regime_id",
        "hot_bottom_pN_mean",
        "hot_bottom_pN_cv",
        "cold_bottom_pN_cv",
        "hot_cold_cv_ratio",
        "structural_cv_mean",
        "hot_structural_cv_ratio",
        "hot_tail_p99_over_mean",
    ]
    bridge = bridge.merge(prep[keep], on=["tag", "regime_id"], how="left")
    bridge = bridge.merge(
        kernels[["regime_id", "route_severity", "susceptibility_slope", "r2_loop_to_overload"]],
        on="regime_id",
        how="left",
    )
    bridge["response_z"] = zscore(bridge["local_response_norm"])
    bridge["prep_z"] = zscore(np.log1p(bridge["hot_cold_cv_ratio"]))
    bridge["fluctuation_response_index"] = bridge["response_z"] + bridge["prep_z"]
    bridge["sector"] = pd.cut(
        bridge["fluctuation_response_index"],
        bins=[-np.inf, -0.25, 1.25, np.inf],
        labels=["quiet reservoir", "responsive", "fluctuation-response"],
    )

    tests = []
    pairs = [
        ("local_response_norm", "hot_cold_cv_ratio", "local response norm vs hot/cold CV ratio"),
        ("local_response_norm", "hot_structural_cv_ratio", "local response norm vs hot/structural CV ratio"),
        ("fluctuation_response_index", "hot_tail_p99_over_mean", "fluctuation-response index vs hot force-tail width"),
        ("fluctuation_response_index", "hot_bottom_pN_mean", "fluctuation-response index vs mean hot load"),
        ("local_response_norm", "hot_bottom_pN_mean", "local response norm vs mean hot load"),
        ("susceptibility_slope", "local_response_norm", "force-loop susceptibility slope vs local matrix response"),
    ]
    for x, y, label in pairs:
        d = bridge[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p = exact_spearman(d[x].to_numpy(float), d[y].to_numpy(float))
        tests.append({"relationship": label, "predictor": x, "target": y, "n": len(d), "spearman": rho, "exact_p": p})
    tests_df = pd.DataFrame(tests)

    bridge.to_csv(SRC / "nphys_fluctuation_response_bridge_summary.csv", index=False)
    tests_df.to_csv(SRC / "nphys_fluctuation_response_bridge_tests.csv", index=False)
    return bridge, tests_df


def draw(bridge: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 4.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1, 1])

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", x=-0.11)
    mat = bridge[
        [
            "d_hot_susceptibility_d_alpha",
            "d_hot_susceptibility_d_friction",
            "d_hot_susceptibility_d_gap",
        ]
    ].to_numpy(float)
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(bridge)), bridge["regime_id"])
    ax.set_xticks([0, 1, 2], [r"$\partial_\alpha$", r"$\partial_\mu$", r"$\partial_\chi$"])
    ax.set_title("local response tensor from 27-case matrix", loc="left", pad=4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.7, color="white" if abs(mat[i, j]) > 0.55 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.58, pad=0.02)
    cb.set_label(r"local derivative of hot susceptibility")
    cb.ax.tick_params(labelsize=5.8, width=0.5)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for _, row in bridge.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["local_response_norm"], row["hot_cold_cv_ratio"], s=50, color=COLORS[rid], edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["local_response_norm"] * 1.02, row["hot_cold_cv_ratio"], rid, color=COLORS[rid], fontsize=6.2, va="center")
    row = tests.query("relationship == 'local response norm vs hot/cold CV ratio'").iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman:.2f}$" + f"\nexact P={row.exact_p:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel(r"local response norm, $|\nabla_{\pi} S_h|$")
    ax.set_ylabel("hot/cold CV ratio")
    ax.set_title("control response is not preparation scatter", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    for _, row in bridge.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["local_response_norm"], row["hot_structural_cv_ratio"], s=50, color=COLORS[rid], edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["local_response_norm"] * 1.02, row["hot_structural_cv_ratio"], rid, color=COLORS[rid], fontsize=6.2, va="center")
    row = tests.query("relationship == 'local response norm vs hot/structural CV ratio'").iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman:.2f}$" + f"\nexact P={row.exact_p:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel(r"local response norm, $|\nabla_{\pi} S_h|$")
    ax.set_ylabel("hot-load CV / structural CV")
    ax.set_yscale("log")
    ax.set_title("load noise is not one response scalar", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    x = np.arange(len(bridge))
    ax.axhline(0, color="#B8BFC7", lw=0.65)
    ax.bar(x - 0.16, bridge["response_z"], width=0.32, color="#9DB4CC", label="response")
    ax.bar(x + 0.16, bridge["prep_z"], width=0.32, color="#DDA15E", label="preparation")
    ax.plot(x, bridge["fluctuation_response_index"], color=INK, marker="o", ms=3, lw=0.9, label="sum")
    ax.set_xticks(x, bridge["regime_id"])
    ax.set_ylabel("standardized score")
    ax.set_title("fluctuation-response sector score", loc="left", pad=4)
    ax.legend(fontsize=5.8, loc="upper left", ncol=1)
    finish(ax, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    plotted = bridge.dropna(subset=["susceptibility_slope"])
    for _, row in plotted.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["local_response_norm"], row["susceptibility_slope"], s=50, color=COLORS[rid], edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["local_response_norm"] * 1.02, row["susceptibility_slope"], rid, color=COLORS[rid], fontsize=6.2, va="center")
    row = tests.query("relationship == 'force-loop susceptibility slope vs local matrix response'").iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman:.2f}$" + f"\nexact P={row.exact_p:.3f}", transform=ax.transAxes, va="top")
    ax.axhline(0, color="#B8BFC7", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel(r"local response norm, $|\nabla_{\pi} S_h|$")
    ax.set_ylabel(r"force-loop gain, $G$")
    ax.set_title("matrix response aligns only with loop gain", loc="left", pad=4)
    finish(ax)

    out = FIG / "nphys_fig34_fluctuation_response_bridge"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(bridge: pd.DataFrame, tests: pd.DataFrame) -> None:
    top = bridge.sort_values("fluctuation_response_index", ascending=False).iloc[0]
    r_cv = tests.query("relationship == 'local response norm vs hot/cold CV ratio'").iloc[0]
    r_struct = tests.query("relationship == 'local response norm vs hot/structural CV ratio'").iloc[0]
    r_loop = tests.query("relationship == 'force-loop susceptibility slope vs local matrix response'").iloc[0]
    lines = [
        "# Fluctuation-response bridge audit",
        "",
        "Date: 2026-06-12",
        "",
        "## Question",
        "",
        "This audit asks whether hot-state danger sits in a fluctuation-response sector: routes that are locally sensitive to control parameters in the 27-case matrix should also amplify preparation-to-preparation fluctuations in the three-seed ensemble. The analysis is a finite-difference diagnostic over the existing DEM matrix, not a critical-fluctuation or fluctuation-dissipation theorem.",
        "",
        "## Main result",
        "",
        f"The simple fluctuation-response hypothesis is not supported. The local response norm of the hot-susceptibility coordinate is only weakly related to the hot/cold preparation-CV ratio across the six targeted regimes (Spearman rho = {r_cv.spearman:.3f}, exact P = {r_cv.exact_p:.3f}), and it is also weak for hot-load CV normalized by structural CV (rho = {r_struct.spearman:.3f}, exact P = {r_struct.exact_p:.3f}). The largest combined fluctuation-response score occurs in {top.regime_id}, but this score should be read as a route classifier rather than a universal noise-response law.",
        "",
        f"The one suggestive bridge is between local matrix response and true-force loop gain in the subset with force-loop kernels (rho = {r_loop.spearman:.3f}, exact P = {r_loop.exact_p:.3f}); however, this uses only four overlapping finite-slope routes because not all six targeted regimes have long-cycle true-force kernels. This prevents overclaiming: the audit supports a multi-axis susceptibility picture, not a scalar fluctuation-response law.",
        "",
        "## Tests",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Regime summary",
        "",
        bridge[
            [
                "regime_id",
                "hot_susceptibility_index",
                "local_response_norm",
                "hot_cold_cv_ratio",
                "hot_structural_cv_ratio",
                "fluctuation_response_index",
                "dominant_response_axis",
                "sector",
                "susceptibility_slope",
            ]
        ].round(4).to_markdown(index=False),
        "",
        "## Manuscript use",
        "",
        "Allowed: local control sensitivity, preparation scatter and force-loop gain are separable susceptibility axes. This helps explain why intermediate routes can be preparation-sensitive even when their mean hot load is not maximal, while high-gain force-loop routes require true-force network evidence.",
        "",
        "Not allowed: do not claim a fluctuation-dissipation relation, criticality or a universal susceptibility law. The simple scalar fluctuation-response hypothesis fails in the six-regime preparation data.",
        "",
        "## Outputs",
        "",
        "- `figures/nphys_fig34_fluctuation_response_bridge.*`",
        "- `source_data/nphys_fluctuation_response_bridge_summary.csv`",
        "- `source_data/nphys_fluctuation_response_bridge_tests.csv`",
        "",
    ]
    (ROOT / "nature_physics_fluctuation_response_bridge.md").write_text("\n".join(lines))


def main() -> None:
    bridge, tests = build_bridge()
    draw(bridge, tests)
    write_report(bridge, tests)
    print(bridge.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
