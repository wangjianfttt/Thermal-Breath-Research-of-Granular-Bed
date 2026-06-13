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

MAPS = SRC / "nphys_return_map_phase_portrait_route_maps.csv"

COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#242A31"
GRID = "#E7EAEE"
MUTED = "#7B8490"
ACCENT = "#B6423E"


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_matrix(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [row["A11_memory_to_memory"], row["A12_hot_to_memory"]],
            [row["A21_memory_to_hot"], row["A22_hot_to_hot"]],
        ],
        dtype=float,
    )


def compute_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(MAPS)
    rows = []
    curves = []
    for _, row in maps.iterrows():
        rid = row["regime_id"]
        A = route_matrix(row)
        eig = np.linalg.eigvals(A)
        rho = float(np.max(np.abs(eig)))
        symmetric = 0.5 * (A + A.T)
        antisymmetric = 0.5 * (A - A.T)
        comm = A.T @ A - A @ A.T
        norm_a = float(np.linalg.norm(A, ord="fro"))
        nonnormality = float(np.linalg.norm(comm, ord="fro") / (norm_a**2 + 1e-12))
        coupling_asymmetry = float(A[0, 1] - A[1, 0])
        antisymmetric_strength = float(abs(antisymmetric[0, 1]))
        reciprocal_strength = float(symmetric[0, 1])
        gains = []
        normalized_gains = []
        for k in range(1, 16):
            Ak = np.linalg.matrix_power(A, k)
            gain = float(np.linalg.norm(Ak, ord=2))
            normalized_gain = float(gain / (rho**k + 1e-12))
            gains.append(gain)
            normalized_gains.append(normalized_gain)
            curves.append(
                {
                    "regime_id": rid,
                    "step": k,
                    "transient_gain": gain,
                    "normalized_transient_gain": normalized_gain,
                }
            )
        rows.append(
            {
                "regime_id": rid,
                "spectral_radius": rho,
                "one_step_gain": gains[0],
                "peak_transient_gain": max(gains),
                "step_at_peak_gain": int(np.argmax(gains) + 1),
                "peak_normalized_gain": max(normalized_gains),
                "coupling_asymmetry": coupling_asymmetry,
                "antisymmetric_strength": antisymmetric_strength,
                "reciprocal_coupling": reciprocal_strength,
                "nonnormality": nonnormality,
                "mean_overload_number": row["mean_overload_number"],
                "mean_dimensionless_loop_number": row["mean_dimensionless_loop_number"],
                "r2_next_memory": row["r2_next_memory"],
                "r2_next_hot_excitation": row["r2_next_hot_excitation"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def draw_figure(metrics: pd.DataFrame, curves: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel(ax_a, "a")
    for _, row in metrics.iterrows():
        rid = row["regime_id"]
        ax_a.scatter(
            row["spectral_radius"],
            row["one_step_gain"],
            s=55,
            color=COLORS.get(rid, MUTED),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax_a.text(row["spectral_radius"] + 0.01, row["one_step_gain"], rid, color=COLORS.get(rid, MUTED), fontsize=6.5, va="center")
    ax_a.axvline(1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_a.axhline(1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_a.fill_between([0.45, 1.0], 1, 1.48, color="#F5D7CF", alpha=0.35, zorder=0)
    ax_a.text(0.47, 1.42, "stable but\ntransiently amplified", color=ACCENT, fontsize=6.4, va="top")
    ax_a.set_xlim(0.45, 1.02)
    ax_a.set_ylim(0.55, 1.48)
    ax_a.set_xlabel("spectral radius")
    ax_a.set_ylabel(r"one-step gain $\|\mathbf{A}\|_2$")
    ax_a.set_title("stable maps can amplify a breath", loc="left", pad=4)
    finish(ax_a)

    panel(ax_b, "b")
    for rid, g in curves.groupby("regime_id", sort=True):
        ax_b.plot(
            g["step"],
            g["normalized_transient_gain"],
            color=COLORS.get(rid, MUTED),
            lw=1.05,
            marker=MARKERS.get(rid, "o"),
            ms=3,
            label=rid,
        )
    ax_b.axhline(1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_b.set_xlabel("cycle-map powers")
    ax_b.set_ylabel(r"$\|\mathbf{A}^k\|_2/\rho^k$")
    ax_b.set_title("non-normal gain over eigenvalue decay", loc="left", pad=4)
    ax_b.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=1.1, columnspacing=0.6)
    finish(ax_b)

    panel(ax_c, "c")
    order = metrics.sort_values("mean_overload_number")["regime_id"].tolist()
    x = np.arange(len(order))
    d = metrics.set_index("regime_id").loc[order]
    ax_c.bar(x - 0.17, d["coupling_asymmetry"], width=0.32, color=ACCENT, label=r"$A_{M\Psi}-A_{\Psi M}$")
    ax_c.bar(x + 0.17, d["nonnormality"], width=0.32, color="#AAB4C0", label="non-normality")
    ax_c.axhline(0, color="#AEB6C0", lw=0.7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(order)
    ax_c.set_ylabel("dimensionless map metric")
    ax_c.set_title("non-reciprocal coupling is route dependent", loc="left", pad=4)
    ax_c.legend(loc="upper left", fontsize=5.8, handlelength=1.1)
    finish(ax_c, "y")

    panel(ax_d, "d")
    x = metrics["antisymmetric_strength"]
    y = metrics["peak_normalized_gain"]
    for _, row in metrics.iterrows():
        rid = row["regime_id"]
        ax_d.scatter(
            row["antisymmetric_strength"],
            row["peak_normalized_gain"],
            s=42 + 11 * max(row["mean_overload_number"], 0),
            color=COLORS.get(rid, MUTED),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax_d.text(row["antisymmetric_strength"] + 0.012, row["peak_normalized_gain"], rid, color=COLORS.get(rid, MUTED), fontsize=6.5, va="center")
    rho = spearmanr(x, y).statistic
    ax_d.text(0.05, 0.96, rf"$\rho={rho:.2f}$", transform=ax_d.transAxes, va="top", ha="left")
    ax_d.set_xlabel("antisymmetric coupling strength")
    ax_d.set_ylabel("peak normalized gain")
    ax_d.set_title("circulation controls hidden amplification", loc="left", pad=4)
    finish(ax_d)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig18_nonreciprocal_transient_map.{ext}", **kwargs)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, curves: pd.DataFrame) -> None:
    rho_ag = spearmanr(metrics["antisymmetric_strength"], metrics["peak_normalized_gain"]).statistic
    rho_overload = spearmanr(metrics["peak_transient_gain"], metrics["mean_overload_number"]).statistic
    lines = [
        "# Non-reciprocal transient-map audit",
        "",
        "This audit asks why a dissipative return map can still show hot overload. The tested mechanism is non-normal transient amplification: eigenvalues can decay while singular-vector perturbations grow over one step.",
        "",
        "## Route metrics",
        "",
        metrics.round(4).to_markdown(index=False),
        "",
        "## Gain curves",
        "",
        curves.round(4).to_markdown(index=False),
        "",
        "## Manuscript-safe interpretation",
        "",
        f"- Antisymmetric coupling strength correlates with peak normalized gain across the five routes with Spearman rho={rho_ag:.2f}.",
        f"- Peak raw transient gain is not a monotonic overload predictor across only five routes (Spearman rho={rho_overload:.2f}); do not claim it explains all overload.",
        "- R3 is the cleanest example of stable-but-amplifying behaviour: spectral radius < 1 but one-step gain > 1.",
        "- This supports a non-equilibrium stability interpretation: breathing can remain dissipative while selected perturbation directions are transiently amplified.",
    ]
    (ROOT / "nature_physics_nonreciprocal_transient_map_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    metrics, curves = compute_metrics()
    metrics.to_csv(SRC / "nphys_nonreciprocal_transient_map_metrics.csv", index=False)
    curves.to_csv(SRC / "nphys_nonreciprocal_transient_map_gain_curves.csv", index=False)
    draw_figure(metrics, curves)
    write_report(metrics, curves)
    print("Wrote non-reciprocal transient-map audit")
    print(metrics[["regime_id", "spectral_radius", "one_step_gain", "peak_normalized_gain", "coupling_asymmetry", "nonnormality", "mean_overload_number"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
