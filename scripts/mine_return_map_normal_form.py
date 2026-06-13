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
GAIN = SRC / "nphys_nonreciprocal_transient_map_metrics.csv"
AUTOCORR = SRC / "nphys_breathing_memory_kernel_autocorrelation.csv"

INK = "#252A31"
GRID = "#E7EAEE"
MUTED = "#8D99A6"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
RED = "#B6423E"
VIOLET = "#7E6AAE"
DARKRED = "#8D3138"
GREEN = "#4F8B67"
COLORS = {"R1": BLUE, "R3": GOLD, "R5": VIOLET, "R6": RED, "R6c": DARKRED}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}


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


def route_matrix(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [row["A11_memory_to_memory"], row["A12_hot_to_memory"]],
            [row["A21_memory_to_hot"], row["A22_hot_to_hot"]],
        ],
        dtype=float,
    )


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(MAPS)
    gain = pd.read_csv(GAIN)[["regime_id", "one_step_gain", "peak_normalized_gain", "nonnormality"]]
    rows = []
    eig_rows = []
    for _, row in maps.iterrows():
        rid = row["regime_id"]
        A = route_matrix(row)
        eig = np.linalg.eigvals(A)
        eig_sorted = sorted(eig, key=lambda z: abs(z), reverse=True)
        dominant = eig_sorted[0]
        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]
        trace_half = 0.5 * (a + d)
        diagonal_split = 0.5 * (a - d)
        reciprocal_shear = 0.5 * (b + c)
        circulation = 0.5 * (c - b)
        rho = float(np.max(np.abs(eig)))
        negative_modes = [float(abs(np.real(z))) for z in eig if abs(np.imag(z)) < 1e-12 and np.real(z) < 0]
        alternating_strength = max(negative_modes) if negative_modes else 0.0
        for k, z in enumerate(eig_sorted, start=1):
            eig_rows.append(
                {
                    "regime_id": rid,
                    "mode_rank": k,
                    "eigenvalue_real": float(np.real(z)),
                    "eigenvalue_imag": float(np.imag(z)),
                    "eigenvalue_abs": float(abs(z)),
                    "dominant": k == 1,
                }
            )
        rows.append(
            {
                "regime_id": rid,
                "trace_half": trace_half,
                "diagonal_split": diagonal_split,
                "reciprocal_shear": reciprocal_shear,
                "circulation": circulation,
                "dominant_eigenvalue": float(np.real(dominant)),
                "dominant_abs": float(abs(dominant)),
                "alternating_mode_strength": alternating_strength,
                "two_cycle_persistence": alternating_strength**2,
                "spectral_radius": rho,
                "stability_margin": 1.0 - rho,
                "mean_overload_number": row["mean_overload_number"],
                "mean_dimensionless_loop_number": row["mean_dimensionless_loop_number"],
                "r2_next_memory": row["r2_next_memory"],
                "r2_next_hot_excitation": row["r2_next_hot_excitation"],
            }
        )
    route = pd.DataFrame(rows).merge(gain, on="regime_id", how="left")

    ac = pd.read_csv(AUTOCORR)
    bridge = ac[ac["series"].isin(["loop activation", "overload"])].copy()
    bridge["parity"] = np.where(bridge["lag_cycles"] % 2 == 0, "even", "odd")
    parity = (
        bridge.groupby(["series", "parity"], observed=True)
        .agg(
            n=("lag_cycles", "count"),
            mean_pooled_spearman=("pooled_spearman", "mean"),
            mean_route_spearman=("route_mean_spearman", "mean"),
            min_route_spearman=("route_min_spearman", "min"),
            max_route_spearman=("route_max_spearman", "max"),
        )
        .reset_index()
    )

    corr_rows = []
    for x, y in [
        ("alternating_mode_strength", "peak_normalized_gain"),
        ("circulation", "peak_normalized_gain"),
        ("stability_margin", "peak_normalized_gain"),
        ("dominant_eigenvalue", "mean_overload_number"),
        ("two_cycle_persistence", "peak_normalized_gain"),
    ]:
        stat = spearmanr(route[x], route[y])
        corr_rows.append({"x": x, "y": y, "n": len(route), "spearman": float(stat.statistic), "p_value": float(stat.pvalue)})
    corr = pd.DataFrame(corr_rows)
    return route, pd.DataFrame(eig_rows), bridge, parity, corr


def make_figure(route: pd.DataFrame, eig: pd.DataFrame, bridge: pd.DataFrame, parity: pd.DataFrame, corr: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.35, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.05], height_ratios=[1.02, 0.98])

    ax = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#B5BDC7", lw=0.8)
    ax.fill_between([-1.0, 0.0], -1.0, 1.0, color="#F6E0C8", alpha=0.28, zorder=0)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    for _, row in eig.iterrows():
        rid = row["regime_id"]
        ax.scatter(
            row["eigenvalue_real"],
            row["eigenvalue_imag"],
            s=48 if row["dominant"] else 28,
            color=COLORS.get(rid, MUTED),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            lw=0.55,
            alpha=0.92 if row["dominant"] else 0.62,
            zorder=3,
        )
        if row["dominant"]:
            offsets = {
                "R1": (-0.10, 0.12),
                "R3": (0.035, 0.06),
                "R5": (-0.06, 0.11),
                "R6": (-0.02, -0.14),
                "R6c": (-0.02, 0.12),
            }
            dx, dy = offsets.get(rid, (0.035, 0.035))
            ax.text(row["eigenvalue_real"] + dx, row["eigenvalue_imag"] + dy, rid, fontsize=6.3, color=COLORS.get(rid, MUTED))
    ax.text(-0.94, 0.80, "alternating\nmemory sector", fontsize=6.1, color=GOLD, ha="left", va="top")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Re(eigenvalue)")
    ax.set_ylabel("Im(eigenvalue)")
    ax.set_title("all route maps are stable; R3 is near a flip mode", loc="left", pad=4)
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    cols = ["trace_half", "diagonal_split", "reciprocal_shear", "circulation"]
    labels = [r"$\frac{1}{2}{\rm tr}A$", "self split", "reciprocal shear", "circulation"]
    heat = route.set_index("regime_id").loc[["R1", "R3", "R5", "R6", "R6c"], cols]
    im = ax.imshow(heat.to_numpy(float), cmap="RdBu_r", vmin=-1.05, vmax=1.05, aspect="auto")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.9, color="white" if abs(v) > 0.55 else INK)
    ax.set_title("normal-form decomposition", loc="left", pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    for _, row in route.iterrows():
        rid = row["regime_id"]
        ax.scatter(
            row["stability_margin"],
            row["peak_normalized_gain"],
            s=42 + 8.5 * max(row["mean_overload_number"], 0),
            color=COLORS.get(rid, MUTED),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            lw=0.55,
            zorder=3,
        )
        ax.text(row["stability_margin"] + 0.012, row["peak_normalized_gain"], rid, fontsize=6.3, color=COLORS.get(rid, MUTED), va="center")
    row = corr[(corr["x"] == "stability_margin") & (corr["y"] == "peak_normalized_gain")].iloc[0]
    ax.text(0.05, 0.96, rf"$\rho={row['spearman']:.2f}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(r"stability margin, $1-\rho(A)$")
    ax.set_ylabel(r"hidden gain, $\max_k\|A^k\|/\rho^k$")
    ax.set_title("hidden gain grows near weak damping", loc="left", pad=4)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    colors = {"loop activation": GREEN, "overload": RED}
    markers = {"loop activation": "o", "overload": "s"}
    for series, g in bridge.groupby("series", sort=True):
        ax.plot(
            g["lag_cycles"],
            g["pooled_spearman"],
            color=colors[series],
            marker=markers[series],
            lw=1.05,
            ms=3.5,
            label=series,
        )
    even = bridge[bridge["lag_cycles"] % 2 == 0]["lag_cycles"].unique()
    for lag in even:
        ax.axvspan(lag - 0.22, lag + 0.22, color="#EAF2EA", alpha=0.65, zorder=0)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("lag (cycles)")
    ax.set_ylabel("pooled Spearman autocorrelation")
    ax.set_title("even-lag memory matches a finite flip kernel", loc="left", pad=4)
    ax.legend(loc="upper right", fontsize=5.9, handlelength=1.1)
    finish(ax)
    panel(ax, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig45_return_map_normal_form.{ext}", **kwargs)
    plt.close(fig)


def write_report(route: pd.DataFrame, eig: pd.DataFrame, bridge: pd.DataFrame, parity: pd.DataFrame, corr: pd.DataFrame) -> None:
    r3 = route.loc[route["regime_id"] == "R3"].iloc[0]
    loop_even = parity[(parity["series"] == "loop activation") & (parity["parity"] == "even")].iloc[0]
    loop_odd = parity[(parity["series"] == "loop activation") & (parity["parity"] == "odd")].iloc[0]
    overload_even = parity[(parity["series"] == "overload") & (parity["parity"] == "even")].iloc[0]
    overload_odd = parity[(parity["series"] == "overload") & (parity["parity"] == "odd")].iloc[0]
    lines = [
        "# Return-map normal-form audit",
        "",
        "This reserve audit reduces each fitted two-dimensional breathing map to normal-form ingredients: damping, self-coordinate split, reciprocal shear and antisymmetric circulation. It asks whether the finite-cycle memory kernel can be interpreted as a stable but weakly damped flip-like response.",
        "",
        "## Route normal-form metrics",
        "",
        route.round(4).to_markdown(index=False),
        "",
        "## Eigenvalues",
        "",
        eig.round(4).to_markdown(index=False),
        "",
        "## Even/odd lag bridge",
        "",
        parity.round(4).to_markdown(index=False),
        "",
        "## Correlations",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## Mechanistic reading",
        "",
        f"All fitted route maps remain inside the unit circle, so the return map is stable in the measured window. R3 is the clearest weakly damped flip route: its dominant eigenvalue is {r3['dominant_eigenvalue']:.3f}, its stability margin is {r3['stability_margin']:.3f}, and its hidden normalized gain is {r3['peak_normalized_gain']:.2f}. This gives a mathematical reason why finite-cycle breathing memory can be even-lag dominated without invoking a critical point.",
        "",
        f"The lag bridge is consistent with that reading: loop activation has mean pooled Spearman {loop_even['mean_pooled_spearman']:.2f} over even lags and {loop_odd['mean_pooled_spearman']:.2f} over odd lags; overload gives {overload_even['mean_pooled_spearman']:.2f} over even lags and {overload_odd['mean_pooled_spearman']:.2f} over odd lags. This is a consistency bridge between the five-route return-map audit and the three-route breathing-kernel audit, not a proof of a universal frequency.",
        "",
        "Interpretation boundary: the normal form is route-conditioned and fitted from finite cycle windows. It supports stable-but-excitable and finite-memory language, but it does not establish a Hopf bifurcation, period-doubling transition or universal oscillator.",
    ]
    (ROOT / "nature_physics_return_map_normal_form.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    route, eig, bridge, parity, corr = build_metrics()
    route.to_csv(SRC / "nphys_return_map_normal_form_route_metrics.csv", index=False)
    eig.to_csv(SRC / "nphys_return_map_normal_form_eigenvalues.csv", index=False)
    bridge.to_csv(SRC / "nphys_return_map_normal_form_lag_bridge.csv", index=False)
    parity.to_csv(SRC / "nphys_return_map_normal_form_parity_summary.csv", index=False)
    corr.to_csv(SRC / "nphys_return_map_normal_form_correlations.csv", index=False)
    make_figure(route, eig, bridge, parity, corr)
    write_report(route, eig, bridge, parity, corr)
    print("Wrote return-map normal-form products")
    print(route[["regime_id", "dominant_eigenvalue", "stability_margin", "peak_normalized_gain", "circulation", "mean_overload_number"]].round(3).to_string(index=False))
    print(parity.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
