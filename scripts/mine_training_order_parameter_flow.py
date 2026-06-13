#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INK = "#252A31"
MEMORY = "#345995"
ACTIVITY = "#C95F3F"
TRANSPORT = "#2F7F6F"
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


def panel(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def zscore(s: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(s, dtype=float)
    sd = np.std(arr, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return arr * 0.0
    return (arr - np.mean(arr)) / sd


def unit_interval(s: pd.Series | np.ndarray, invert: bool = False) -> np.ndarray:
    arr = np.asarray(s, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    out = (arr - lo) / (hi - lo) if hi > lo else arr * 0.0
    return 1.0 - out if invert else out


def exp_decay(n: np.ndarray, yinf: float, amp: float, tau: float) -> np.ndarray:
    return yinf + amp * np.exp(-(n - 1.0) / tau)


def exp_rise(n: np.ndarray, yinf: float, amp: float, tau: float) -> np.ndarray:
    return yinf - amp * np.exp(-(n - 1.0) / tau)


def fit_relaxation(cycle: np.ndarray, y: np.ndarray, mode: str) -> dict[str, float]:
    if mode == "decay":
        p0 = [float(np.mean(y[-5:])), float(y[0] - np.mean(y[-5:])), 3.0]
        bounds = ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 100.0])
        popt, _ = curve_fit(exp_decay, cycle, y, p0=p0, bounds=bounds, maxfev=50000)
        yhat = exp_decay(cycle, *popt)
    elif mode == "rise":
        p0 = [float(np.mean(y[-5:])), float(np.mean(y[-5:]) - y[0]), 3.0]
        bounds = ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 100.0])
        popt, _ = curve_fit(exp_rise, cycle, y, p0=p0, bounds=bounds, maxfev=50000)
        yhat = exp_rise(cycle, *popt)
    else:
        raise ValueError(mode)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    tau = float(popt[2])
    return {
        "mode": mode,
        "y_inf": float(popt[0]),
        "amplitude": float(popt[1]),
        "tau_cycles": tau,
        "half_life_cycles": tau * float(np.log(2.0)),
        "r2": r2,
        "start_value": float(y[0]),
        "end_value": float(y[-1]),
        "fold_change_end_over_start": float(y[-1] / y[0]) if y[0] != 0 else np.nan,
    }


def load_training_data() -> pd.DataFrame:
    memory = pd.read_csv(SRC / "nphys_fig1_memory_source.csv")
    entropy = pd.read_csv(SRC / "fig8_contact_entropy_source.csv")
    cols = [
        "cycle",
        "coordination_entropy_Hz",
        "cumulative_edge_entropy",
        "persistent_edge_fraction",
        "edge_pool",
        "nonGaussian_alpha2",
        "irreversible_work_proxy_J",
    ]
    df = memory.merge(entropy[cols], on="cycle", how="left")
    df["activity_index"] = (
        unit_interval(df["rms_displacement_m"])
        + unit_interval(df["irreversible_work_proxy_J"])
        + unit_interval(df["nonGaussian_alpha2"])
    ) / 3.0
    df["memory_index"] = (
        unit_interval(df["z_mean"])
        + unit_interval(df["contact_survival"])
        + unit_interval(df["k_eff_W_mK"])
        + unit_interval(df["Porosity"], invert=True)
        + unit_interval(df["phi_voro_std"], invert=True)
    ) / 5.0
    df["free_volume_width_norm"] = df["phi_voro_std"] / df["phi_voro_std"].iloc[0]
    df["transport_norm"] = df["k_eff_W_mK"] / df["k_eff_W_mK"].iloc[0]
    df["work_norm"] = df["irreversible_work_proxy_J"] / df["irreversible_work_proxy_J"].iloc[0]
    df["rms_norm"] = df["rms_displacement_m"] / df["rms_displacement_m"].iloc[0]
    df["edge_pool_norm"] = df["edge_pool"] / df["edge_pool"].iloc[0]
    df["cycle_fraction"] = (df["cycle"] - df["cycle"].min()) / (df["cycle"].max() - df["cycle"].min())
    return df


def make_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle = df["cycle"].to_numpy(float)
    targets = [
        ("rms_displacement_m", "decay", "activity"),
        ("irreversible_work_proxy_J", "decay", "activity"),
        ("nonGaussian_alpha2", "decay", "activity"),
        ("Porosity", "decay", "fabric"),
        ("phi_voro_std", "decay", "fabric"),
        ("contact_survival", "rise", "fabric"),
        ("z_mean", "rise", "fabric"),
        ("k_eff_W_mK", "rise", "transport"),
        ("activity_index", "decay", "reduced_state"),
        ("memory_index", "rise", "reduced_state"),
    ]
    rows = []
    for name, mode, family in targets:
        y = df[name].to_numpy(float)
        result = fit_relaxation(cycle, y, mode)
        result.update({"quantity": name, "family": family})
        rows.append(result)
    fits = pd.DataFrame(rows)

    pairs = [
        ("activity_index", "memory_index"),
        ("activity_index", "contact_survival"),
        ("activity_index", "z_mean"),
        ("activity_index", "k_eff_W_mK"),
        ("activity_index", "phi_voro_std"),
        ("memory_index", "cumulative_edge_entropy"),
        ("memory_index", "persistent_edge_fraction"),
        ("memory_index", "edge_pool_norm"),
    ]
    corr_rows = []
    for x_name, y_name in pairs:
        sp = spearmanr(df[x_name], df[y_name])
        pr = pearsonr(df[x_name], df[y_name])
        corr_rows.append(
            {
                "x": x_name,
                "y": y_name,
                "n": len(df),
                "spearman_rho": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
            }
        )
    return fits, pd.DataFrame(corr_rows)


def build_figure(df: pd.DataFrame, fits: pd.DataFrame, corr: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.05, 1.05])

    ax = fig.add_subplot(gs[:, 0])
    sc = ax.scatter(
        df["activity_index"],
        df["memory_index"],
        c=df["cycle"],
        cmap="viridis",
        s=28 + 42 * df["cycle_fraction"],
        edgecolor="white",
        lw=0.55,
        zorder=4,
    )
    ax.plot(df["activity_index"], df["memory_index"], color="#AEB5BE", lw=1.0, zorder=2)
    for cyc in [1, 2, 5, 10, 20]:
        row = df.loc[df["cycle"] == cyc].iloc[0]
        ax.text(row["activity_index"] + 0.025, row["memory_index"], str(cyc), fontsize=6.2, va="center")
    ax.annotate(
        "training flow",
        xy=(df["activity_index"].iloc[6], df["memory_index"].iloc[6]),
        xytext=(0.48, 0.42),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": INK},
        fontsize=7.0,
        color=INK,
    )
    rho = corr.query("x == 'activity_index' and y == 'memory_index'")["spearman_rho"].iloc[0]
    ax.text(0.05, 0.94, f"Spearman rho = {rho:.2f}", transform=ax.transAxes, fontsize=6.8, va="top", color=INK)
    ax.set_xlabel("activity order parameter, $A$")
    ax.set_ylabel("memory order parameter, $M$")
    ax.set_title("activity extinguishes as memory forms", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "a")
    cb = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label("cycle", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8)

    ax = fig.add_subplot(gs[0, 1])
    for col, label, color in [
        ("rms_norm", "particle activity", ACTIVITY),
        ("work_norm", "loss per cycle", "#9B4B65"),
    ]:
        ax.plot(df["cycle"], df[col], "o-", ms=2.8, lw=1.05, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("cycle")
    ax.set_ylabel("normalised activity")
    fast_half = fits.query("quantity == 'activity_index'")["half_life_cycles"].iloc[0]
    ax.text(0.07, 0.12, rf"$t_{{1/2}}^A={fast_half:.2f}$ cycles", transform=ax.transAxes, fontsize=6.6, color=ACTIVITY)
    ax.set_title("fast mechanical deactivation", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.8, loc="upper right")
    finish(ax, axis="both")
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for col, label, color in [
        ("z_mean", r"$Z$", MEMORY),
        ("contact_survival", "survival", "#6E78B7"),
        ("transport_norm", "transport", TRANSPORT),
    ]:
        y = unit_interval(df[col])
        ax.plot(df["cycle"], y, "o-", ms=2.8, lw=1.05, color=color, label=label)
    mem_half = fits.query("quantity == 'memory_index'")["half_life_cycles"].iloc[0]
    ax.text(0.06, 0.12, rf"$t_{{1/2}}^M={mem_half:.1f}$ cycles", transform=ax.transAxes, fontsize=6.6, color=MEMORY)
    ax.set_xlabel("cycle")
    ax.set_ylabel("trained-state coordinate")
    ax.set_title("slower reservoir writing", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.6, loc="lower right")
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df["cycle"], df["free_volume_width_norm"], "o-", color=MEMORY, ms=2.8, lw=1.05, label="free-volume width")
    ax2 = ax.twinx()
    ax2.plot(df["cycle"], df["cumulative_edge_entropy"], "o-", color=ACTIVITY, ms=2.8, lw=1.05, label="edge entropy")
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\sigma_\phi/\sigma_{\phi,1}$", color=MEMORY)
    ax2.set_ylabel("cumulative edge entropy", color=ACTIVITY)
    ax.tick_params(axis="y", colors=MEMORY)
    ax2.tick_params(axis="y", colors=ACTIVITY, width=0.65)
    ax2.spines["top"].set_visible(False)
    ax.set_title("free volume narrows while contacts explore", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "d", x=-0.18, y=1.14)

    ax = fig.add_subplot(gs[1, 2])
    show = fits[fits["quantity"].isin(["activity_index", "memory_index", "z_mean", "Porosity", "k_eff_W_mK", "contact_survival"])].copy()
    show["label"] = show["quantity"].map(
        {
            "activity_index": "activity",
            "memory_index": "memory",
            "z_mean": "coordination",
            "Porosity": "porosity",
            "k_eff_W_mK": "transport",
            "contact_survival": "survival",
        }
    )
    show = show.sort_values("half_life_cycles")
    colors = [ACTIVITY if q == "activity_index" else MEMORY if q in {"memory_index", "z_mean", "Porosity", "contact_survival"} else TRANSPORT for q in show["quantity"]]
    ax.barh(np.arange(len(show)), show["half_life_cycles"], color=colors, height=0.68)
    ax.set_yticks(np.arange(len(show)), show["label"])
    ax.set_xlabel("half-life / half-rise (cycles)")
    ax.set_title("two training time scales", fontsize=7.5, pad=5)
    finish(ax, axis="x")
    panel(ax, "e")

    out = FIG / "nphys_fig29_training_order_parameter_flow"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, fits: pd.DataFrame, corr: pd.DataFrame) -> None:
    def val(quantity: str, col: str) -> float:
        return float(fits.query("quantity == @quantity")[col].iloc[0])

    rho_am = float(corr.query("x == 'activity_index' and y == 'memory_index'")["spearman_rho"].iloc[0])
    p_am = float(corr.query("x == 'activity_index' and y == 'memory_index'")["spearman_p"].iloc[0])
    text = f"""# Training order-parameter flow audit

Date: 2026-06-12

## Question

The manuscript describes a granular bed that is trained by thermal cycling and later read out as cold memory or hot overload. This audit asks whether the free-surface training data contain a measurable statistical-physics trajectory rather than only monotonic compaction.

## Main result

The training dynamics separate into two time scales. The reduced activity order parameter A, built from particle displacement, non-Gaussian displacement bursts and an irreversible-work proxy, decays with a fitted half-life of {val('activity_index','half_life_cycles'):.2f} cycles (R2 = {val('activity_index','r2'):.3f}). The reduced memory order parameter M, built from coordination, contact survival, conductivity, porosity and free-volume width, rises with a half-rise of {val('memory_index','half_life_cycles'):.2f} cycles (R2 = {val('memory_index','r2'):.3f}). Activity and memory are strongly anti-correlated across the training trajectory (Spearman rho = {rho_am:.3f}, P = {p_am:.2e}, n = {len(df)}).

Individual variables show the same hierarchy. Particle-scale activity and irreversible work lose more than 90% of their initial value by cycle 20. Coordination and the thermal-conductivity proxy saturate more slowly, with fitted half-rises of {val('z_mean','half_life_cycles'):.2f} and {val('k_eff_W_mK','half_life_cycles'):.2f} cycles. Porosity relaxes with a half-life of {val('Porosity','half_life_cycles'):.2f} cycles. Contact survival jumps quickly, indicating that the initially violent rearrangement is rapidly replaced by smaller reversible excursions.

## Interpretation allowed in the manuscript

Allowed: thermal cycling produces an absorbing-state-like training flow in which mechanical activity is rapidly extinguished while a fabric/transport memory reservoir continues to be written over several cycles.

Not allowed: this is not evidence for a sharp reversible-irreversible critical point. The dataset is a single free-surface trajectory, so the result should be used as an order-parameter diagnostic and as motivation for the two-coordinate return map, not as finite-size scaling.

## Generated files

- `figures/nphys_fig29_training_order_parameter_flow.*`
- `source_data/nphys_training_order_parameter_flow_cycle_metrics.csv`
- `source_data/nphys_training_order_parameter_flow_fits.csv`
- `source_data/nphys_training_order_parameter_flow_correlations.csv`
"""
    (ROOT / "nature_physics_training_order_parameter_flow.md").write_text(text, encoding="utf-8")


def main() -> None:
    df = load_training_data()
    fits, corr = make_metrics(df)
    df.to_csv(SRC / "nphys_training_order_parameter_flow_cycle_metrics.csv", index=False)
    fits.to_csv(SRC / "nphys_training_order_parameter_flow_fits.csv", index=False)
    corr.to_csv(SRC / "nphys_training_order_parameter_flow_correlations.csv", index=False)
    build_figure(df, fits, corr)
    write_report(df, fits, corr)
    a_half = fits.query("quantity == 'activity_index'")["half_life_cycles"].iloc[0]
    m_half = fits.query("quantity == 'memory_index'")["half_life_cycles"].iloc[0]
    print(f"Training order-parameter flow complete: A half-life={a_half:.2f} cycles, M half-rise={m_half:.2f} cycles.")


if __name__ == "__main__":
    main()
