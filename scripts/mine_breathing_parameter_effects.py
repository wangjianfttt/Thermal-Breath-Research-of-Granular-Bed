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
ACCENT = "#B6423E"


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


def within_spearman(df: pd.DataFrame, predictor: str, target: str) -> dict[str, float | int | str]:
    d = df[["regime_id", predictor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    raw = spearmanr(d[predictor], d[target], nan_policy="omit") if len(d) >= 6 else None
    dc = d.copy()
    dc[predictor] = dc[predictor] - dc.groupby("regime_id")[predictor].transform("mean")
    dc[target] = dc[target] - dc.groupby("regime_id")[target].transform("mean")
    wc = spearmanr(dc[predictor], dc[target], nan_policy="omit") if len(dc) >= 6 else None
    return {
        "predictor": predictor,
        "target": target,
        "spearman_raw": float(raw.statistic) if raw else np.nan,
        "p_raw": float(raw.pvalue) if raw else np.nan,
        "spearman_within_regime": float(wc.statistic) if wc else np.nan,
        "p_within_regime": float(wc.pvalue) if wc else np.nan,
        "n": int(len(d)),
    }


def segment_label(cycle: int) -> str:
    if cycle <= 10:
        return "early"
    if cycle <= 20:
        return "middle"
    return "late"


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(SRC / "nphys_memory_induced_breathing_cycle_metrics.csv")
    m = m.sort_values(["regime_id", "cycle"]).copy()

    m["breath_amplitude"] = m["inhalation_norm"]
    m["breath_imprint"] = m["exhalation_imprint_norm"]
    m["imprint_efficiency"] = m["closure_ratio"]
    m["overload_cost"] = m["force_p99_hot_minus_cold"] / (m["breath_amplitude"] + 1e-12)
    m["loop_cost"] = m["force_h1_birth_force_share_hot_minus_cold"] / (m["breath_amplitude"] + 1e-12)
    m["memory_dose"] = m["breath_amplitude"] * m["imprint_efficiency"]
    m["loop_dose"] = m["breath_amplitude"] * m["force_h1_birth_force_share_hot_minus_cold"]

    m["amplitude_next_change"] = m.groupby("regime_id")["breath_amplitude"].shift(-1) - m["breath_amplitude"]
    m["amplitude_drop_fraction"] = -m["amplitude_next_change"] / (m["breath_amplitude"] + 1e-12)
    m["breath_irregularity"] = m.groupby("regime_id")["breath_amplitude"].diff().abs()
    m["segment"] = m["cycle"].astype(int).map(segment_label)

    rows: list[dict[str, float | str | int]] = []
    for rid, g in m.groupby("regime_id", sort=True):
        g = g.sort_values("cycle")
        amp = g["breath_amplitude"].to_numpy(float)
        cycles = g["cycle"].to_numpy(float)
        positive = amp > 0
        slope, intercept = np.polyfit(cycles[positive], np.log(amp[positive]), 1)
        half_life = np.log(0.5) / slope if slope < 0 else np.inf
        rows.append(
            {
                "regime_id": rid,
                "log_amplitude_slope_per_cycle": float(slope),
                "amplitude_half_life_cycles": float(half_life),
                "mean_amplitude": float(np.mean(amp)),
                "cv_amplitude": float(np.std(amp, ddof=0) / np.mean(amp)),
                "mean_imprint_efficiency": float(g["imprint_efficiency"].mean()),
                "mean_overload_cost": float(g["overload_cost"].mean()),
                "mean_loop_cost": float(g["loop_cost"].mean()),
                "mean_irregularity": float(g["breath_irregularity"].mean()),
            }
        )
    regime = pd.DataFrame(rows)

    segment = (
        m.groupby(["regime_id", "segment"], observed=True)
        .agg(
            n=("cycle", "count"),
            amplitude_mean=("breath_amplitude", "mean"),
            amplitude_cv=("breath_amplitude", lambda s: float(s.std(ddof=0) / s.mean())),
            irregularity_mean=("breath_irregularity", "mean"),
            imprint_efficiency_mean=("imprint_efficiency", "mean"),
            overload_cost_mean=("overload_cost", "mean"),
            loop_cost_mean=("loop_cost", "mean"),
            next_fabric_imprint_mean=("Z_geom_next_cold_minus_current", "mean"),
            next_loop_imprint_mean=("force_h1_birth_force_share_next_cold_minus_current", "mean"),
        )
        .reset_index()
    )

    pairs = [
        ("breath_amplitude", "force_p99_hot_minus_cold"),
        ("breath_amplitude", "Z_geom_next_cold_minus_current"),
        ("breath_amplitude", "force_h1_birth_force_share_next_cold_minus_current"),
        ("imprint_efficiency", "force_p99_hot_minus_cold"),
        ("imprint_efficiency", "Z_geom_next_cold_minus_current"),
        ("imprint_efficiency", "force_h1_birth_force_share_next_cold_minus_current"),
        ("overload_cost", "Z_geom_next_cold_minus_current"),
        ("loop_cost", "force_p99_hot_minus_cold"),
        ("breath_irregularity", "force_p99_hot_minus_cold"),
        ("breath_irregularity", "Z_geom_next_cold_minus_current"),
        ("memory_dose", "Z_geom_next_cold_minus_current"),
        ("loop_dose", "force_p99_hot_minus_cold"),
        ("amplitude_drop_fraction", "force_p99_hot_minus_cold"),
    ]
    corr = pd.DataFrame([within_spearman(m, x, y) for x, y in pairs])
    return m, corr, regime.merge(segment, on="regime_id", how="left")


def plot_figure(metrics: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.43)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel(ax_a, "a")
    for rid, g in metrics.groupby("regime_id", sort=True):
        ax_a.plot(g["cycle"], g["breath_amplitude"], color=REGIME_COLORS[rid], lw=1.05)
        ax_a.scatter(g["cycle"], g["breath_amplitude"], s=12, color=REGIME_COLORS[rid], edgecolor="white", lw=0.25, zorder=3)
        slope = summary.loc[summary["regime_id"] == rid, "log_amplitude_slope_per_cycle"].iloc[0]
        half = summary.loc[summary["regime_id"] == rid, "amplitude_half_life_cycles"].iloc[0]
        label = f"{rid}: t1/2={half:.1f}" if np.isfinite(half) else f"{rid}: no damping"
        ax_a.text(g["cycle"].max() + 0.35, g["breath_amplitude"].iloc[-1], label, color=REGIME_COLORS[rid], fontsize=6.1, va="center")
    ax_a.set_xlabel("cycle")
    ax_a.set_ylabel("breathing amplitude")
    ax_a.set_title("effective rhythm: damping per breath", loc="left", pad=2)
    ax_a.set_xlim(0.5, 36.0)
    finish(ax_a)

    panel(ax_b, "b")
    for rid, g in metrics.groupby("regime_id", sort=True):
        ax_b.scatter(
            g["imprint_efficiency"],
            g["force_p99_hot_minus_cold"],
            s=22,
            marker=REGIME_MARKERS[rid],
            color=REGIME_COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.86,
            label=rid,
        )
    row = corr[(corr["predictor"] == "imprint_efficiency") & (corr["target"] == "force_p99_hot_minus_cold")].iloc[0]
    ax_b.text(0.05, 0.95, rf"$\rho_{{within}}={row.spearman_within_regime:.2f}$" + f"\nP={row.p_within_regime:.1e}", transform=ax_b.transAxes, ha="left", va="top")
    ax_b.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax_b.set_xlabel("imprint efficiency\nnext-cold imprint / inhale")
    ax_b.set_ylabel("hot overload")
    ax_b.set_title("efficient exhalation buffers overload", loc="left", pad=2)
    ax_b.legend(ncol=3, loc="upper right", handlelength=1.0, columnspacing=0.75)
    finish(ax_b)

    panel(ax_c, "c")
    plot_data = metrics.copy()
    x = "loop_cost"
    y = "force_p99_hot_minus_cold"
    plot_data[f"{x}_wc"] = plot_data[x] - plot_data.groupby("regime_id")[x].transform("mean")
    plot_data[f"{y}_wc"] = plot_data[y] - plot_data.groupby("regime_id")[y].transform("mean")
    for rid, g in plot_data.groupby("regime_id", sort=True):
        ax_c.scatter(
            g[f"{x}_wc"],
            g[f"{y}_wc"],
            s=22,
            marker=REGIME_MARKERS[rid],
            color=REGIME_COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.86,
        )
    xx = plot_data[f"{x}_wc"].to_numpy(float)
    yy = plot_data[f"{y}_wc"].to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    coef = np.polyfit(xx[ok], yy[ok], 1)
    line_x = np.linspace(xx[ok].min(), xx[ok].max(), 100)
    ax_c.plot(line_x, coef[0] * line_x + coef[1], color=INK, lw=0.75)
    row = corr[(corr["predictor"] == "loop_cost") & (corr["target"] == "force_p99_hot_minus_cold")].iloc[0]
    ax_c.text(0.05, 0.95, rf"$\rho_{{within}}={row.spearman_within_regime:.2f}$" + f"\nP={row.p_within_regime:.1e}", transform=ax_c.transAxes, ha="left", va="top")
    ax_c.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax_c.axvline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax_c.set_xlabel("regime-centred loop cost\nloop activation / inhale")
    ax_c.set_ylabel("regime-centred hot overload")
    ax_c.set_title("breathing cost controls damage", loc="left", pad=2)
    finish(ax_c)

    panel(ax_d, "d")
    cols = ["amplitude_mean", "amplitude_cv", "imprint_efficiency_mean", "overload_cost_mean", "loop_cost_mean"]
    heat = summary.drop_duplicates(["regime_id", "segment"])[["regime_id", "segment", *cols]].copy()
    for col in cols:
        heat[col] = (heat[col] - heat[col].mean()) / heat[col].std(ddof=0)
    heat["row"] = heat["regime_id"] + "-" + heat["segment"].str[0]
    image = heat[cols].to_numpy(float)
    im = ax_d.imshow(image, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax_d.set_yticks(np.arange(len(heat)))
    ax_d.set_yticklabels(heat["row"])
    ax_d.set_xticks(np.arange(len(cols)))
    ax_d.set_xticklabels(["amp.", "irreg.", "imprint\neff.", "overload\ncost", "loop\ncost"])
    ax_d.set_title("breathing parameters define route state", loc="left", pad=2)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(size=2, width=0.5)
    cbar.set_label("z-score", labelpad=2)

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig14_breathing_parameter_effects.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Breathing-parameter effects",
        "",
        "The simulations use a fixed imposed thermal cycle period, so they do not directly test externally varied breathing frequency. This audit instead extracts effective rhythm and breathing parameters from the cycle-to-cycle response.",
        "",
        "## Definitions",
        "",
        "- Breathing amplitude: cold-to-hot standardized state distance.",
        "- Effective rhythm/damping: exponential slope of breathing amplitude per cycle and corresponding half-life.",
        "- Imprint efficiency: next-cold imprint divided by cold-to-hot amplitude.",
        "- Overload cost: hot overload divided by breathing amplitude.",
        "- Loop cost: force-loop activation divided by breathing amplitude.",
        "- Irregularity: absolute breath-to-breath amplitude change.",
        "",
        "## Correlation audit",
        "",
        corr.to_markdown(index=False),
        "",
        "## Regime and segment summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The imposed thermal frequency is fixed, but the bed develops an effective breathing rhythm. R6 has a short damping half-life and a high early overload cost, indicating a violent transient inhale. R1 is buffered with low loop cost and low imprint efficiency after early training. R3 maintains a lossy breathing response with persistent imprint efficiency. Across cycles, imprint efficiency is anti-correlated with hot overload, whereas loop cost is strongly correlated with overload. Thus breathing parameters do not merely describe the bed; they feed back on whether the next cycle is buffered, lossy or damaging.",
    ]
    (ROOT / "nature_physics_breathing_parameter_effects_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metrics, corr, summary = build_metrics()
    metrics.to_csv(SRC / "nphys_breathing_parameter_effects_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_parameter_effects_correlations.csv", index=False)
    summary.to_csv(SRC / "nphys_breathing_parameter_effects_summary.csv", index=False)
    plot_figure(metrics, corr, summary)
    write_report(metrics, corr, summary)
    print(corr[["predictor", "target", "spearman_within_regime", "p_within_regime", "n"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
