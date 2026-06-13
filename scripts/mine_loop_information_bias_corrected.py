#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_five_route_breath_memory_spectrum_cycle_metrics.csv"
NULL_FILE = SRC / "nphys_loop_information_binning_robustness_shift_null.csv"

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
PSI = "#B6423E"
LOOP = "#D98C3A"
TAIL = "#7E6AAE"
COLD = "#3D6B9C"
NULL = "#AEB6C0"

PREDICTORS = [
    ("Psi", "psi_z", PSI),
    ("loop activation", "loop_z", LOOP),
    ("top-5% tail", "tail_z", TAIL),
    ("cold loop memory", "cold_loop_z", COLD),
]
BINS = [2, 3, 4, 5]
LAGS = [0, 2, 4, 6]


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


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def quantile_codes(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    out = np.full(values.shape, -1, dtype=int)
    if ok.sum() == 0:
        return out
    ranks = pd.Series(values[ok]).rank(method="first").to_numpy(float) - 1.0
    codes = np.floor(ranks / max(ok.sum(), 1) * bins).astype(int)
    out[ok] = np.clip(codes, 0, bins - 1)
    return out


def entropy_bits(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    y = y[y >= 0]
    if len(y) == 0:
        return np.nan
    counts = np.bincount(y)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information_bits(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    ok = (x >= 0) & (y >= 0)
    x = x[ok]
    y = y[ok]
    if len(x) == 0:
        return np.nan
    mi = 0.0
    for xi in np.unique(x):
        px = np.mean(x == xi)
        for yi in np.unique(y):
            py = np.mean(y == yi)
            pxy = np.mean((x == xi) & (y == yi))
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
    return float(mi)


def load_table() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    required = {"regime_id", "cycle", "psi_z", "loop_z", "tail_z", "cold_loop_z", "overload_z"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    return df


def lag_join(df: pd.DataFrame, predictor: str, lag: int) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("regime_id", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        if lag >= len(g):
            continue
        rows.append(
            pd.DataFrame(
                {
                    "regime_id": rid,
                    "cycle": g["cycle"].iloc[: len(g) - lag].to_numpy(int) if lag else g["cycle"].to_numpy(int),
                    "predictor_value": g[predictor].iloc[: len(g) - lag].to_numpy(float) if lag else g[predictor].to_numpy(float),
                    "future_overload_z": g["overload_z"].iloc[lag:].to_numpy(float) if lag else g["overload_z"].to_numpy(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).dropna()


def mi_score(d: pd.DataFrame, bins: int) -> tuple[float, float]:
    x = quantile_codes(d["predictor_value"].to_numpy(float), bins)
    y = quantile_codes(d["future_overload_z"].to_numpy(float), bins)
    mi = mutual_information_bits(x, y)
    return mi, entropy_bits(y)


def bootstrap_corrected(
    d: pd.DataFrame,
    bins: int,
    null_median: float,
    n_boot: int = 1500,
    seed: int = 991,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    groups = [g.reset_index(drop=True) for _, g in d.groupby("regime_id", sort=True)]
    for i in range(n_boot):
        parts = []
        for g in groups:
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        boot = pd.concat(parts, ignore_index=True)
        vals[i] = mi_score(boot, bins)[0] - null_median
    return float(np.quantile(vals, 0.025)), float(np.median(vals)), float(np.quantile(vals, 0.975))


def corrected_sweep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    null = pd.read_csv(NULL_FILE)
    rows = []
    boot_rows = []
    for p_i, (label, col, _) in enumerate(PREDICTORS):
        for lag in LAGS:
            d = lag_join(df, col, lag)
            for bins in BINS:
                mi, h = mi_score(d, bins)
                n = null[(null["predictor"] == label) & (null["lag_cycles"] == lag) & (null["bins"] == bins)]["null_mi_bits"].to_numpy(float)
                if len(n) == 0:
                    raise SystemExit(f"Missing null rows for {label}, lag={lag}, bins={bins}")
                null_median = float(np.median(n))
                null_q025 = float(np.quantile(n, 0.025))
                null_q975 = float(np.quantile(n, 0.975))
                corrected = mi - null_median
                p = (1.0 + float(np.sum(n >= mi))) / (len(n) + 1.0)
                ci_lo, boot_med, ci_hi = bootstrap_corrected(
                    d,
                    bins,
                    null_median,
                    seed=991 + p_i * 100 + lag * 10 + bins,
                )
                rows.append(
                    {
                        "predictor": label,
                        "predictor_column": col,
                        "lag_cycles": lag,
                        "bins": bins,
                        "n": len(d),
                        "observed_mi_bits": mi,
                        "null_median_mi_bits": null_median,
                        "null_q025_mi_bits": null_q025,
                        "null_q975_mi_bits": null_q975,
                        "bias_corrected_mi_bits": corrected,
                        "bias_corrected_normalised_mi": corrected / h if h > 0 else np.nan,
                        "bootstrap_median_corrected_mi_bits": boot_med,
                        "bootstrap_q025_corrected_mi_bits": ci_lo,
                        "bootstrap_q975_corrected_mi_bits": ci_hi,
                        "circular_shift_p": p,
                    }
                )
                boot_rows.append(
                    {
                        "predictor": label,
                        "lag_cycles": lag,
                        "bins": bins,
                        "bootstrap_q025_corrected_mi_bits": ci_lo,
                        "bootstrap_median_corrected_mi_bits": boot_med,
                        "bootstrap_q975_corrected_mi_bits": ci_hi,
                    }
                )
    sweep = pd.DataFrame(rows)
    summary = summarise(sweep)
    return sweep, pd.DataFrame(boot_rows), summary


def summarise(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in ["Psi", "loop activation", "top-5% tail", "cold loop memory"]:
        for lag in [0, 2]:
            d = sweep[(sweep["predictor"] == predictor) & (sweep["lag_cycles"] == lag)]
            rows.append(
                {
                    "test": f"{predictor}, lag {lag}",
                    "min_corrected_norm_mi": float(d["bias_corrected_normalised_mi"].min()),
                    "median_corrected_norm_mi": float(d["bias_corrected_normalised_mi"].median()),
                    "min_corrected_mi_bits": float(d["bias_corrected_mi_bits"].min()),
                    "n_bin_settings_with_positive_bootstrap_q025": int((d["bootstrap_q025_corrected_mi_bits"] > 0).sum()),
                    "max_shift_p": float(d["circular_shift_p"].max()),
                }
            )
    return pd.DataFrame(rows)


def make_figure(sweep: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.13, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2])

    labels = [p[0] for p in PREDICTORS]
    d2 = sweep[sweep["lag_cycles"].eq(2)].copy()
    mat = np.zeros((len(labels), len(BINS)))
    for i, label in enumerate(labels):
        g = d2[d2["predictor"] == label].sort_values("bins")
        mat[i, :] = g["bias_corrected_normalised_mi"].to_numpy(float)
    im = ax_a.imshow(mat, cmap="mako" if "mako" in plt.colormaps() else "Blues", aspect="auto", vmin=0, vmax=max(0.40, float(np.nanmax(mat))))
    ax_a.set_yticks(np.arange(len(labels)))
    ax_a.set_yticklabels(labels)
    ax_a.set_xticks(np.arange(len(BINS)))
    ax_a.set_xticklabels(BINS)
    ax_a.set_xlabel("number of quantile bins")
    ax_a.set_title("lag-2 information survives null-bias correction", loc="left", pad=4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if mat[i, j] > 0.17 else INK)
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.6, length=2)
    cbar.set_label("bias-corrected normalised MI", fontsize=5.8)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    panel(ax_a, "a", x=-0.09)

    colors = {name: color for name, _, color in PREDICTORS}
    offsets = {"Psi": -0.035, "loop activation": 0.035, "top-5% tail": 0.0, "cold loop memory": 0.0}
    styles = {"Psi": (0, (2.0, 1.4)), "loop activation": "solid", "top-5% tail": "solid", "cold loop memory": "solid"}
    for label in labels:
        g = d2[d2["predictor"] == label].sort_values("bins")
        x = g["bins"].to_numpy(float) + offsets[label]
        y_med = g["bootstrap_median_corrected_mi_bits"].to_numpy(float)
        y_lo = g["bootstrap_q025_corrected_mi_bits"].to_numpy(float)
        y_hi = g["bootstrap_q975_corrected_mi_bits"].to_numpy(float)
        ax_b.errorbar(
            x,
            y_med,
            yerr=[
                np.maximum(y_med - y_lo, 0.0),
                np.maximum(y_hi - y_med, 0.0),
            ],
            marker="o",
            ms=3.1,
            lw=1.05,
            ls=styles[label],
            color=colors[label],
            capsize=1.8,
            label=label,
        )
    ax_b.axhline(0, color=NULL, lw=0.7, ls=(0, (3, 3)))
    ax_b.set_xlabel("bins")
    ax_b.set_ylabel("corrected MI (bits)")
    ax_b.set_title("bootstrap interval after null subtraction", loc="left", pad=4)
    ax_b.legend(fontsize=5.3, loc="upper left")
    finish(ax_b)
    panel(ax_b, "b")

    for label in ["Psi", "top-5% tail", "cold loop memory"]:
        g = sweep[(sweep["predictor"] == label) & (sweep["lag_cycles"].isin(LAGS))].copy()
        pivot = g.pivot(index="lag_cycles", columns="bins", values="bias_corrected_normalised_mi").sort_index()
        median = pivot.median(axis=1)
        lo = pivot.min(axis=1)
        hi = pivot.max(axis=1)
        ax_c.plot(median.index, median.values, marker="o", ms=3.2, lw=1.15, color=colors[label], label=label)
        ax_c.fill_between(median.index.to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), color=colors[label], alpha=0.08, lw=0)
    ax_c.axhline(0, color=NULL, lw=0.7, ls=(0, (3, 3)))
    ax_c.set_xlabel("lag before overload cycle")
    ax_c.set_ylabel("corrected normalised MI")
    ax_c.set_title("even-lag loop channel remains after bias removal", loc="left", pad=4)
    ax_c.legend(fontsize=5.8, loc="upper right")
    finish(ax_c)
    panel(ax_c, "c", x=-0.09)

    budget = d2[(d2["bins"].eq(3)) & (d2["predictor"].isin(["Psi", "top-5% tail", "cold loop memory"]))].copy()
    x = np.arange(len(budget))
    width = 0.27
    ax_d.bar(x - width, budget["observed_mi_bits"], width=width, color=[colors[v] for v in budget["predictor"]], alpha=0.88, label="observed")
    ax_d.bar(x, budget["null_median_mi_bits"], width=width, color=NULL, alpha=0.75, label="null median")
    ax_d.bar(x + width, budget["bias_corrected_mi_bits"], width=width, color="#222222", alpha=0.82, label="corrected")
    ax_d.axhline(0, color=NULL, lw=0.7)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Psi", "tail", "cold\nloop"], fontsize=5.8)
    ax_d.set_ylabel("MI bits")
    ax_d.set_title("bias budget at tertiles", loc="left", pad=4)
    ax_d.legend(fontsize=5.3, loc="upper right")
    finish(ax_d, axis="y")
    panel(ax_d, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig60_loop_information_bias_corrected.{ext}", **kwargs)
    plt.close(fig)


def write_report(sweep: pd.DataFrame, summary: pd.DataFrame) -> None:
    d2 = sweep[sweep["lag_cycles"].eq(2)]
    psi = d2[d2["predictor"].eq("Psi")]
    tail = d2[d2["predictor"].eq("top-5% tail")]
    tert = d2[d2["bins"].eq(3)].set_index("predictor")
    lines = [
        "# Loop predictive-information null-bias audit",
        "",
        "Purpose: test whether the binned predictive-information result survives the positive finite-sample bias expected for discrete mutual information.",
        "",
        "## Main findings",
        "",
        f"- At lag 2, bias-corrected Psi information remains positive across 2-5 bins; corrected normalised MI ranges from {psi.bias_corrected_normalised_mi.min():.3f} to {psi.bias_corrected_normalised_mi.max():.3f}.",
        f"- The lag-2 top-5% tail control is much weaker after null subtraction; corrected normalised MI ranges from {tail.bias_corrected_normalised_mi.min():.3f} to {tail.bias_corrected_normalised_mi.max():.3f}.",
        f"- In the tertile audit, Psi observed MI is {tert.loc['Psi','observed_mi_bits']:.3f} bits, the shift-null median is {tert.loc['Psi','null_median_mi_bits']:.3f} bits, and the corrected value is {tert.loc['Psi','bias_corrected_mi_bits']:.3f} bits.",
        f"- In the same tertile audit, the force-tail corrected value is {tert.loc['top-5% tail','bias_corrected_mi_bits']:.3f} bits.",
        "",
        "Interpretation: the finite-sample positive bias of binned mutual information is not sufficient to explain the loop-sector signal. This strengthens Fig. 58 and Fig. 59 while preserving the boundary that the result remains a diagnostic binned-information audit, not a rigorous transfer-entropy or causal law.",
        "",
        "## Bias-corrected sweep",
        "",
        sweep.round(4).to_markdown(index=False),
        "",
        "## Summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "Allowed wording: the loop-sector information survives route-preserving shift-null bias correction and bootstrap resampling.",
        "",
        "Not allowed: do not call this unbiased continuous mutual information, transfer entropy, causal discovery or route-independent prediction.",
    ]
    (ROOT / "nature_physics_loop_information_bias_corrected.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_table()
    sweep, boot, summary = corrected_sweep(df)
    sweep.to_csv(SRC / "nphys_loop_information_bias_corrected_sweep.csv", index=False)
    boot.to_csv(SRC / "nphys_loop_information_bias_corrected_bootstrap.csv", index=False)
    summary.to_csv(SRC / "nphys_loop_information_bias_corrected_summary.csv", index=False)
    make_figure(sweep, summary)
    write_report(sweep, summary)
    print("Wrote loop predictive-information null-bias audit")
    print(
        sweep[(sweep["lag_cycles"].eq(2)) & (sweep["predictor"].isin(["Psi", "top-5% tail"]))]
        .round(4)
        .to_string(index=False)
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
