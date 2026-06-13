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
INFILE = SRC / "nphys_five_route_breath_memory_spectrum_cycle_metrics.csv"

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


def p_text(p: float) -> str:
    return "P<0.001" if p < 0.001 else f"P={p:.3f}"


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def route_centered_spearman(d: pd.DataFrame, x: str, y: str) -> float:
    xx = d[x] - d.groupby("regime_id")[x].transform("mean")
    yy = d[y] - d.groupby("regime_id")[y].transform("mean")
    return safe_spearman(xx.to_numpy(float), yy.to_numpy(float))


def quantile_codes(values: np.ndarray, bins: int = 3) -> np.ndarray:
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
    n = len(x)
    for xi in np.unique(x):
        px = np.mean(x == xi)
        for yi in np.unique(y):
            py = np.mean(y == yi)
            pxy = np.mean((x == xi) & (y == yi))
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
    return float(mi)


def conditional_mi_bits(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=int)
    ok = (x >= 0) & (y >= 0) & (z >= 0)
    x = x[ok]
    y = y[ok]
    z = z[ok]
    if len(x) == 0:
        return np.nan
    total = 0.0
    for zi in np.unique(z):
        m = z == zi
        total += float(m.mean()) * mutual_information_bits(x[m], y[m])
    return float(total)


def load_table() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    required = {"regime_id", "cycle", "psi_z", "loop_z", "tail_z", "cold_loop_z", "overload_z", "rare_overload"}
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
        pred = g[predictor].iloc[: len(g) - lag].to_numpy(float) if lag else g[predictor].to_numpy(float)
        target = g["overload_z"].iloc[lag:].to_numpy(float) if lag else g["overload_z"].to_numpy(float)
        rare = g["rare_overload"].iloc[lag:].to_numpy(int) if lag else g["rare_overload"].to_numpy(int)
        cycles = g["cycle"].iloc[: len(g) - lag].to_numpy(int) if lag else g["cycle"].to_numpy(int)
        rows.append(
            pd.DataFrame(
                {
                    "regime_id": rid,
                    "cycle": cycles,
                    "predictor_value": pred,
                    "future_overload_z": target,
                    "future_rare_overload": rare,
                }
            )
        )
    return pd.concat(rows, ignore_index=True).dropna()


def score_table(d: pd.DataFrame, target: str, compute_rho: bool = True) -> tuple[float, float, float]:
    x = quantile_codes(d["predictor_value"].to_numpy(float), bins=3)
    if target == "future_rare_overload":
        y = d[target].to_numpy(int)
    else:
        y = quantile_codes(d[target].to_numpy(float), bins=3)
    mi = mutual_information_bits(x, y)
    h = entropy_bits(y)
    norm = mi / h if np.isfinite(h) and h > 0 else np.nan
    rho = route_centered_spearman(d.rename(columns={"predictor_value": "x", target: "y"}), "x", "y") if compute_rho else np.nan
    return mi, norm, rho


def circular_shift_null(d: pd.DataFrame, target: str, n_null: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if target == "future_rare_overload":
        y = d[target].to_numpy(int)
    else:
        y = quantile_codes(d[target].to_numpy(float), bins=3)
    route_arrays = [g["predictor_value"].to_numpy(float) for _, g in d.groupby("regime_id", sort=True)]
    null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        vals = []
        for arr in route_arrays:
            vals.extend(np.roll(arr, int(rng.integers(0, len(arr)))))
        null[i] = mutual_information_bits(quantile_codes(np.asarray(vals, dtype=float), bins=3), y)
    return null


def information_spectrum(df: pd.DataFrame, max_lag: int = 6, n_null: int = 2500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    null_rows = []
    seed_base = 120
    for p_i, (label, col, _) in enumerate(PREDICTORS):
        for lag in range(max_lag + 1):
            joined = lag_join(df, col, lag)
            for target, target_label in [("future_overload_z", "overload tertile"), ("future_rare_overload", "rare overload")]:
                mi, norm, rho = score_table(joined, target)
                null = circular_shift_null(joined, target, n_null=n_null, seed=seed_base + p_i * 100 + lag * 10 + (0 if target == "future_overload_z" else 1))
                p_val = (1.0 + float(np.sum(null >= mi))) / (len(null) + 1.0)
                rows.append(
                    {
                        "predictor": label,
                        "predictor_column": col,
                        "target": target_label,
                        "lag_cycles": lag,
                        "n": len(joined),
                        "mi_bits": mi,
                        "normalised_mi": norm,
                        "route_centered_spearman": rho,
                        "circular_shift_p": p_val,
                        "null_q025": float(np.quantile(null, 0.025)),
                        "null_q975": float(np.quantile(null, 0.975)),
                    }
                )
                for j, v in enumerate(null):
                    null_rows.append(
                        {
                            "predictor": label,
                            "target": target_label,
                            "lag_cycles": lag,
                            "permutation": j,
                            "null_mi_bits": float(v),
                            "observed_mi_bits": mi,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def conditional_information(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("Psi | tail", "psi_z", "tail_z"),
        ("tail | Psi", "tail_z", "psi_z"),
        ("loop | tail", "loop_z", "tail_z"),
        ("cold loop | Psi", "cold_loop_z", "psi_z"),
    ]
    for lag in [0, 2]:
        base = {}
        for name, col, _ in PREDICTORS:
            base[col] = lag_join(df, col, lag)["predictor_value"].to_numpy(float)
        target = quantile_codes(lag_join(df, "psi_z", lag)["future_overload_z"].to_numpy(float), bins=3)
        h = entropy_bits(target)
        for label, x_col, z_col in pairs:
            x = quantile_codes(base[x_col], bins=3)
            z = quantile_codes(base[z_col], bins=3)
            cmi = conditional_mi_bits(x, target, z)
            rows.append(
                {
                    "test": label,
                    "lag_cycles": lag,
                    "n": int(len(target)),
                    "conditional_mi_bits": cmi,
                    "conditional_mi_fraction_of_target_entropy": cmi / h if h > 0 else np.nan,
                    "target_entropy_bits": h,
                }
            )
    return pd.DataFrame(rows)


def route_summary(df: pd.DataFrame, spectrum: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("regime_id", sort=True):
        rows.append(
            {
                "regime_id": rid,
                "n_cycles": int(len(g)),
                "mean_overload_z": float(g["overload_z"].mean()),
                "rare_overload_rate": float(g["rare_overload"].mean()),
                "psi_z_std": float(g["psi_z"].std(ddof=0)),
                "tail_z_std": float(g["tail_z"].std(ddof=0)),
                "loop_z_std": float(g["loop_z"].std(ddof=0)),
            }
        )
    best = (
        spectrum[spectrum["target"].eq("overload tertile")]
        .sort_values(["predictor", "circular_shift_p", "lag_cycles"])
        .groupby("predictor", as_index=False)
        .first()[["predictor", "lag_cycles", "mi_bits", "normalised_mi", "circular_shift_p"]]
    )
    best["regime_id"] = "all_routes_best_lag"
    return pd.concat([pd.DataFrame(rows), best], ignore_index=True, sort=False)


def make_figure(spectrum: pd.DataFrame, conditional: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.13, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2])

    d = spectrum[spectrum["target"].eq("overload tertile")].copy()
    labels = [p[0] for p in PREDICTORS]
    mat = np.zeros((len(labels), 7))
    for i, label in enumerate(labels):
        g = d[d["predictor"] == label].sort_values("lag_cycles")
        mat[i, :] = g["normalised_mi"].to_numpy(float)
    im = ax_a.imshow(mat, cmap="mako" if "mako" in plt.colormaps() else "Blues", aspect="auto", vmin=0, vmax=max(0.30, float(np.nanmax(mat))))
    ax_a.set_yticks(np.arange(len(labels)))
    ax_a.set_yticklabels(labels)
    ax_a.set_xticks(np.arange(7))
    ax_a.set_xticklabels(np.arange(7))
    ax_a.set_xlabel("lag before overload cycle")
    ax_a.set_title("predictive information stays in the loop sector", loc="left", pad=4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if mat[i, j] > 0.17 else INK)
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.6, length=2)
    cbar.set_label("normalised MI", fontsize=5.8)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    panel(ax_a, "a", x=-0.09)

    lag2 = d[d["lag_cycles"].eq(2)].set_index("predictor").loc[labels].reset_index()
    y = np.arange(len(lag2))
    colors = [p[2] for p in PREDICTORS]
    ax_b.barh(y, lag2["mi_bits"], color=colors, alpha=0.88)
    for yi, row in zip(y, lag2.itertuples(index=False)):
        ax_b.plot([row.null_q025, row.null_q975], [yi, yi], color=NULL, lw=1.4, zorder=0)
        ax_b.text(row.mi_bits + 0.006, yi, p_text(row.circular_shift_p), va="center", fontsize=5.7, color=INK)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(lag2["predictor"], fontsize=5.9)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("MI bits")
    ax_b.set_title("two-cycle memory", loc="left", pad=4)
    finish(ax_b, axis="x")
    panel(ax_b, "b")

    offsets = {"Psi": -0.035, "loop activation": 0.035, "top-5% tail": 0.0, "cold loop memory": 0.0}
    styles = {"Psi": (0, (2.0, 1.4)), "loop activation": "solid", "top-5% tail": "solid", "cold loop memory": "solid"}
    for label, _, color in PREDICTORS:
        g = d[d["predictor"] == label].sort_values("lag_cycles")
        xplot = g["lag_cycles"].to_numpy(float) + offsets[label]
        ax_c.plot(xplot, g["mi_bits"], marker="o", ms=3.2, lw=1.1, ls=styles[label], color=color, label=label)
        sig = g[g["circular_shift_p"] < 0.05]
        ax_c.scatter(sig["lag_cycles"].to_numpy(float) + offsets[label], sig["mi_bits"] + 0.018, marker="*", s=18, color=color, zorder=4)
    ax_c.set_xlabel("lag before overload cycle")
    ax_c.set_ylabel("MI with overload tertile (bits)")
    ax_c.set_title("phase-aligned information spectrum", loc="left", pad=4)
    ax_c.legend(fontsize=5.7, loc="upper right", ncol=2)
    finish(ax_c)
    panel(ax_c, "c", x=-0.09)

    c = conditional.copy()
    order = ["Psi | tail", "tail | Psi", "loop | tail", "cold loop | Psi"]
    width = 0.36
    x = np.arange(len(order))
    c0 = c[c["lag_cycles"].eq(0)].set_index("test").loc[order]
    c2 = c[c["lag_cycles"].eq(2)].set_index("test").loc[order]
    ax_d.bar(x - width / 2, c0["conditional_mi_bits"], width=width, color="#8D3138", alpha=0.88, label="lag 0")
    ax_d.bar(x + width / 2, c2["conditional_mi_bits"], width=width, color="#D98C3A", alpha=0.88, label="lag 2")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Psi\ngiven tail", "tail\ngiven Psi", "loop\ngiven tail", "cold loop\ngiven Psi"], fontsize=5.8)
    ax_d.set_ylabel("conditional MI (bits)")
    ax_d.set_title("tail control is not the channel", loc="left", pad=4)
    ax_d.legend(fontsize=5.8, loc="upper right")
    finish(ax_d, axis="y")
    panel(ax_d, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig58_loop_predictive_information.{ext}", **kwargs)
    plt.close(fig)


def write_report(spectrum: pd.DataFrame, conditional: pd.DataFrame, route: pd.DataFrame) -> None:
    def row(pred: str, lag: int, target: str = "overload tertile") -> pd.Series:
        return spectrum[(spectrum["predictor"] == pred) & (spectrum["lag_cycles"] == lag) & (spectrum["target"] == target)].iloc[0]

    psi0 = row("Psi", 0)
    psi2 = row("Psi", 2)
    tail2 = row("top-5% tail", 2)
    loop2 = row("loop activation", 2)
    rare2 = row("Psi", 2, target="rare overload")
    c_lag2 = conditional[conditional["lag_cycles"].eq(2)].set_index("test")
    lines = [
        "# Loop-sector predictive-information audit",
        "",
        "Purpose: test whether future overload information is carried by the route-centred loop sector rather than by a force-tail surrogate or route label. The audit uses only existing five-route true-force source data.",
        "",
        "## Main findings",
        "",
        f"- Same-cycle Psi carries {psi0.mi_bits:.3f} bits of overload-tertile information ({psi0.normalised_mi:.1%} of target entropy) with route-preserving circular-shift P={psi0.circular_shift_p:.4f}.",
        f"- Two cycles before the target overload, Psi still carries {psi2.mi_bits:.3f} bits (P={psi2.circular_shift_p:.4f}) and raw loop activation carries {loop2.mi_bits:.3f} bits (P={loop2.circular_shift_p:.4f}); the top-5% force-tail surrogate carries {tail2.mi_bits:.3f} bits (P={tail2.circular_shift_p:.4f}).",
        f"- For route-local rare overload two cycles later, Psi carries {rare2.mi_bits:.3f} bits with P={rare2.circular_shift_p:.4f}.",
        f"- Conditional information at lag 2 is asymmetric: I(overload; Psi | tail) = {c_lag2.loc['Psi | tail'].conditional_mi_bits:.3f} bits, while I(overload; tail | Psi) = {c_lag2.loc['tail | Psi'].conditional_mi_bits:.3f} bits.",
        "",
        "Interpretation: the loop-sector coordinate acts as an information bottleneck for overload in this measured five-route ensemble. This supports the language that breathing writes a finite-cycle loop memory, while preserving the boundary that the result is a binned diagnostic rather than a universal transfer-entropy or causal law.",
        "",
        "## Predictive information spectrum",
        "",
        spectrum.round(4).to_markdown(index=False),
        "",
        "## Conditional information",
        "",
        conditional.round(4).to_markdown(index=False),
        "",
        "## Route and best-lag summary",
        "",
        route.round(4).to_markdown(index=False),
        "",
        "Allowed wording: the route-centred loop sector retains phase-aligned predictive information about future overload beyond the force-tail surrogate.",
        "",
        "Not allowed: do not call this a rigorous transfer entropy, a universal causal channel, a material constant or a route-independent forecasting law.",
    ]
    (ROOT / "nature_physics_loop_predictive_information.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_table()
    spectrum, null = information_spectrum(df)
    conditional = conditional_information(df)
    route = route_summary(df, spectrum)
    spectrum.to_csv(SRC / "nphys_loop_predictive_information_spectrum.csv", index=False)
    null.to_csv(SRC / "nphys_loop_predictive_information_shift_null.csv", index=False)
    conditional.to_csv(SRC / "nphys_loop_predictive_information_conditional.csv", index=False)
    route.to_csv(SRC / "nphys_loop_predictive_information_route_summary.csv", index=False)
    make_figure(spectrum, conditional)
    write_report(spectrum, conditional, route)
    print("Wrote loop predictive-information audit")
    print(
        spectrum[spectrum["target"].eq("overload tertile")]
        .sort_values(["circular_shift_p", "lag_cycles"])
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print(conditional.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
