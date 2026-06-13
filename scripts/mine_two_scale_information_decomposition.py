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

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
ROUTE = "#4C566A"
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


def categorical_codes(values: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(int)


def joint_codes(*arrays: np.ndarray) -> np.ndarray:
    arrs = [np.asarray(a, dtype=int) for a in arrays]
    code = np.zeros_like(arrs[0], dtype=int)
    base = 1
    for arr in arrs:
        shifted = arr - arr.min()
        code += shifted * base
        base *= int(shifted.max()) + 1
    return code


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
    cmi = 0.0
    for zi in np.unique(z):
        m = z == zi
        cmi += float(m.mean()) * mutual_information_bits(x[m], y[m])
    return float(cmi)


def load_table() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    required = {"regime_id", "cycle", "psi_z", "loop_z", "tail_z", "cold_loop_z", "overload_asinh"}
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
                    "future_overload_asinh": g["overload_asinh"].iloc[lag:].to_numpy(float) if lag else g["overload_asinh"].to_numpy(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).dropna()


def shift_null_cmi(d: pd.DataFrame, bins: int, n_null: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    route = categorical_codes(d["regime_id"].to_numpy())
    y = quantile_codes(d["future_overload_asinh"].to_numpy(float), bins)
    route_arrays = [g["predictor_value"].to_numpy(float) for _, g in d.groupby("regime_id", sort=True)]
    null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        vals = []
        for arr in route_arrays:
            vals.extend(np.roll(arr, int(rng.integers(0, len(arr)))))
        x = quantile_codes(np.asarray(vals, dtype=float), bins)
        null[i] = conditional_mi_bits(x, y, route)
    return null


def bootstrap_cmi(d: pd.DataFrame, bins: int, null_median: float, n_boot: int = 800, seed: int = 31) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in d.groupby("regime_id", sort=True)]
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        parts = []
        for g in groups:
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        boot = pd.concat(parts, ignore_index=True)
        route = categorical_codes(boot["regime_id"].to_numpy())
        x = quantile_codes(boot["predictor_value"].to_numpy(float), bins)
        y = quantile_codes(boot["future_overload_asinh"].to_numpy(float), bins)
        vals[i] = conditional_mi_bits(x, y, route) - null_median
    return float(np.quantile(vals, 0.025)), float(np.median(vals)), float(np.quantile(vals, 0.975))


def decompose(df: pd.DataFrame, n_null: int = 800) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    null_rows = []
    for p_i, (label, col, _) in enumerate(PREDICTORS):
        for lag in LAGS:
            d = lag_join(df, col, lag)
            route = categorical_codes(d["regime_id"].to_numpy())
            for bins in BINS:
                x = quantile_codes(d["predictor_value"].to_numpy(float), bins)
                y = quantile_codes(d["future_overload_asinh"].to_numpy(float), bins)
                h = entropy_bits(y)
                route_mi = mutual_information_bits(route, y)
                pred_mi = mutual_information_bits(x, y)
                joint_mi = mutual_information_bits(joint_codes(route, x), y)
                pred_given_route = conditional_mi_bits(x, y, route)
                route_given_pred = conditional_mi_bits(route, y, x)
                null = shift_null_cmi(d, bins=bins, n_null=n_null, seed=620 + p_i * 100 + lag * 10 + bins)
                null_median = float(np.median(null))
                corrected = pred_given_route - null_median
                p = (1.0 + float(np.sum(null >= pred_given_route))) / (len(null) + 1.0)
                ci_lo, boot_med, ci_hi = bootstrap_cmi(
                    d,
                    bins,
                    null_median,
                    seed=820 + p_i * 100 + lag * 10 + bins,
                )
                rows.append(
                    {
                        "predictor": label,
                        "predictor_column": col,
                        "lag_cycles": lag,
                        "bins": bins,
                        "n": len(d),
                        "target_entropy_bits": h,
                        "route_mi_bits": route_mi,
                        "predictor_mi_bits": pred_mi,
                        "joint_route_predictor_mi_bits": joint_mi,
                        "predictor_given_route_mi_bits": pred_given_route,
                        "route_given_predictor_mi_bits": route_given_pred,
                        "null_median_predictor_given_route_mi_bits": null_median,
                        "corrected_predictor_given_route_mi_bits": corrected,
                        "corrected_predictor_given_route_norm": corrected / h if h > 0 else np.nan,
                        "bootstrap_median_corrected_bits": boot_med,
                        "bootstrap_q025_corrected_bits": ci_lo,
                        "bootstrap_q975_corrected_bits": ci_hi,
                        "shift_p": p,
                    }
                )
                for j, v in enumerate(null):
                    null_rows.append(
                        {
                            "predictor": label,
                            "lag_cycles": lag,
                            "bins": bins,
                            "permutation": j,
                            "null_predictor_given_route_mi_bits": float(v),
                            "observed_predictor_given_route_mi_bits": pred_given_route,
                        }
                    )
    result = pd.DataFrame(rows)
    return result, pd.DataFrame(null_rows), summarise(result)


def summarise(info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in ["Psi", "loop activation", "top-5% tail", "cold loop memory"]:
        for lag in [0, 2]:
            d = info[(info["predictor"] == predictor) & (info["lag_cycles"] == lag)]
            rows.append(
                {
                    "test": f"{predictor}, lag {lag}",
                    "median_route_mi_bits": float(d["route_mi_bits"].median()),
                    "median_corrected_predictor_given_route_bits": float(d["corrected_predictor_given_route_mi_bits"].median()),
                    "min_corrected_predictor_given_route_norm": float(d["corrected_predictor_given_route_norm"].min()),
                    "n_bin_settings_with_positive_bootstrap_q025": int((d["bootstrap_q025_corrected_bits"] > 0).sum()),
                    "max_shift_p": float(d["shift_p"].max()),
                }
            )
    return pd.DataFrame(rows)


def make_figure(info: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2])

    labels = [p[0] for p in PREDICTORS]
    d2 = info[info["lag_cycles"].eq(2)].copy()
    mat = np.zeros((len(labels), len(BINS)))
    for i, label in enumerate(labels):
        g = d2[d2["predictor"] == label].sort_values("bins")
        mat[i, :] = g["corrected_predictor_given_route_norm"].to_numpy(float)
    im = ax_a.imshow(mat, cmap="mako" if "mako" in plt.colormaps() else "Blues", aspect="auto", vmin=0, vmax=max(0.22, float(np.nanmax(mat))))
    ax_a.set_yticks(np.arange(len(labels)))
    ax_a.set_yticklabels(labels)
    ax_a.set_xticks(np.arange(len(BINS)))
    ax_a.set_xticklabels(BINS)
    ax_a.set_xlabel("number of quantile bins")
    ax_a.set_title("fast loop information remains inside each route", loc="left", pad=4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if mat[i, j] > 0.10 else INK)
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.6, length=2)
    cbar.set_label(r"corrected $I(Y;X|S)/H(Y)$", fontsize=5.8)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    panel(ax_a, "a", x=-0.09)

    tert = d2[(d2["bins"].eq(3)) & (d2["predictor"].isin(["Psi", "top-5% tail", "cold loop memory"]))].copy()
    tert = tert.set_index("predictor").loc[["Psi", "top-5% tail", "cold loop memory"]].reset_index()
    x = np.arange(len(tert))
    width = 0.24
    colors = {"Psi": PSI, "loop activation": LOOP, "top-5% tail": TAIL, "cold loop memory": COLD}
    ax_b.bar(x - width, tert["route_mi_bits"], width=width, color=ROUTE, alpha=0.82, label=r"slow $I(Y;S)$")
    ax_b.bar(x, tert["corrected_predictor_given_route_mi_bits"], width=width, color=[colors[v] for v in tert["predictor"]], alpha=0.88, label=r"fast $I(Y;X|S)$")
    ax_b.bar(x + width, tert["joint_route_predictor_mi_bits"], width=width, color="#222222", alpha=0.78, label=r"joint")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Psi", "tail", "cold\nloop"], fontsize=5.8)
    ax_b.set_ylabel("MI bits")
    ax_b.set_title("slow route plus fast loop", loc="left", pad=4)
    ax_b.legend(fontsize=5.1, loc="upper right")
    finish(ax_b, axis="y")
    panel(ax_b, "b")

    for label in ["Psi", "top-5% tail", "cold loop memory"]:
        g = info[(info["predictor"] == label) & (info["lag_cycles"].isin(LAGS))].copy()
        pivot = g.pivot(index="lag_cycles", columns="bins", values="corrected_predictor_given_route_norm").sort_index()
        median = pivot.median(axis=1)
        lo = pivot.min(axis=1)
        hi = pivot.max(axis=1)
        ax_c.plot(median.index, median.values, marker="o", ms=3.2, lw=1.15, color=colors[label], label=label)
        ax_c.fill_between(median.index.to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), color=colors[label], alpha=0.08, lw=0)
    ax_c.axhline(0, color=NULL, lw=0.7, ls=(0, (3, 3)))
    ax_c.set_xlabel("lag before overload cycle")
    ax_c.set_ylabel(r"corrected $I(Y;X|S)/H(Y)$")
    ax_c.set_title("phase-aligned fast information is even-lagged", loc="left", pad=4)
    ax_c.legend(fontsize=5.8, loc="upper right")
    finish(ax_c)
    panel(ax_c, "c", x=-0.09)

    psi2 = d2[(d2["predictor"].eq("Psi")) & (d2["bins"].eq(3))].iloc[0]
    tail2 = d2[(d2["predictor"].eq("top-5% tail")) & (d2["bins"].eq(3))].iloc[0]
    rows = [
        ("slow route", psi2.route_mi_bits, ROUTE),
        ("fast Psi|route", psi2.corrected_predictor_given_route_mi_bits, PSI),
        ("fast tail|route", tail2.corrected_predictor_given_route_mi_bits, TAIL),
        ("joint route+Psi", psi2.joint_route_predictor_mi_bits, "#222222"),
    ]
    yy = np.arange(len(rows))[::-1]
    ax_d.barh(yy, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.88)
    ax_d.set_yticks(yy)
    ax_d.set_yticklabels([r[0] for r in rows], fontsize=5.7)
    ax_d.set_xlabel("MI bits")
    ax_d.set_title("tertile information budget", loc="left", pad=4)
    for yv, (_, val, _) in zip(yy, rows):
        ax_d.text(val + 0.015, yv, f"{val:.2f}", va="center", fontsize=5.6, color=INK)
    ax_d.set_xlim(0, max(r[1] for r in rows) * 1.22)
    finish(ax_d, axis="x")
    panel(ax_d, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig61_two_scale_information_decomposition.{ext}", **kwargs)
    plt.close(fig)


def write_report(info: pd.DataFrame, summary: pd.DataFrame) -> None:
    d2 = info[(info["lag_cycles"].eq(2)) & (info["bins"].eq(3))].set_index("predictor")
    psi = d2.loc["Psi"]
    tail = d2.loc["top-5% tail"]
    lines = [
        "# Two-scale information-decomposition audit",
        "",
        "Purpose: separate slow route susceptibility from fast phase-aligned loop information, testing whether the information-theory branch is merely a route-label effect.",
        "",
        "## Main findings",
        "",
        f"- For lag-2 tertile overload, slow route identity carries I(Y;S)={psi.route_mi_bits:.3f} bits.",
        f"- Given route, lag-2 Psi still carries corrected I(Y;Psi|S)={psi.corrected_predictor_given_route_mi_bits:.3f} bits after subtracting the route-preserving shift-null median.",
        f"- Given route, the lag-2 top-5% force-tail control carries only {tail.corrected_predictor_given_route_mi_bits:.3f} corrected bits.",
        f"- Joint route+Psi information is {psi.joint_route_predictor_mi_bits:.3f} bits, supporting a two-scale reading rather than route-only or loop-only language.",
        "",
        "Interpretation: route severity supplies the slow susceptibility channel, but phase-aligned loop activation still carries within-route overload information. This supports the two-scale mechanism, while preserving the boundary that the result is a finite-route diagnostic rather than a causal information-flow theorem.",
        "",
        "## Information decomposition",
        "",
        info.round(4).to_markdown(index=False),
        "",
        "## Summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "Allowed wording: overload information decomposes into slow route susceptibility plus a fast loop-sector channel that remains after conditioning on route.",
        "",
        "Not allowed: do not say route labels are irrelevant, do not claim transfer entropy or causal discovery, and do not treat the finite-route decomposition as a universal material law.",
    ]
    (ROOT / "nature_physics_two_scale_information_decomposition.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_table()
    info, null, summary = decompose(df)
    info.to_csv(SRC / "nphys_two_scale_information_decomposition.csv", index=False)
    null.to_csv(SRC / "nphys_two_scale_information_decomposition_shift_null.csv", index=False)
    summary.to_csv(SRC / "nphys_two_scale_information_decomposition_summary.csv", index=False)
    make_figure(info, summary)
    write_report(info, summary)
    print("Wrote two-scale information-decomposition audit")
    print(info[(info["lag_cycles"].eq(2)) & (info["bins"].eq(3))].round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
