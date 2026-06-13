#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_breathing_quality_factor_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
RED = "#B6423E"
VIOLET = "#7E6AAE"
GREEN = "#4F8B67"
COLORS = {"loop": RED, "hazard": GOLD, "amplitude": BLUE, "buffer": GREEN, "tail": VIOLET}
ROUTE_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}


def safe_spearman(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if len(xx) < 3 or np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    return float(spearmanr(xx, yy).statistic)


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


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def route_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    for col in [
        "loop_activation_positive",
        "breathing_hazard_number",
        "response_amplitude",
        "buffer_efficiency",
        "overload_asinh",
        "force_share_top5_edges_hot_minus_cold",
    ]:
        if col not in df.columns:
            raise KeyError(col)
    df["log_hazard"] = np.log10(df["breathing_hazard_number"].clip(lower=0.0) + 1e-12)
    df["loop_z"] = df.groupby("regime_id")["loop_activation_positive"].transform(zscore)
    df["hazard_z"] = df.groupby("regime_id")["log_hazard"].transform(zscore)
    df["amplitude_z"] = df.groupby("regime_id")["response_amplitude"].transform(zscore)
    df["buffer_z"] = df.groupby("regime_id")["buffer_efficiency"].transform(zscore)
    df["tail_z"] = df.groupby("regime_id")["force_share_top5_edges_hot_minus_cold"].transform(zscore)
    df["overload_z"] = df.groupby("regime_id")["overload_asinh"].transform(zscore)
    return df


def lag_table(df: pd.DataFrame, max_lag: int = 6, n_perm: int = 2500, seed: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    predictors = {
        "loop": "loop_z",
        "hazard": "hazard_z",
        "amplitude": "amplitude_z",
        "buffer": "buffer_z",
        "tail": "tail_z",
    }
    rows: list[dict[str, float | int | str]] = []
    null_rows: list[dict[str, float | int | str]] = []
    for name, col in predictors.items():
        for lag in range(max_lag + 1):
            pairs = []
            route_rhos = []
            for rid, g in df.groupby("regime_id", sort=True):
                g = g.sort_values("cycle")
                x = g[col].iloc[: len(g) - lag].to_numpy(float) if lag else g[col].to_numpy(float)
                y = g["overload_z"].iloc[lag:].to_numpy(float) if lag else g["overload_z"].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 4:
                    pairs.append(pd.DataFrame({"route": rid, "x": x[ok], "y": y[ok]}))
                    route_rhos.append(safe_spearman(x[ok], y[ok]))
            d = pd.concat(pairs, ignore_index=True)
            obs = safe_spearman(d["x"].to_numpy(float), d["y"].to_numpy(float))
            null = []
            for p in range(n_perm):
                shifted = []
                for _, g in d.groupby("route", sort=True):
                    vals = g["x"].to_numpy(float)
                    shift = int(rng.integers(0, len(vals)))
                    shifted.extend(np.roll(vals, shift))
                rho = safe_spearman(shifted, d["y"].to_numpy(float))
                null.append(rho)
                null_rows.append({"predictor": name, "lag_cycles": lag, "permutation": p, "null_spearman": rho, "observed_spearman": obs})
            null_arr = np.asarray(null)
            null_finite = null_arr[np.isfinite(null_arr)]
            pval = (np.sum(np.abs(null_finite) >= abs(obs)) + 1.0) / (len(null_finite) + 1.0)
            rows.append(
                {
                    "predictor": name,
                    "lag_cycles": lag,
                    "n": int(len(d)),
                    "spearman": obs,
                    "route_mean_spearman": float(np.nanmean(route_rhos)),
                    "route_min_spearman": float(np.nanmin(route_rhos)),
                    "route_max_spearman": float(np.nanmax(route_rhos)),
                    "circular_shift_p": float(pval),
                    "null_q025": float(np.quantile(null_finite, 0.025)),
                    "null_q975": float(np.quantile(null_finite, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def phase_matched_null(df: pd.DataFrame, max_lag: int = 6, n_perm: int = 2500, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    predictors = {
        "loop": "loop_z",
        "hazard": "hazard_z",
        "amplitude": "amplitude_z",
        "buffer": "buffer_z",
        "tail": "tail_z",
    }
    rows: list[dict[str, float | int | str]] = []
    for name, col in predictors.items():
        for lag in range(max_lag + 1):
            blocks = []
            for rid, g in df.groupby("regime_id", sort=True):
                g = g.sort_values("cycle")
                x = g[col].iloc[: len(g) - lag].to_numpy(float) if lag else g[col].to_numpy(float)
                y = g["overload_z"].iloc[lag:].to_numpy(float) if lag else g["overload_z"].to_numpy(float)
                cycle = g["cycle"].iloc[: len(g) - lag].to_numpy(int) if lag else g["cycle"].to_numpy(int)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 4:
                    blocks.append(pd.DataFrame({"route": rid, "cycle": cycle[ok], "parity": cycle[ok] % 2, "x": x[ok], "y": y[ok]}))
            d = pd.concat(blocks, ignore_index=True)
            obs = safe_spearman(d["x"].to_numpy(float), d["y"].to_numpy(float))
            null = []
            for _ in range(n_perm):
                shuffled = d.copy()
                for _, idx in shuffled.groupby(["route", "parity"], sort=True).groups.items():
                    idx_arr = np.asarray(list(idx), dtype=int)
                    shuffled.loc[idx_arr, "x"] = rng.permutation(shuffled.loc[idx_arr, "x"].to_numpy(float))
                null.append(safe_spearman(shuffled["x"].to_numpy(float), shuffled["y"].to_numpy(float)))
            null_arr = np.asarray(null)
            null_finite = null_arr[np.isfinite(null_arr)]
            pval = (np.sum(np.abs(null_finite) >= abs(obs)) + 1.0) / (len(null_finite) + 1.0)
            rows.append(
                {
                    "predictor": name,
                    "lag_cycles": lag,
                    "n": int(len(d)),
                    "observed_spearman": obs,
                    "phase_matched_p": float(pval),
                    "phase_null_q025": float(np.quantile(null_finite, 0.025)),
                    "phase_null_q975": float(np.quantile(null_finite, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def lagged_design(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    rows = []
    base_cols = ["loop_z", "hazard_z", "amplitude_z", "buffer_z", "tail_z"]
    for rid, g in df.groupby("regime_id", sort=True):
        g = g.sort_values("cycle").copy()
        for lag in range(max_lag + 1):
            for col in base_cols:
                g[f"{col}_lag{lag}"] = g[col].shift(lag)
        keep = ["regime_id", "cycle", "overload_z"] + [f"{col}_lag{lag}" for col in base_cols for lag in range(max_lag + 1)]
        rows.append(g[keep])
    return pd.concat(rows, ignore_index=True).dropna()


def distributed_lag_tests(df: pd.DataFrame, max_lag: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = lagged_design(df, max_lag=max_lag)
    models = {
        "loop kernel": [f"loop_z_lag{i}" for i in range(max_lag + 1)],
        "hazard kernel": [f"hazard_z_lag{i}" for i in range(max_lag + 1)],
        "tail kernel": [f"tail_z_lag{i}" for i in range(max_lag + 1)],
        "amplitude-buffer kernel": [f"amplitude_z_lag{i}" for i in range(max_lag + 1)] + [f"buffer_z_lag{i}" for i in range(max_lag + 1)],
        "loop plus hazard kernel": [f"loop_z_lag{i}" for i in range(max_lag + 1)] + [f"hazard_z_lag{i}" for i in range(max_lag + 1)],
    }
    rows = []
    coef_rows = []
    for name, features in models.items():
        y_all = []
        yh_all = []
        base_all = []
        coefs = []
        for rid in sorted(d["regime_id"].unique()):
            train = d[d["regime_id"] != rid].copy()
            test = d[d["regime_id"] == rid].copy()
            scaler = StandardScaler().fit(train[features].to_numpy(float))
            x_train = scaler.transform(train[features].to_numpy(float))
            x_test = scaler.transform(test[features].to_numpy(float))
            y_train = train["overload_z"].to_numpy(float)
            y_test = test["overload_z"].to_numpy(float)
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, y_train)
            yh = model.predict(x_test)
            y_all.extend(y_test)
            yh_all.extend(yh)
            base_all.extend(np.repeat(float(y_train.mean()), len(y_test)))
            for feature, coef in zip(features, model.coef_):
                coefs.append({"model": name, "left_out_route": rid, "feature": feature, "coefficient": float(coef)})
        y = np.asarray(y_all)
        yh = np.asarray(yh_all)
        base = np.asarray(base_all)
        sse = float(np.sum((y - yh) ** 2))
        sse0 = float(np.sum((y - base) ** 2))
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": int(len(y)),
                "r2_vs_training_mean": 1.0 - sse / sse0,
                "spearman_y_yhat": safe_spearman(y, yh),
            }
        )
        coef_rows.extend(coefs)
    coef_df = pd.DataFrame(coef_rows)
    coef_df["lag_cycles"] = coef_df["feature"].str.extract(r"lag(\d+)").astype(int)
    coef_summary = (
        coef_df.groupby(["model", "feature", "lag_cycles"], as_index=False)
        .agg(mean_coefficient=("coefficient", "mean"), sd_coefficient=("coefficient", "std"))
    )
    return pd.DataFrame(rows), coef_summary


def autocorrelation_table(df: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    rows = []
    for col, name in [("loop_z", "loop activation"), ("overload_z", "overload"), ("hazard_z", "hazard")]:
        for lag in range(1, max_lag + 1):
            route_vals = []
            pooled_x = []
            pooled_y = []
            for rid, g in df.groupby("regime_id", sort=True):
                vals = g.sort_values("cycle")[col].to_numpy(float)
                if len(vals) <= lag:
                    continue
                x = vals[:-lag]
                y = vals[lag:]
                route_vals.append(safe_spearman(x, y))
                pooled_x.extend(x)
                pooled_y.extend(y)
            rows.append(
                {
                    "series": name,
                    "lag_cycles": lag,
                    "pooled_spearman": safe_spearman(pooled_x, pooled_y),
                    "route_mean_spearman": float(np.nanmean(route_vals)),
                    "route_min_spearman": float(np.nanmin(route_vals)),
                    "route_max_spearman": float(np.nanmax(route_vals)),
                }
            )
    return pd.DataFrame(rows)


def make_figure(lag: pd.DataFrame, tests: pd.DataFrame, coefs: pd.DataFrame, ac: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    for name, ls, lw, alpha in [
        ("loop", "-", 1.65, 0.98),
        ("hazard", "-", 1.35, 0.92),
        ("tail", (0, (3, 2)), 1.15, 0.82),
        ("buffer", (0, (1, 2)), 1.05, 0.75),
    ]:
        g = lag[lag["predictor"] == name].sort_values("lag_cycles")
        ax.plot(
            g["lag_cycles"],
            g["spearman"],
            color=COLORS[name],
            lw=lw,
            ls=ls,
            marker="o",
            ms=3.4 if name == "loop" else 3.0,
            alpha=alpha,
            label=name,
        )
        last = g.iloc[-1]
        ax.text(last["lag_cycles"] + 0.13, last["spearman"], name, color=COLORS[name], fontsize=6.1, va="center")
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    sig = lag[(lag["predictor"] == "loop") & (lag["circular_shift_p"] < 0.05)]
    ax.scatter(sig["lag_cycles"], sig["spearman"] + 0.055, marker="*", s=18, color=COLORS["loop"], zorder=5)
    ax.set_xlabel("predictor leads overload by k cycles")
    ax.set_ylabel("route-centred Spearman")
    ax.set_title("breathing memory kernel", loc="left", pad=4)
    ax.set_xlim(-0.25, 6.65)
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    show = tests.sort_values("r2_vs_training_mean")
    y = np.arange(len(show))
    colors = []
    for m in show["model"]:
        if "tail" in m:
            colors.append(COLORS["tail"])
        elif "amplitude" in m:
            colors.append(COLORS["buffer"])
        elif "loop" in m:
            colors.append(COLORS["loop"])
        else:
            colors.append(COLORS["hazard"])
    ax.barh(y, show["r2_vs_training_mean"], color=colors, alpha=0.88)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(show["model"], fontsize=6.1)
    ax.set_xlabel(r"leave-route $R^2$")
    ax.set_title("distributed-lag transfer", loc="left", pad=4)
    finish(ax, axis="x")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    kernel = coefs[coefs["model"].isin(["loop kernel", "hazard kernel", "tail kernel"])].copy()
    for model, label in [("loop kernel", "loop"), ("hazard kernel", "hazard"), ("tail kernel", "tail")]:
        g = kernel[kernel["model"] == model].sort_values("lag_cycles")
        ax.errorbar(
            g["lag_cycles"],
            g["mean_coefficient"],
            yerr=g["sd_coefficient"].fillna(0),
            color=COLORS[label],
            marker="o",
            ms=3.2,
            lw=1.2,
            capsize=2,
            label=label,
        )
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("lag in distributed kernel")
    ax.set_ylabel("standardised ridge coefficient")
    ax.set_title("which breaths write forward risk?", loc="left", pad=4)
    ax.legend(loc="upper right", fontsize=6.0)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    for series, color in [("loop activation", RED), ("overload", BLUE), ("hazard", GOLD)]:
        g = ac[ac["series"] == series].sort_values("lag_cycles")
        ax.plot(g["lag_cycles"], g["route_mean_spearman"], color=color, lw=1.2, marker="o", ms=3.0, label=series)
        ax.fill_between(g["lag_cycles"], g["route_min_spearman"], g["route_max_spearman"], color=color, alpha=0.045, lw=0)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("cycle lag")
    ax.set_ylabel("route-mean autocorrelation")
    ax.set_title("finite-memory rhythm", loc="left", pad=4)
    ax.legend(loc="upper right", fontsize=5.9)
    finish(ax)
    panel(ax, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig42_breathing_memory_kernel.{ext}", **kwargs)
    plt.close(fig)


def write_report(lag: pd.DataFrame, tests: pd.DataFrame, coefs: pd.DataFrame, ac: pd.DataFrame, phase_null: pd.DataFrame) -> None:
    best = lag.sort_values(["predictor", "circular_shift_p", "lag_cycles"]).groupby("predictor", as_index=False).first()
    lines = [
        "# Breathing memory-kernel audit",
        "",
        "This audit asks whether the operational breathing variables carry only instantaneous overload information or whether they form a short memory kernel over previous cycles.",
        "",
        "## Lagged route-centred correlations",
        "",
        lag.round(4).to_markdown(index=False),
        "",
        "## Best lag by predictor",
        "",
        best.round(4).to_markdown(index=False),
        "",
        "## Leave-one-route distributed-lag transfer",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Phase-matched null",
        "",
        phase_null.round(4).to_markdown(index=False),
        "",
        "## Kernel coefficients",
        "",
        coefs.round(4).to_markdown(index=False),
        "",
        "## Autocorrelation rhythm",
        "",
        ac.round(4).to_markdown(index=False),
        "",
        "Interpretation boundary: the result is a finite-cycle diagnostic of the measured five-route ensemble. It should be described as a route-conditioned breathing memory kernel, not as a universal frequency law or a critical slowing-down signature.",
    ]
    (ROOT / "nature_physics_breathing_memory_kernel.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare()
    lag, null = lag_table(df)
    phase_null = phase_matched_null(df)
    tests, coefs = distributed_lag_tests(df)
    ac = autocorrelation_table(df)
    lag.to_csv(SRC / "nphys_breathing_memory_kernel_lag_scan.csv", index=False)
    null.to_csv(SRC / "nphys_breathing_memory_kernel_shift_null.csv", index=False)
    phase_null.to_csv(SRC / "nphys_breathing_memory_kernel_phase_matched_null.csv", index=False)
    tests.to_csv(SRC / "nphys_breathing_memory_kernel_transfer_tests.csv", index=False)
    coefs.to_csv(SRC / "nphys_breathing_memory_kernel_coefficients.csv", index=False)
    ac.to_csv(SRC / "nphys_breathing_memory_kernel_autocorrelation.csv", index=False)
    make_figure(lag, tests, coefs, ac)
    write_report(lag, tests, coefs, ac, phase_null)
    print("Wrote breathing memory-kernel products")
    print(lag.sort_values("circular_shift_p").head(10).round(3).to_string(index=False))
    print(phase_null.sort_values("phase_matched_p").head(10).round(3).to_string(index=False))
    print(tests.sort_values("r2_vs_training_mean", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
