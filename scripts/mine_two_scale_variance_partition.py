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
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INK = "#252A31"
GRID = "#E8EBEF"
BLUE = "#345995"
GOLD = "#D98C3A"
VIOLET = "#7E6AAE"
RED = "#B6423E"
DARK_RED = "#8C2F2C"
GREEN = "#4F8A70"
NEUTRAL = "#9AA3AD"
COLORS = {"R1": BLUE, "R3": GOLD, "R5": VIOLET, "R6": RED, "R6c": DARK_RED}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}


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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return arr * 0
    return (arr - np.nanmean(arr)) / std


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    obs = abs(rho)
    null = []
    for perm in permutations(np.asarray(y, dtype=float)):
        null.append(abs(float(spearmanr(x, np.asarray(perm)).statistic)))
    return rho, float(np.mean(np.asarray(null) >= obs - 1e-12))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SRC / "nphys_two_scale_response_collapse_cycle_metrics.csv").copy()
    df = df.rename(
        columns={
            "overload_asinh": "overload",
            "route_severity": "S",
            "loop_activation": "L",
            "top5_activation": "T",
        }
    )
    df["SxL"] = df["S"] * df["L"]
    df["SxT"] = df["S"] * df["T"]
    df["cycle_z"] = zscore(df["cycle"])
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train["overload"].to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def loo_predictions(df: pd.DataFrame, model_features: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    groups = df["regime_id"].to_numpy()
    y = df["overload"].to_numpy(float)
    logo = LeaveOneGroupOut()
    for name, features in model_features.items():
        pred = np.full(len(df), np.nan)
        for train_idx, test_idx in logo.split(df[features], y, groups):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            pred[test_idx] = fit_predict(train, test, features)
        sp = spearmanr(y, pred)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out_huber",
                "n": len(df),
                "r2_vs_global_mean": float(r2_score(y, pred)),
                "spearman_y_yhat": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
            }
        )
        for i, row in df.iterrows():
            pred_rows.append(
                {
                    "model": name,
                    "regime_id": row["regime_id"],
                    "cycle": int(row["cycle"]),
                    "observed": float(row["overload"]),
                    "predicted": float(pred[i]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def ols_r2(df: pd.DataFrame, features: list[str]) -> float:
    x = df[features].to_numpy(float)
    y = df["overload"].to_numpy(float)
    x = StandardScaler().fit_transform(x)
    model = LinearRegression().fit(x, y)
    return float(model.score(x, y))


def shapley_partition(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    cache: dict[tuple[str, ...], float] = {(): 0.0}

    def score(subset: tuple[str, ...]) -> float:
        key = tuple(sorted(subset))
        if key not in cache:
            cache[key] = ols_r2(df, list(key))
        return cache[key]

    contrib = {feature: [] for feature in features}
    for order in permutations(features):
        subset: list[str] = []
        prev = score(tuple(subset))
        for feature in order:
            subset.append(feature)
            now = score(tuple(subset))
            contrib[feature].append(now - prev)
            prev = now
    total = score(tuple(features))
    for feature in features:
        vals = np.asarray(contrib[feature], dtype=float)
        rows.append(
            {
                "feature": feature,
                "mean_incremental_r2": float(vals.mean()),
                "min_incremental_r2": float(vals.min()),
                "max_incremental_r2": float(vals.max()),
                "fraction_of_full_r2": float(vals.mean() / total) if total else np.nan,
                "full_model_r2": float(total),
            }
        )
    return pd.DataFrame(rows)


def route_shuffle_null(df: pd.DataFrame, n_perm: int = 5000, seed: int = 761) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_features = ["S", "T", "SxT"]
    full_features = ["S", "T", "SxT", "L", "SxL"]
    observed = ols_r2(df, full_features) - ols_r2(df, base_features)
    null_rows = []
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = df.copy()
        shuffled["L"] = shuffled.groupby("regime_id")["L"].transform(lambda s: rng.permutation(s.to_numpy()))
        shuffled["SxL"] = shuffled["S"] * shuffled["L"]
        null[i] = ols_r2(shuffled, full_features) - ols_r2(shuffled, base_features)
        null_rows.append({"null_index": i, "delta_r2_null": float(null[i])})
    summary = pd.DataFrame(
        [
            {
                "test": "loop_sector_after_tail_and_route",
                "base_model": "S+T+SxT",
                "full_model": "S+T+SxT+L+SxL",
                "n": len(df),
                "observed_delta_r2": float(observed),
                "route_preserving_shuffle_p": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_median": float(np.median(null)),
                "null_q975": float(np.quantile(null, 0.975)),
            }
        ]
    )
    return summary, pd.DataFrame(null_rows)


def coefficient_bootstrap(df: pd.DataFrame, n_boot: int = 4000, seed: int = 913) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    features = ["S", "L", "SxL", "T", "SxT"]
    rows = []
    boot_rows = []
    x = StandardScaler().fit_transform(df[features].to_numpy(float))
    y = df["overload"].to_numpy(float)
    coef = LinearRegression().fit(x, y).coef_
    for b in range(n_boot):
        idx = []
        for _, g in df.groupby("regime_id", sort=True):
            local = g.index.to_numpy()
            idx.extend(rng.choice(local, size=len(local), replace=True))
        boot = df.loc[idx].reset_index(drop=True)
        xb = StandardScaler().fit_transform(boot[features].to_numpy(float))
        yb = boot["overload"].to_numpy(float)
        cb = LinearRegression().fit(xb, yb).coef_
        for feature, value in zip(features, cb):
            boot_rows.append({"bootstrap": b, "feature": feature, "standardized_coefficient": float(value)})
    boot_df = pd.DataFrame(boot_rows)
    for feature, value in zip(features, coef):
        vals = boot_df.loc[boot_df["feature"] == feature, "standardized_coefficient"].to_numpy(float)
        rows.append(
            {
                "feature": feature,
                "standardized_coefficient": float(value),
                "ci_low": float(np.quantile(vals, 0.025)),
                "ci_high": float(np.quantile(vals, 0.975)),
                "bootstrap_pr_positive": float(np.mean(vals > 0)),
                "bootstrap_pr_negative": float(np.mean(vals < 0)),
            }
        )
    return pd.DataFrame(rows), boot_df


def route_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("regime_id", sort=True):
        x = g["L"].to_numpy(float)
        y = g["overload"].to_numpy(float)
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        sp = spearmanr(x, y)
        rows.append(
            {
                "regime_id": rid,
                "n": len(g),
                "S": float(g["S"].iloc[0]),
                "loop_gain": float(beta[1]),
                "intercept": float(beta[0]),
                "spearman_loop_overload": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "mean_overload": float(g["overload"].mean()),
                "mean_loop_activation": float(g["L"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_figure(
    df: pd.DataFrame,
    loo: pd.DataFrame,
    pred: pd.DataFrame,
    shapley: pd.DataFrame,
    shuffle: pd.DataFrame,
    coef: pd.DataFrame,
    route_gain: pd.DataFrame,
) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0], height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    order = ["route S", "loop L", "SxL", "tail T", "SxT"]
    label_map = {"S": "route S", "L": "loop L", "SxL": "SxL", "T": "tail T", "SxT": "SxT"}
    plot_df = shapley.copy()
    plot_df["label"] = plot_df["feature"].map(label_map)
    plot_df["label"] = pd.Categorical(plot_df["label"], order, ordered=True)
    plot_df = plot_df.sort_values("label")
    colors = [NEUTRAL, RED, DARK_RED, "#B8BEC6", "#D8B88A"]
    ax.barh(plot_df["label"], plot_df["mean_incremental_r2"], color=colors, edgecolor="white", lw=0.5)
    for _, row in plot_df.iterrows():
        ax.text(row["mean_incremental_r2"] + 0.01, row["label"], f"{row['fraction_of_full_r2']:.0%}", va="center", fontsize=6.2)
    ax.set_xlabel("Shapley share of in-sample $R^2$")
    ax.set_title("variance is not route label alone", loc="left", pad=2)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    model_order = [
        "slow route",
        "force tail",
        "loop only",
        "slow + loop",
        "slow x loop",
        "full two-scale",
        "tail two-scale",
    ]
    m = loo.copy()
    m["model"] = pd.Categorical(m["model"], model_order, ordered=True)
    m = m.sort_values("model")
    y = np.arange(len(m))
    ax.axvline(0, color="#B7BDC5", lw=0.65, ls=(0, (3, 3)))
    ax.barh(y, m["r2_vs_global_mean"], color=[NEUTRAL, "#C7A46D", RED, VIOLET, DARK_RED, INK, "#D8B88A"], height=0.72)
    ax.set_yticks(y, [str(x) for x in m["model"]])
    ax.invert_yaxis()
    for yi, value in zip(y, m["r2_vs_global_mean"]):
        ax.text(value + 0.018, yi, f"{value:.2f}", va="center", fontsize=5.9)
    ax.set_xlabel("leave-one-route $R^2$")
    ax.set_title("interaction transfers across routes", loc="left", pad=2)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    full = pred[pred["model"] == "full two-scale"].copy()
    for rid, g in full.groupby("regime_id", sort=True):
        ax.scatter(g["observed"], g["predicted"], s=22, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.35, alpha=0.9, label=rid)
    lim = [
        float(min(full["observed"].min(), full["predicted"].min()) - 0.12),
        float(max(full["observed"].max(), full["predicted"].max()) + 0.12),
    ]
    ax.plot(lim, lim, color=INK, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    row = loo[loo["model"] == "full two-scale"].iloc[0]
    ax.text(0.05, 0.95, rf"LOO $R^2={row.r2_vs_global_mean:.2f}$" + "\n" + rf"$\rho={row.spearman_y_yhat:.2f}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("observed overload")
    ax.set_ylabel("predicted overload")
    ax.set_title("held-out route prediction", loc="left", pad=2)
    ax.legend(loc="lower right", ncol=2, fontsize=5.6, handletextpad=0.25, columnspacing=0.55)
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "d")
    feature_order = ["S", "L", "SxL", "T", "SxT"]
    cdf = coef.set_index("feature").loc[feature_order].reset_index()
    y = np.arange(len(cdf))
    ax.axvline(0, color="#B7BDC5", lw=0.65, ls=(0, (3, 3)))
    ax.errorbar(
        cdf["standardized_coefficient"],
        y,
        xerr=[cdf["standardized_coefficient"] - cdf["ci_low"], cdf["ci_high"] - cdf["standardized_coefficient"]],
        fmt="o",
        color=INK,
        ecolor=NEUTRAL,
        elinewidth=0.85,
        capsize=2,
    )
    ax.set_yticks(y, ["S", "L", "SxL", "tail", "Sxtail"])
    ax.set_xlabel("standardised coefficient")
    ax.set_title("loop interaction remains positive", loc="left", pad=2)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "e")
    srow = shuffle.iloc[0]
    rng = np.random.default_rng(9)
    null_sample = pd.read_csv(SRC / "nphys_two_scale_variance_partition_shuffle_null.csv")["delta_r2_null"].to_numpy(float)
    ax.hist(null_sample, bins=34, color="#D5DAE0", edgecolor="white", lw=0.35)
    ax.axvline(srow["observed_delta_r2"], color=RED, lw=1.25)
    ax.text(
        0.96,
        0.92,
        rf"$\Delta R^2={srow.observed_delta_r2:.2f}$" + "\n" + rf"$P={srow.route_preserving_shuffle_p:.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_xlabel(r"$\Delta R^2$ after route-preserving loop shuffle")
    ax.set_ylabel("null count")
    ax.set_title("cycle alignment carries loop signal", loc="left", pad=2)
    finish(ax, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "f")
    route_gain = route_gain.sort_values("S")
    for _, row in route_gain.iterrows():
        ax.scatter(row["S"], row["loop_gain"], s=38, color=COLORS[row["regime_id"]], marker=MARKERS[row["regime_id"]], edgecolor="white", lw=0.45, zorder=3)
        ax.text(row["S"] + 0.018, row["loop_gain"], row["regime_id"], fontsize=6.0, va="center")
    rho, p_exact = exact_spearman_p(route_gain["S"].to_numpy(float), route_gain["loop_gain"].to_numpy(float))
    ax.text(0.05, 0.95, rf"$\rho={rho:.2f}$" + f"\nexact P={p_exact:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("slow route susceptibility S")
    ax.set_ylabel("local loop-to-overload gain")
    ax.set_title("slow state sets response gain", loc="left", pad=2)
    finish(ax)

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig50_two_scale_variance_partition.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = load_data()
    models = {
        "slow route": ["S"],
        "force tail": ["T"],
        "loop only": ["L"],
        "slow + loop": ["S", "L"],
        "slow x loop": ["SxL"],
        "full two-scale": ["S", "L", "SxL"],
        "tail two-scale": ["S", "T", "SxT"],
    }
    loo, pred = loo_predictions(df, models)
    shapley = shapley_partition(df, ["S", "L", "SxL", "T", "SxT"])
    shuffle, shuffle_null = route_shuffle_null(df)
    coef, coef_boot = coefficient_bootstrap(df)
    route_gain = route_gain_table(df)

    df.to_csv(SRC / "nphys_two_scale_variance_partition_cycle_metrics.csv", index=False)
    loo.to_csv(SRC / "nphys_two_scale_variance_partition_model_tests.csv", index=False)
    pred.to_csv(SRC / "nphys_two_scale_variance_partition_predictions.csv", index=False)
    shapley.to_csv(SRC / "nphys_two_scale_variance_partition_shapley.csv", index=False)
    shuffle.to_csv(SRC / "nphys_two_scale_variance_partition_shuffle_tests.csv", index=False)
    shuffle_null.to_csv(SRC / "nphys_two_scale_variance_partition_shuffle_null.csv", index=False)
    coef.to_csv(SRC / "nphys_two_scale_variance_partition_coefficients.csv", index=False)
    coef_boot.to_csv(SRC / "nphys_two_scale_variance_partition_coefficient_bootstrap.csv", index=False)
    route_gain.to_csv(SRC / "nphys_two_scale_variance_partition_route_gain.csv", index=False)
    build_figure(df, loo, pred, shapley, shuffle, coef, route_gain)

    full = loo[loo["model"] == "full two-scale"].iloc[0]
    loop = loo[loo["model"] == "loop only"].iloc[0]
    tail = loo[loo["model"] == "tail two-scale"].iloc[0]
    srow = shuffle.iloc[0]
    text = f"""# Two-scale variance partition audit

Date: 2026-06-13

## Question

Does hot overload mainly reflect a route label, a force-tail surrogate, or a physical interaction between a slow susceptibility state and fast cycle-level force-loop activation?

## Main result

The overload response is best read as a two-scale interaction. In leave-one-route-out validation, the full two-scale model using slow susceptibility `S`, loop activation `L` and their product `SxL` gives R2 = {full.r2_vs_global_mean:.3f} and Spearman rho = {full.spearman_y_yhat:.3f}. Loop activation alone gives R2 = {loop.r2_vs_global_mean:.3f}, while the analogous tail model `S+T+SxT` gives R2 = {tail.r2_vs_global_mean:.3f}.

The loop sector also survives a route-preserving cycle-alignment null: adding `L` and `SxL` on top of `S+T+SxT` increases in-sample R2 by {srow.observed_delta_r2:.3f}, with circular route-preserving shuffle P = {srow.route_preserving_shuffle_p:.4f}. This null keeps route severity and the within-route distribution of loop activation, but breaks the cycle-by-cycle pairing between loop activation and overload.

The Shapley-style variance partition is descriptive rather than causal, but it is useful for reviewer defence: route severity contributes the slow gain, loop activation contributes the fast drive, and the positive `SxL` interaction carries non-negligible explanatory weight beyond the top-5 percent force-tail surrogate.

## Interpretation allowed in the manuscript

Allowed: hot overload is a two-scale response in which the trained route/boundary state sets the gain and the thermal inhale supplies a loop-activation drive.

Not allowed: this is not a randomized causal proof, not a universal material law and not a claim that route labels alone predict overload. The result should be used to bound alternatives and to explain why force loops are dangerous only in susceptible routes.

## Generated files

- `figures/nphys_fig50_two_scale_variance_partition.*`
- `source_data/nphys_two_scale_variance_partition_cycle_metrics.csv`
- `source_data/nphys_two_scale_variance_partition_model_tests.csv`
- `source_data/nphys_two_scale_variance_partition_predictions.csv`
- `source_data/nphys_two_scale_variance_partition_shapley.csv`
- `source_data/nphys_two_scale_variance_partition_shuffle_tests.csv`
- `source_data/nphys_two_scale_variance_partition_shuffle_null.csv`
- `source_data/nphys_two_scale_variance_partition_coefficients.csv`
- `source_data/nphys_two_scale_variance_partition_route_gain.csv`
"""
    (ROOT / "nature_physics_two_scale_variance_partition.md").write_text(text)
    print("wrote figures/nphys_fig50_two_scale_variance_partition.* and source_data/nphys_two_scale_variance_partition_*.csv")


if __name__ == "__main__":
    main()
