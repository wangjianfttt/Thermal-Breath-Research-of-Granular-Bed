#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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

INK = "#242A31"
GRID = "#E8EBEF"
BLUE = "#3D6B9C"
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


def zscore(a: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return arr * 0
    return (arr - np.nanmean(arr)) / std


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle = pd.read_csv(SRC / "nphys_two_scale_response_collapse_cycle_metrics.csv").copy()
    route = pd.read_csv(SRC / "nphys_two_scale_response_collapse_route_kernels.csv").copy()
    cycle["target"] = cycle["overload_asinh"]
    cycle["loop_number"] = cycle["dimensionless_loop_number"]
    cycle["tail_number"] = cycle["dimensionless_top5_number"]
    cycle["cycle_z"] = zscore(cycle["cycle"])
    return cycle, route


def route_cycle_design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [np.ones(len(df)), df["cycle_z"].to_numpy(float)]
    for rid in sorted(df["regime_id"].unique())[1:]:
        parts.append((df["regime_id"] == rid).astype(float).to_numpy())
    for col in cols:
        parts.append(zscore(df[col]))
    return np.column_stack(parts)


def ols_r2(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    return beta, pred, float(r2_score(y, pred))


def residual_after_controls(df: pd.DataFrame, value: str, controls: list[str]) -> np.ndarray:
    X = route_cycle_design(df, controls)
    y = df[value].to_numpy(float)
    beta, pred, _ = ols_r2(X, y)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full(len(df), np.nan)
    resid[ok] = y[ok] - pred
    return resid


def partial_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_resid_base = residual_after_controls(df, "target", [])
    for predictor in ["loop_number", "tail_number", "loop_activation", "top5_activation"]:
        pred_resid = residual_after_controls(df, predictor, [])
        ok = np.isfinite(target_resid_base) & np.isfinite(pred_resid)
        sp = spearmanr(pred_resid[ok], target_resid_base[ok])
        lr = LinearRegression().fit(pred_resid[ok, None], target_resid_base[ok])
        rows.append(
            {
                "predictor": predictor,
                "controls": "route_fixed_effects+cycle",
                "n": int(ok.sum()),
                "partial_spearman": float(sp.statistic),
                "p_value": float(sp.pvalue),
                "partial_slope": float(lr.coef_[0]),
                "partial_intercept": float(lr.intercept_),
                "partial_r2": float(lr.score(pred_resid[ok, None], target_resid_base[ok])),
            }
        )
    return pd.DataFrame(rows)


def permutation_gain_table(df: pd.DataFrame, n_perm: int = 4000, seed: int = 426) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    null_rows = []
    y = df["target"].to_numpy(float)
    base_sets = {
        "FE+cycle": [],
        "FE+cycle+tail": ["tail_number"],
        "FE+cycle+loop": ["loop_number"],
    }
    additions = {
        "loop_number": ("FE+cycle", "loop_number"),
        "tail_number": ("FE+cycle", "tail_number"),
        "loop_after_tail": ("FE+cycle+tail", "loop_number"),
        "tail_after_loop": ("FE+cycle+loop", "tail_number"),
    }
    base_r2_cache: dict[str, float] = {}
    for base_name, base_cols in base_sets.items():
        _, _, base_r2_cache[base_name] = ols_r2(route_cycle_design(df, base_cols), y)
    for label, (base_name, add_col) in additions.items():
        base_cols = base_sets[base_name]
        _, _, full_r2 = ols_r2(route_cycle_design(df, base_cols + [add_col]), y)
        observed = full_r2 - base_r2_cache[base_name]
        null = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = df.copy()
            shuffled[add_col] = shuffled.groupby("regime_id")[add_col].transform(lambda s: rng.permutation(s.to_numpy()))
            _, _, r2p = ols_r2(route_cycle_design(shuffled, base_cols + [add_col]), y)
            null[i] = r2p - base_r2_cache[base_name]
            null_rows.append({"test": label, "null_index": i, "delta_r2_null": float(null[i])})
        rows.append(
            {
                "test": label,
                "base_model": base_name,
                "added_variable": add_col,
                "n": len(df),
                "r2_base": float(base_r2_cache[base_name]),
                "r2_full": float(full_r2),
                "delta_r2": float(observed),
                "within_route_permutation_p": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def leave_one_route_models(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "cold memory": ["cold_memory_index"],
        "force tail": ["tail_number"],
        "route severity": ["route_severity"],
        "loop activation": ["loop_activation"],
        "dimensionless loop": ["loop_number"],
        "route + tail": ["route_severity", "tail_number"],
        "route + loop": ["route_severity", "loop_activation"],
        "route + loop number": ["route_severity", "loop_number"],
    }
    y = df["target"].to_numpy(float)
    groups = df["regime_id"].to_numpy()
    logo = LeaveOneGroupOut()
    rows = []
    for name, features in models.items():
        d = df.dropna(subset=["target", *features]).copy()
        y_d = d["target"].to_numpy(float)
        pred = np.full(len(d), np.nan)
        for train, test in logo.split(d[features], y_d, d["regime_id"].to_numpy()):
            X_train = d.iloc[train][features].to_numpy(float)
            X_test = d.iloc[test][features].to_numpy(float)
            scaler = StandardScaler().fit(X_train)
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            model.fit(scaler.transform(X_train), y_d[train])
            pred[test] = model.predict(scaler.transform(X_test))
        sp = spearmanr(y_d, pred)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out_huber",
                "n": len(d),
                "r2_vs_mean": float(r2_score(y_d, pred)),
                "spearman_y_yhat": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
            }
        )
    return pd.DataFrame(rows)


def draw_path_panel(ax: plt.Axes, gain: pd.DataFrame, loo: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "a", x=-0.04)

    def box(x: float, y: float, w: float, h: float, text: str, edge: str, face: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.0, color=INK)

    def arrow(xy1: tuple[float, float], xy2: tuple[float, float], color: str, lw: float = 1.1, text: str | None = None) -> None:
        ax.add_patch(
            FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color, shrinkA=2, shrinkB=2)
        )
        if text:
            ax.text((xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2 + 0.04, text, color=color, fontsize=6.2, ha="center")

    loop_delta = gain.loc[gain["test"] == "loop_after_tail", "delta_r2"].iloc[0]
    tail_delta = gain.loc[gain["test"] == "tail_after_loop", "delta_r2"].iloc[0]
    loop_r2 = loo.loc[loo["model"] == "dimensionless loop", "r2_vs_mean"].iloc[0]

    box(0.05, 0.62, 0.25, 0.13, "slow route\nseverity", BLUE, "#F4F8FC")
    box(0.39, 0.62, 0.25, 0.13, "cycle loop\nactivation", RED, "#FFF4F1")
    box(0.72, 0.62, 0.23, 0.13, "hot\noverload", RED, "#FFF4F1")
    box(0.39, 0.24, 0.25, 0.13, "force-tail\ncontrol", NEUTRAL, "#F8F8F8")
    box(0.05, 0.24, 0.25, 0.13, "cold fabric\ncontrol", NEUTRAL, "#F8F8F8")
    arrow((0.30, 0.685), (0.39, 0.685), GREEN, 1.1)
    arrow((0.64, 0.685), (0.72, 0.685), RED, 1.4)
    arrow((0.53, 0.37), (0.73, 0.61), NEUTRAL, 0.75)
    arrow((0.30, 0.305), (0.72, 0.625), NEUTRAL, 0.65)
    ax.text(0.345, 0.78, "sets gain", color=GREEN, fontsize=6.5, ha="center")
    ax.text(0.67, 0.78, f"LOO $R^2$={loop_r2:.2f}", color=RED, fontsize=6.5, ha="center")
    ax.text(0.42, 0.51, f"loop after tail\n$\\Delta R^2$={loop_delta:.2f}", color=RED, fontsize=6.6, ha="left")
    ax.text(0.66, 0.43, f"tail after loop\n$\\Delta R^2$={tail_delta:.3f}", color=NEUTRAL, fontsize=6.1, ha="center")
    ax.text(0.05, 0.09, "route-fixed effects + cycle\nwithin-route shuffled nulls", fontsize=6.4, color=NEUTRAL)


def build_figure(df: pd.DataFrame, route: pd.DataFrame, partial: pd.DataFrame, gain: pd.DataFrame, loo: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0])
    ax_path = fig.add_subplot(gs[0, 0])
    draw_path_panel(ax_path, gain, loo)

    # Partial residual panels
    for ax, predictor, color, label, xpos in [
        (fig.add_subplot(gs[0, 1]), "loop_number", RED, "b", -0.14),
        (fig.add_subplot(gs[0, 2]), "tail_number", NEUTRAL, "c", -0.14),
    ]:
        panel(ax, label, x=xpos)
        x = residual_after_controls(df, predictor, [])
        y = residual_after_controls(df, "target", [])
        ok = np.isfinite(x) & np.isfinite(y)
        for rid, g in df.loc[ok].groupby("regime_id", sort=True):
            idx = g.index
            ax.scatter(x[idx], y[idx], s=22, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.3, alpha=0.88)
        beta = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(xx, beta[0] * xx + beta[1], color=color, lw=1.1)
        row = partial.loc[partial["predictor"] == predictor].iloc[0]
        ax.text(0.05, 0.92, f"$\\rho_p$={row.partial_spearman:.2f}\n$P$={row.p_value:.1e}", transform=ax.transAxes, fontsize=6.5, va="top", color=color)
        ax.set_xlabel(f"{predictor.replace('_', ' ')} residual")
        ax.set_ylabel("overload residual")
        finish(ax)

    # Model ladder
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "d", x=-0.10)
    order = [
        "cold memory",
        "force tail",
        "route severity",
        "loop activation",
        "dimensionless loop",
        "route + tail",
        "route + loop",
    ]
    table = loo.set_index("model").loc[order].reset_index()
    colors = [NEUTRAL, NEUTRAL, BLUE, RED, RED, GOLD, GREEN]
    y = np.arange(len(table))
    ax.barh(y, table["r2_vs_mean"], color=colors, alpha=0.9)
    ax.axvline(0, color=INK, lw=0.65)
    ax.set_yticks(y, table["model"])
    ax.invert_yaxis()
    ax.set_xlabel("leave-one-route $R^2$")
    ax.set_xlim(min(-1.05, table["r2_vs_mean"].min() - 0.08), 1.0)
    finish(ax, axis="x")

    # Permutation gain
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "e", x=-0.14)
    labels = ["loop_number", "tail_number", "loop_after_tail", "tail_after_loop"]
    sub = gain.set_index("test").loc[labels].reset_index()
    x = np.arange(len(sub))
    ax.bar(x, sub["delta_r2"], color=[RED, NEUTRAL, RED, NEUTRAL], alpha=0.9)
    ax.errorbar(
        x,
        (sub["null_q025"] + sub["null_q975"]) / 2,
        yerr=[(sub["null_q975"] - sub["null_q025"]) / 2],
        fmt="none",
        ecolor=INK,
        lw=0.8,
        capsize=2.0,
    )
    ax.axhline(0, color=INK, lw=0.65)
    ax.set_xticks(x, ["loop", "tail", "loop|tail", "tail|loop"], rotation=25, ha="right")
    ax.set_ylabel("$\\Delta R^2$ over base")
    ax.text(0.43, 0.92, "bars: observed\nwhiskers: shuffled null 95%", transform=ax.transAxes, fontsize=6.2, va="top", color=NEUTRAL)
    finish(ax, axis="y")

    # Route kernel
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "f", x=-0.14)
    for _, r in route.iterrows():
        rid = r["regime_id"]
        ax.errorbar(
            r["route_severity"],
            r["susceptibility_slope"],
            yerr=[[r["susceptibility_slope"] - r["slope_ci_low"]], [r["slope_ci_high"] - r["susceptibility_slope"]]],
            fmt=MARKERS[rid],
            ms=5.3,
            color=COLORS[rid],
            mec="white",
            mew=0.35,
            elinewidth=0.8,
            capsize=2.0,
        )
        ax.text(r["route_severity"] + 0.015, r["susceptibility_slope"], rid, fontsize=6.2, color=COLORS[rid], va="center")
    sp = spearmanr(route["route_severity"], route["susceptibility_slope"])
    ax.text(0.05, 0.92, f"route-order $\\rho$={sp.statistic:.2f}", transform=ax.transAxes, fontsize=6.5, va="top", color=GREEN)
    ax.set_xlabel("slow route severity $S$")
    ax.set_ylabel("loop-to-overload gain $G(S)$")
    finish(ax)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig36_causal_path_audit.{ext}", **kwargs)
    plt.close(fig)


def write_report(partial: pd.DataFrame, gain: pd.DataFrame, loo: pd.DataFrame, route: pd.DataFrame) -> None:
    loop_partial = partial.loc[partial["predictor"] == "loop_number"].iloc[0]
    tail_partial = partial.loc[partial["predictor"] == "tail_number"].iloc[0]
    loop_gain = gain.loc[gain["test"] == "loop_after_tail"].iloc[0]
    tail_gain = gain.loc[gain["test"] == "tail_after_loop"].iloc[0]
    dim_loop = loo.loc[loo["model"] == "dimensionless loop"].iloc[0]
    route_loop = loo.loc[loo["model"] == "route + loop"].iloc[0]
    route_tail = loo.loc[loo["model"] == "route + tail"].iloc[0]
    sp_route = spearmanr(route["route_severity"], route["susceptibility_slope"])
    lines = [
        "# Causal-path audit for memory-induced thermal breathing",
        "",
        "## Question",
        "",
        "Does the hot-overload path require a route-conditioned force-loop coordinate, or can it be explained by cold fabric, route severity or the strongest-force tail alone?",
        "",
        "## Data",
        "",
        "- Five true-force routes, 150 paired hot-minus-cold cycle states.",
        "- Target: asinh-transformed overload response.",
        "- Main tested path: slow route severity sets the gain; cycle-scale loop activation supplies the fast drive.",
        "- Controls: cold memory, force-tail activation, route fixed effects and cycle trend.",
        "",
        "## Main results",
        "",
        f"- After route fixed effects and cycle trend, the dimensionless loop number has partial Spearman rho = {loop_partial.partial_spearman:.3f} (P = {loop_partial.p_value:.2e}) with overload residuals.",
        f"- The corresponding force-tail number is negative, rho = {tail_partial.partial_spearman:.3f} (P = {tail_partial.p_value:.2e}).",
        f"- Adding the loop number after the force-tail control increases R2 by {loop_gain.delta_r2:.3f} under the route-fixed audit (within-route shuffled-null P = {loop_gain.within_route_permutation_p:.4f}).",
        f"- Adding the force-tail control after the loop number changes R2 by {tail_gain.delta_r2:.3f} (within-route shuffled-null P = {tail_gain.within_route_permutation_p:.4f}).",
        f"- In leave-one-route-out Huber prediction, the dimensionless loop model gives R2 = {dim_loop.r2_vs_mean:.3f}; route + loop gives R2 = {route_loop.r2_vs_mean:.3f}; route + tail gives R2 = {route_tail.r2_vs_mean:.3f}.",
        "- The high route + tail score reflects the fact that route severity itself carries slow control information; the discriminating test is therefore the route-fixed permutation audit, where loop after tail remains significant but tail after loop does not.",
        f"- Route severity orders fitted loop-to-overload gain across the five routes with Spearman rho = {sp_route.statistic:.3f} (as an n=5 diagnostic, not a calibrated law).",
        "",
        "## Interpretation",
        "",
        "The audit supports a path decomposition rather than a single scalar susceptibility: route controls define a slow gain, while hot force-loop activation is the cycle-level variable that carries overload. The strongest-force tail is a necessary control but not a substitute for graph-cycle embedding.",
        "",
        "## Boundary",
        "",
        "This is not a randomized causal experiment. It is a route-conditioned path audit on DEM trajectories. The result should be phrased as evidence that the current data require a loop-mediated state coordinate, not as proof of a universal constitutive pathway.",
        "",
        "## Files",
        "",
        "- Figure: `figures/nphys_fig36_causal_path_audit.*`",
        "- Source data: `source_data/nphys_causal_path_partial_residuals.csv`",
        "- Source data: `source_data/nphys_causal_path_permutation_gain.csv`",
        "- Source data: `source_data/nphys_causal_path_permutation_null.csv`",
        "- Source data: `source_data/nphys_causal_path_leave_one_route_models.csv`",
        "- Source data: `source_data/nphys_causal_path_route_kernels.csv`",
    ]
    (ROOT / "nature_physics_causal_path_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    SRC.mkdir(exist_ok=True)
    df, route = load_data()
    partial = partial_residual_table(df)
    gain, null = permutation_gain_table(df)
    loo = leave_one_route_models(df)
    partial.to_csv(SRC / "nphys_causal_path_partial_residuals.csv", index=False)
    gain.to_csv(SRC / "nphys_causal_path_permutation_gain.csv", index=False)
    null.to_csv(SRC / "nphys_causal_path_permutation_null.csv", index=False)
    loo.to_csv(SRC / "nphys_causal_path_leave_one_route_models.csv", index=False)
    route.to_csv(SRC / "nphys_causal_path_route_kernels.csv", index=False)
    build_figure(df, route, partial, gain, loo)
    write_report(partial, gain, loo, route)
    print("Wrote causal-path audit, source data and figures/nphys_fig36_causal_path_audit.*")


if __name__ == "__main__":
    main()
