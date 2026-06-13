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

DELTA = SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv"
ROUTES = SRC / "nphys_route_phase_space_27case.csv"

COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#9E3D34"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
GRID = "#E7EAEE"
INK = "#252A31"
ACCENT = "#B6423E"
NEUTRAL = "#6F7C8A"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 7.3,
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


def panel(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def ols_fit(x: np.ndarray, y: np.ndarray, *, intercept: bool = True) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    X = x[:, None]
    if intercept:
        X = np.column_stack([np.ones(len(x)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def ols_predict(beta: np.ndarray, x: np.ndarray, *, intercept: bool = True) -> np.ndarray:
    X = x[:, None]
    if intercept:
        X = np.column_stack([np.ones(len(x)), X])
    return X @ beta


def r2_against_baseline(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(baseline)
    y = y[ok]
    yhat = yhat[ok]
    baseline = baseline[ok]
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def leave_one_route_out(df: pd.DataFrame, predictor: str, target: str, *, intercept: bool = True) -> dict[str, float | str | int]:
    y_all: list[float] = []
    yhat_all: list[float] = []
    baseline_all: list[float] = []
    for tag, _ in df.groupby("tag", sort=True):
        train = df[df["tag"] != tag]
        test = df[df["tag"] == tag]
        beta = ols_fit(train[predictor].to_numpy(float), train[target].to_numpy(float), intercept=intercept)
        yhat = ols_predict(beta, test[predictor].to_numpy(float), intercept=intercept)
        baseline = np.repeat(float(train[target].mean()), len(test))
        y_all.extend(test[target].to_numpy(float))
        yhat_all.extend(yhat)
        baseline_all.extend(baseline)
    y = np.asarray(y_all)
    yhat = np.asarray(yhat_all)
    baseline = np.asarray(baseline_all)
    return {
        "predictor": predictor,
        "target": target,
        "validation": "leave_one_route_out",
        "n": int(len(y)),
        "r2_vs_training_mean": r2_against_baseline(y, yhat, baseline),
        "spearman_y_yhat": float(spearmanr(y, yhat).statistic),
    }


def build_table() -> pd.DataFrame:
    delta = pd.read_csv(DELTA)
    routes = pd.read_csv(ROUTES)[["tag", "alpha_mult", "friction", "lid_gap_radii", "hot_susceptibility_index", "cold_memory_index", "loss_index"]]
    df = delta.merge(routes, on="tag", how="left")
    cold_ref = df.groupby("tag")["force_p99_cold"].transform("median").abs()
    df["overload_number"] = df["force_p99_hot_minus_cold"] / cold_ref.replace(0, np.nan)
    df["loop_activation"] = df["force_h1_birth_force_share_hot_minus_cold"]
    df["top5_activation"] = df["force_share_top5_edges_hot_minus_cold"]
    df["giant_activation"] = df["giant_fraction_after_top5_edges_hot_minus_cold"]
    df["frictional_expansion"] = df["alpha_mult"] * df["friction"]
    df["boundary_attenuation"] = 1.0 / (1.0 + df["lid_gap_radii"])
    df["dimensionless_loop_number"] = df["frictional_expansion"] * df["boundary_attenuation"] * df["loop_activation"]
    df["dimensionless_top5_number"] = df["frictional_expansion"] * df["boundary_attenuation"] * df["top5_activation"]
    df["loop_number_no_boundary"] = df["frictional_expansion"] * df["loop_activation"]
    df["loop_number_no_friction"] = df["alpha_mult"] * df["loop_activation"]
    df["loop_number_friction_only"] = df["friction"] * df["loop_activation"]
    df["overload_positive"] = df["overload_number"] > 0
    return df


def build_model_tests(df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "loop_activation",
        "top5_activation",
        "dimensionless_loop_number",
        "dimensionless_top5_number",
        "loop_number_no_boundary",
        "loop_number_no_friction",
        "loop_number_friction_only",
    ]
    rows: list[dict[str, float | str | int]] = []
    target = "overload_number"
    for pred in predictors:
        x = df[pred].to_numpy(float)
        y = df[target].to_numpy(float)
        beta = ols_fit(x, y, intercept=True)
        yhat = ols_predict(beta, x, intercept=True)
        baseline = np.repeat(float(np.mean(y)), len(y))
        rows.append(
            {
                "predictor": pred,
                "target": target,
                "validation": "in_sample_linear",
                "n": int(len(df)),
                "slope": float(beta[-1]),
                "intercept": float(beta[0]),
                "r2_vs_mean": r2_against_baseline(y, yhat, baseline),
                "spearman_x_y": float(spearmanr(x, y).statistic),
            }
        )
        rows.append(leave_one_route_out(df, pred, target))
    return pd.DataFrame(rows)


def build_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "alpha_mult",
        "friction",
        "lid_gap_radii",
        "overload_number",
        "force_p99_hot_minus_cold",
        "loop_activation",
        "top5_activation",
        "dimensionless_loop_number",
        "dimensionless_top5_number",
    ]
    summary = df.groupby(["tag", "regime_id"], sort=True)[cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    return summary.reset_index()


def build_figure(df: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.95), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["dimensionless_loop_number"],
            g["overload_number"],
            s=18,
            color=COLORS.get(rid, NEUTRAL),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            label=rid,
        )
    beta = ols_fit(df["dimensionless_loop_number"].to_numpy(float), df["overload_number"].to_numpy(float), intercept=True)
    xx = np.linspace(df["dimensionless_loop_number"].min(), df["dimensionless_loop_number"].max(), 200)
    ax.plot(xx, ols_predict(beta, xx, intercept=True), color=INK, lw=0.9)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    rho = spearmanr(df["dimensionless_loop_number"], df["overload_number"]).statistic
    loro = tests[(tests["predictor"] == "dimensionless_loop_number") & (tests["validation"] == "leave_one_route_out")].iloc[0]
    ax.text(
        0.05,
        0.96,
        rf"$\rho={rho:.2f}$" + "\n" + rf"LOO $R^2={loro['r2_vs_training_mean']:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
    )
    ax.set_xlabel(r"dimensionless loop number $\Psi$")
    ax.set_ylabel(r"overload number $\widehat{\Omega}$")
    ax.set_title("five-route collapse", loc="left", pad=4)
    ax.legend(loc="lower right", ncol=2, fontsize=6.0, handletextpad=0.35, columnspacing=0.65)
    finish(ax)
    panel(ax, "a", x=-0.09)

    ax = fig.add_subplot(gs[0, 1])
    labels = ["loop\nonly", "top-5%\nonly", "loop\nnumber", "top-5%\nnumber"]
    preds = ["loop_activation", "top5_activation", "dimensionless_loop_number", "dimensionless_top5_number"]
    vals = []
    colors = []
    for pred in preds:
        row = tests[(tests["predictor"] == pred) & (tests["validation"] == "leave_one_route_out")].iloc[0]
        vals.append(float(row["r2_vs_training_mean"]))
        colors.append(ACCENT if pred == "dimensionless_loop_number" else NEUTRAL)
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=0.62)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"leave-one-route-out $R^2$")
    ax.set_title("control comparison", loc="left", pad=4)
    finish(ax, "y")
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    route = summary.sort_values("dimensionless_loop_number_mean")
    y = np.arange(len(route))
    ax.errorbar(
        route["dimensionless_loop_number_mean"],
        y,
        xerr=route["dimensionless_loop_number_std"].fillna(0),
        fmt="o",
        color=ACCENT,
        ecolor="#D7A09D",
        ms=4,
        lw=0.8,
    )
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_yticks(y, route["regime_id"])
    ax.set_xlabel(r"route mean $\Psi$")
    ax.set_title("route ordering", loc="left", pad=4)
    finish(ax, "x")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(df["dimensionless_top5_number"], df["overload_number"], s=15, color=NEUTRAL, edgecolor="white", linewidth=0.3, alpha=0.75)
    beta_top = ols_fit(df["dimensionless_top5_number"].to_numpy(float), df["overload_number"].to_numpy(float), intercept=True)
    xx = np.linspace(df["dimensionless_top5_number"].min(), df["dimensionless_top5_number"].max(), 200)
    ax.plot(xx, ols_predict(beta_top, xx, intercept=True), color=NEUTRAL, lw=0.85)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    rho_top = spearmanr(df["dimensionless_top5_number"], df["overload_number"]).statistic
    ax.text(0.05, 0.95, rf"$\rho={rho_top:.2f}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(r"dimensionless top-5% number")
    ax.set_ylabel(r"$\widehat{\Omega}$")
    ax.set_title("tail concentration fails", loc="left", pad=4)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 2])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.plot(g["cycle"], g["dimensionless_loop_number"], color=COLORS.get(rid, NEUTRAL), lw=0.95, alpha=0.9)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Psi$")
    ax.set_title("trained loop number", loc="left", pad=4)
    finish(ax)
    panel(ax, "e")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig16_dimensionless_loop_collapse.{ext}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    main = tests[(tests["predictor"] == "dimensionless_loop_number") & (tests["validation"] == "leave_one_route_out")].iloc[0]
    loop = tests[(tests["predictor"] == "loop_activation") & (tests["validation"] == "leave_one_route_out")].iloc[0]
    top = tests[(tests["predictor"] == "dimensionless_top5_number") & (tests["validation"] == "leave_one_route_out")].iloc[0]
    rho = float(spearmanr(df["dimensionless_loop_number"], df["overload_number"]).statistic)
    rho_top = float(spearmanr(df["dimensionless_top5_number"], df["overload_number"]).statistic)
    lines = [
        "# Dimensionless loop-collapse audit",
        "",
        "## Question",
        "",
        "Can a route-level dimensionless number collapse the hot overload response across the five long-cycle true-force routes?",
        "",
        "## Definition",
        "",
        "The overload number is defined as `Omega_hat = (f99_hot - f99_cold) / median_route(f99_cold)`.",
        "The dimensionless loop number is defined as `Psi = (alpha/alpha0) * mu * Delta L_f / (1 + chi)`, where `chi = h_gap / d`.",
        "",
        "## Main result",
        "",
        f"- Spearman(Psi, Omega_hat) = {rho:.3f} over 150 paired cycle states.",
        f"- Leave-one-route-out R2 for Psi = {main['r2_vs_training_mean']:.3f}.",
        f"- Pure loop activation leave-one-route-out R2 = {loop['r2_vs_training_mean']:.3f}.",
        f"- Dimensionless top-5% force-tail control Spearman = {rho_top:.3f}, leave-one-route-out R2 = {top['r2_vs_training_mean']:.3f}.",
        "",
        "## Conservative interpretation",
        "",
        "The collapse supports the view that hot overload is organised by loop activation weighted by thermal expansion, friction and boundary clearance. It is not a fitted constitutive law: the current dataset has five routes, and the boundary attenuation is a simple nondimensional control rather than a calibrated exponent.",
        "",
        "## Route ordering",
        "",
        summary[["tag", "regime_id", "dimensionless_loop_number_mean", "overload_number_mean", "loop_activation_mean", "top5_activation_mean"]].to_markdown(index=False),
        "",
    ]
    (ROOT / "nature_physics_dimensionless_loop_collapse_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = build_table()
    tests = build_model_tests(df)
    summary = build_route_summary(df)
    df.to_csv(SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv", index=False)
    tests.to_csv(SRC / "nphys_dimensionless_loop_collapse_model_tests.csv", index=False)
    summary.to_csv(SRC / "nphys_dimensionless_loop_collapse_route_summary.csv", index=False)
    build_figure(df, tests, summary)
    write_report(df, tests, summary)
    print("wrote dimensionless loop collapse outputs")
    print(tests)


if __name__ == "__main__":
    main()
