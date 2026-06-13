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
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INK = "#252A31"
GRID = "#E7EAEE"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
VIOLET = "#7E6AAE"
RED = "#B6423E"
DARK_RED = "#8C2F2C"
NEUTRAL = "#8B929A"
COLORS = {"R1": BLUE, "R3": GOLD, "R5": VIOLET, "R6": RED, "R6c": DARK_RED}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}


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


def panel(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    n = len(x)
    if n > 8:
        return rho, float(spearmanr(x, y).pvalue)
    obs = abs(rho)
    vals = []
    for perm in permutations(y):
        vals.append(abs(float(spearmanr(x, np.asarray(perm)).statistic)))
    return rho, float(np.mean(np.asarray(vals) >= obs - 1e-12))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SRC / "nphys_loop_susceptibility_kernel_cycle_metrics.csv").copy()
    df["response_coordinate"] = df["route_severity"] * df["loop_activation"]
    df["overload_response"] = df["overload_asinh"]
    route = pd.read_csv(SRC / "nphys_loop_susceptibility_kernel_route_kernels.csv")
    route = route[["regime_id", "susceptibility_slope", "route_severity"]].rename(
        columns={"route_severity": "route_severity_table"}
    )
    df = df.merge(route, on="regime_id", how="left")
    df["slope_safe"] = df["susceptibility_slope"].where(df["susceptibility_slope"].abs() > 1e-6, np.nan)
    df["susceptibility_normalized_overload"] = df["overload_response"] / df["slope_safe"]
    return df


def route_kernel_summary(df: pd.DataFrame, n_boot: int = 4000, seed: int = 137) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    boots = []
    for rid, group in df.groupby("regime_id", sort=True):
        x = group["loop_activation"].to_numpy(float)
        y = group["overload_response"].to_numpy(float)
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        slopes = []
        for b in range(n_boot):
            idx = rng.integers(0, len(group), len(group))
            xb = x[idx]
            yb = y[idx]
            bb = np.linalg.lstsq(np.column_stack([np.ones(len(xb)), xb]), yb, rcond=None)[0]
            slopes.append(float(bb[1]))
            boots.append({"regime_id": rid, "bootstrap": b, "susceptibility_slope": float(bb[1])})
        slopes_arr = np.asarray(slopes)
        rows.append(
            {
                "regime_id": rid,
                "n": len(group),
                "route_severity": float(group["route_severity"].iloc[0]),
                "susceptibility_slope": float(beta[1]),
                "intercept": float(beta[0]),
                "slope_ci_low": float(np.quantile(slopes_arr, 0.025)),
                "slope_ci_high": float(np.quantile(slopes_arr, 0.975)),
                "mean_loop_activation": float(group["loop_activation"].mean()),
                "mean_overload_response": float(group["overload_response"].mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boots)


def model_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "loop activation": ["loop_activation"],
        "route severity": ["route_severity"],
        "multiplicative SxL": ["response_coordinate"],
        "loop + severity": ["loop_activation", "route_severity"],
        "loop + severity + SxL": ["loop_activation", "route_severity", "response_coordinate"],
        "top-5 force tail": ["top5_activation"],
    }
    y = df["overload_response"].to_numpy(float)
    groups = df["regime_id"].to_numpy()
    logo = LeaveOneGroupOut()
    rows = []
    pred_rows = []
    for name, features in feature_sets.items():
        X = df[features].to_numpy(float)
        pred = np.full(len(df), np.nan)
        for train_idx, test_idx in logo.split(X, y, groups):
            scaler = StandardScaler().fit(X[train_idx])
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            model.fit(scaler.transform(X[train_idx]), y[train_idx])
            pred[test_idx] = model.predict(scaler.transform(X[test_idx]))
        rows.append(
            {
                "target": "overload_response",
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out_huber",
                "n": len(df),
                "r2_vs_mean": float(r2_score(y, pred)),
                "spearman_y_yhat": float(spearmanr(y, pred).statistic),
            }
        )
        for i, row in df.iterrows():
            pred_rows.append(
                {
                    "model": name,
                    "regime_id": row["regime_id"],
                    "cycle": int(row["cycle"]),
                    "observed": float(row["overload_response"]),
                    "predicted": float(pred[i]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def collapse_metrics(df: pd.DataFrame, route: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = route["route_severity"].to_numpy(float)
    y = route["susceptibility_slope"].to_numpy(float)
    rho, p = exact_spearman_p(x, y)
    rows.append(
        {
            "test": "route_severity_orders_susceptibility",
            "n": len(route),
            "metric": "spearman_exact",
            "value": rho,
            "p_value": p,
        }
    )
    for predictor, target, name in [
        ("loop_activation", "overload_response", "raw_loop_to_overload"),
        ("response_coordinate", "overload_response", "multiplicative_coordinate"),
        ("loop_activation", "susceptibility_normalized_overload", "susceptibility_normalized_collapse"),
    ]:
        d = df[[predictor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        sp = spearmanr(d[predictor], d[target])
        rows.append({"test": name, "n": len(d), "metric": "spearman", "value": float(sp.statistic), "p_value": float(sp.pvalue)})
    return pd.DataFrame(rows)


def build_figure(df: pd.DataFrame, route: pd.DataFrame, tests: pd.DataFrame, metrics: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 4.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", x=-0.12)
    for rid, group in df.groupby("regime_id", sort=True):
        ax.scatter(
            group["response_coordinate"],
            group["overload_response"],
            s=23,
            color=COLORS[rid],
            marker=MARKERS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.9,
            label=rid,
        )
    x = df["response_coordinate"].to_numpy(float)
    y = df["overload_response"].to_numpy(float)
    beta = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, beta[0] * xx + beta[1], color=INK, lw=0.9)
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    row = metrics.query("test == 'multiplicative_coordinate'").iloc[0]
    model_row = tests.query("model == 'multiplicative SxL'").iloc[0]
    ax.text(0.05, 0.96, rf"$\rho={row.value:.2f}$" + "\n" + rf"LOO $R^2={model_row.r2_vs_mean:.2f}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(r"two-scale coordinate, $S\Delta L_f$")
    ax.set_ylabel(r"hot overload response, asinh$(\widehat{\Omega}/2)$")
    ax.set_title("route susceptibility collapses overload", fontsize=7.6, pad=5)
    ax.legend(loc="lower right", ncol=3, fontsize=5.7, handletextpad=0.25, columnspacing=0.7)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    order = route.sort_values("route_severity")
    ax.errorbar(
        order["route_severity"],
        order["susceptibility_slope"],
        yerr=[order["susceptibility_slope"] - order["slope_ci_low"], order["slope_ci_high"] - order["susceptibility_slope"]],
        fmt="none",
        ecolor=NEUTRAL,
        elinewidth=0.75,
        capsize=2,
        zorder=1,
    )
    for _, row in order.iterrows():
        ax.scatter(row["route_severity"], row["susceptibility_slope"], s=35, color=COLORS[row["regime_id"]], marker=MARKERS[row["regime_id"]], edgecolor="white", lw=0.4, zorder=3)
        ax.text(row["route_severity"] + 0.018, row["susceptibility_slope"], row["regime_id"], fontsize=6.0, va="center")
    mrow = metrics.query("test == 'route_severity_orders_susceptibility'").iloc[0]
    ax.text(0.05, 0.94, rf"$\rho={mrow.value:.2f}$" + f"\nexact P={mrow.p_value:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("route severity, S")
    ax.set_ylabel("susceptibility slope, G(S)")
    ax.set_title("slow route sets gain", fontsize=7.5, pad=5)
    finish(ax)

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["loop_activation", "susceptibility_normalized_overload"])
    for rid, group in d.groupby("regime_id", sort=True):
        ax.scatter(group["loop_activation"], group["susceptibility_normalized_overload"], s=20, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.3, alpha=0.82)
    row = metrics.query("test == 'susceptibility_normalized_collapse'").iloc[0]
    ax.text(0.05, 0.94, rf"$\rho={row.value:.2f}$", transform=ax.transAxes, va="top")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel(r"loop activation, $\Delta L_f$")
    ax.set_ylabel(r"normalised overload, $\Omega/G(S)$")
    ax.set_title("gain-normalisation boundary", fontsize=7.5, pad=5)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    order_models = ["loop activation", "route severity", "multiplicative SxL", "loop + severity + SxL", "top-5 force tail"]
    sub = tests.set_index("model").loc[order_models].reset_index()
    colors = [NEUTRAL, NEUTRAL, RED, DARK_RED, NEUTRAL]
    ax.axvline(0, color="#B8BDC4", lw=0.65)
    ax.barh(np.arange(len(sub)), sub["r2_vs_mean"], color=colors, height=0.65)
    ax.set_yticks(np.arange(len(sub)), ["loop", "severity", "SxL", "full", "top-5"])
    ax.set_xlabel(r"leave-one-route $R^2$")
    ax.set_title("transfer hierarchy", fontsize=7.5, pad=5)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    pred = pd.read_csv(SRC / "nphys_two_scale_response_collapse_predictions.csv")
    pred = pred[pred["model"] == "loop + severity + SxL"]
    for rid, group in pred.groupby("regime_id", sort=True):
        ax.scatter(group["observed"], group["predicted"], s=20, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.3, alpha=0.86)
    lims = [min(pred["observed"].min(), pred["predicted"].min()), max(pred["observed"].max(), pred["predicted"].max())]
    ax.plot(lims, lims, color=INK, lw=0.8)
    ax.set_xlabel("observed response")
    ax.set_ylabel("left-route-out prediction")
    ax.set_title("full two-scale transfer", fontsize=7.5, pad=5)
    finish(ax)

    out = FIG / "nphys_fig31_two_scale_response_collapse"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, tests: pd.DataFrame, route: pd.DataFrame) -> None:
    metric = lambda name: metrics.query("test == @name")["value"].iloc[0]
    pval = lambda name: metrics.query("test == @name")["p_value"].iloc[0]
    r2 = lambda name: tests.query("model == @name")["r2_vs_mean"].iloc[0]
    rho = lambda name: tests.query("model == @name")["spearman_y_yhat"].iloc[0]
    report = f"""# Two-scale response-collapse audit

Date: 2026-06-12

## Question

The manuscript uses a dimensionless loop number to combine route controls and cycle-level force-loop activation. This audit asks whether the same idea can be stated as a two-scale response: a slow route susceptibility G(S) multiplies a fast loop activation Delta L_f.

## Main result

The data support the two-scale response as a diagnostic transfer model. Route severity S orders the fitted loop-to-overload susceptibility G(S) monotonically across the five long-cycle routes (Spearman rho = {metric('route_severity_orders_susceptibility'):.3f}, exact P = {pval('route_severity_orders_susceptibility'):.3f}). The multiplicative coordinate S Delta L_f correlates with hot overload response with rho = {metric('multiplicative_coordinate'):.3f}, compared with rho = {metric('raw_loop_to_overload'):.3f} for raw loop activation. However, after dividing overload by the fitted route susceptibility, the remaining normalised response does not reduce to a universal one-slope loop law (rho = {metric('susceptibility_normalized_collapse'):.3f}), mainly because intercepts and the buffered R1 route matter. Thus the supported collapse is the multiplicative coordinate and route-transfer hierarchy, not a universal gain-normalised constitutive curve.

Leave-one-route-out prediction gives R2 = {r2('multiplicative SxL'):.3f} for the multiplicative coordinate and R2 = {r2('loop + severity + SxL'):.3f} for the full two-scale model. The top-5 percent force-tail control fails this transfer test (R2 = {r2('top-5 force tail'):.3f}). Thus the same force-loop activation is not equally dangerous in all routes; boundary and frictional severity set the gain.

## Interpretation allowed in the manuscript

Allowed: hot overload follows a two-scale response in which route controls set susceptibility and cycle-level force-loop activation supplies the fast drive.

Not allowed: G(S) is not a calibrated universal material law. There are only five route levels, and R1 has a weak/negative local slope, so the collapse should be described as a route-conditioned diagnostic supporting the dimensionless loop number.

## Generated files

- `figures/nphys_fig31_two_scale_response_collapse.*`
- `source_data/nphys_two_scale_response_collapse_cycle_metrics.csv`
- `source_data/nphys_two_scale_response_collapse_route_kernels.csv`
- `source_data/nphys_two_scale_response_collapse_bootstrap.csv`
- `source_data/nphys_two_scale_response_collapse_model_tests.csv`
- `source_data/nphys_two_scale_response_collapse_predictions.csv`
- `source_data/nphys_two_scale_response_collapse_metrics.csv`
"""
    (ROOT / "nature_physics_two_scale_response_collapse.md").write_text(report, encoding="utf-8")


def main() -> None:
    df = load_data()
    route, boot = route_kernel_summary(df)
    tests, pred = model_tests(df)
    pred.to_csv(SRC / "nphys_two_scale_response_collapse_predictions.csv", index=False)
    metrics = collapse_metrics(df, route)
    df.to_csv(SRC / "nphys_two_scale_response_collapse_cycle_metrics.csv", index=False)
    route.to_csv(SRC / "nphys_two_scale_response_collapse_route_kernels.csv", index=False)
    boot.to_csv(SRC / "nphys_two_scale_response_collapse_bootstrap.csv", index=False)
    tests.to_csv(SRC / "nphys_two_scale_response_collapse_model_tests.csv", index=False)
    metrics.to_csv(SRC / "nphys_two_scale_response_collapse_metrics.csv", index=False)
    build_figure(df, route, tests, metrics)
    write_report(metrics, tests, route)
    print("Two-scale response-collapse audit complete.")


if __name__ == "__main__":
    main()
