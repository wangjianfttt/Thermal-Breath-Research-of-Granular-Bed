#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

LOOP = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"
NONNORMAL = SRC / "nphys_nonreciprocal_transient_map_metrics.csv"
GEOM = SRC / "nphys_geometric_phase_flow_cycle_metrics.csv"

INK = "#242A31"
GRID = "#E7EAEE"
ACCENT = "#B6423E"
MUTED = "#AAB4C0"
COOL = "#3D6B9C"
GOLD = "#D98C3A"


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


def build_dataset() -> pd.DataFrame:
    loop = pd.read_csv(LOOP)
    geom_cols = [
        "tag",
        "regime_id",
        "cycle",
        "absolute_circulation",
        "phase_speed",
        "tangential_fraction",
        "radial_contraction",
    ]
    geom = pd.read_csv(GEOM)[geom_cols]
    non = pd.read_csv(NONNORMAL)[
        [
            "regime_id",
            "spectral_radius",
            "one_step_gain",
            "peak_normalized_gain",
            "antisymmetric_strength",
            "nonnormality",
        ]
    ]
    df = loop.merge(geom, on=["tag", "regime_id", "cycle"], how="left")
    df = df.merge(non, on="regime_id", how="left")
    df["overload_number_asinh"] = np.arcsinh(df["overload_number"] / 2.0)
    df["route_centered_overload_asinh"] = df["overload_number_asinh"] - df.groupby("regime_id")[
        "overload_number_asinh"
    ].transform("mean")
    df["route_centered_overload_number"] = df["overload_number"] - df.groupby("regime_id")["overload_number"].transform("mean")
    return df


MODEL_FEATURES = {
    "force tail": ["dimensionless_top5_number"],
    "loop activation": ["loop_activation"],
    "dimensionless loop": ["dimensionless_loop_number"],
    "loop + non-normal": ["dimensionless_loop_number", "peak_normalized_gain", "antisymmetric_strength"],
    "loop + geometric flow": ["dimensionless_loop_number", "absolute_circulation", "phase_speed", "tangential_fraction"],
    "full audited model": [
        "dimensionless_loop_number",
        "peak_normalized_gain",
        "antisymmetric_strength",
        "absolute_circulation",
        "phase_speed",
        "tangential_fraction",
    ],
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train[target].to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    if len(features) == 1:
        model = LinearRegression().fit(x_train_s, y_train)
    else:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train_s, y_train)
    return model.predict(x_test_s)


def r2_vs_baseline(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(baseline)
    y = y[ok]
    yhat = yhat[ok]
    baseline = baseline[ok]
    sse = np.sum((y - yhat) ** 2)
    sse0 = np.sum((y - baseline) ** 2)
    return float(1 - sse / sse0) if sse0 > 0 else np.nan


def model_tests(df: pd.DataFrame) -> pd.DataFrame:
    target = "overload_number_asinh"
    rows = []
    for name, features in MODEL_FEATURES.items():
        d = df.dropna(subset=[target, *features]).copy()
        # In-sample, mainly for sanity.
        yhat = fit_predict(d, d, features, target)
        y = d[target].to_numpy(float)
        rows.append(
            {
                "model": name,
                "validation": "in_sample",
                "target": target,
                "n": len(d),
                "r2_vs_mean": r2_vs_baseline(y, yhat, np.repeat(y.mean(), len(y))),
            }
        )
        # Leave one route out. This is deliberately harsh for route-conditioned physics.
        y_all = []
        yhat_all = []
        base_all = []
        for rid in sorted(d["regime_id"].unique()):
            train = d[d["regime_id"] != rid]
            test = d[d["regime_id"] == rid]
            y_test = test[target].to_numpy(float)
            yhat_test = fit_predict(train, test, features, target)
            baseline = np.repeat(train[target].mean(), len(test))
            y_all.extend(y_test)
            yhat_all.extend(yhat_test)
            base_all.extend(baseline)
        rows.append(
            {
                "model": name,
                "validation": "leave_one_route_out",
                "target": target,
                "n": len(y_all),
                "r2_vs_training_mean": r2_vs_baseline(np.asarray(y_all), np.asarray(yhat_all), np.asarray(base_all)),
            }
        )
        # Within-route forward split tests later-cycle transfer without asking for cross-route universality.
        y_all = []
        yhat_all = []
        base_all = []
        for rid, g in d.groupby("regime_id", sort=True):
            train = g[g["cycle"] <= 18]
            test = g[g["cycle"] > 18]
            if len(train) < 5 or len(test) < 5:
                continue
            y_test = test[target].to_numpy(float)
            yhat_test = fit_predict(train, test, features, target)
            baseline = np.repeat(train[target].mean(), len(test))
            y_all.extend(y_test)
            yhat_all.extend(yhat_test)
            base_all.extend(baseline)
        rows.append(
            {
                "model": name,
                "validation": "within_route_forward_60_40",
                "target": target,
                "n": len(y_all),
                "r2_vs_training_mean": r2_vs_baseline(np.asarray(y_all), np.asarray(yhat_all), np.asarray(base_all)),
            }
        )
    return pd.DataFrame(rows)


def incremental_tests(df: pd.DataFrame) -> pd.DataFrame:
    target = "overload_number_asinh"
    baseline_features = ["dimensionless_loop_number"]
    additions = {
        "+ top-5% force tail": ["dimensionless_top5_number"],
        "+ non-normal gain": ["peak_normalized_gain", "antisymmetric_strength"],
        "+ geometric flow": ["absolute_circulation", "phase_speed", "tangential_fraction"],
        "+ all auxiliaries": [
            "dimensionless_top5_number",
            "peak_normalized_gain",
            "antisymmetric_strength",
            "absolute_circulation",
            "phase_speed",
            "tangential_fraction",
        ],
    }
    rows = []
    for label, extra in additions.items():
        features0 = baseline_features
        features1 = baseline_features + extra
        d = df.dropna(subset=[target, *features1]).copy()
        y_all = []
        yhat0_all = []
        yhat1_all = []
        base_all = []
        for rid, g in d.groupby("regime_id", sort=True):
            train = g[g["cycle"] <= 18]
            test = g[g["cycle"] > 18]
            if len(train) < 5 or len(test) < 5:
                continue
            y_test = test[target].to_numpy(float)
            yhat0 = fit_predict(train, test, features0, target)
            yhat1 = fit_predict(train, test, features1, target)
            baseline = np.repeat(train[target].mean(), len(test))
            y_all.extend(y_test)
            yhat0_all.extend(yhat0)
            yhat1_all.extend(yhat1)
            base_all.extend(baseline)
        y = np.asarray(y_all)
        base = np.asarray(base_all)
        r20 = r2_vs_baseline(y, np.asarray(yhat0_all), base)
        r21 = r2_vs_baseline(y, np.asarray(yhat1_all), base)
        rows.append(
            {
                "addition": label,
                "validation": "within_route_forward_60_40",
                "n": len(y),
                "r2_loop_only": r20,
                "r2_augmented": r21,
                "delta_r2": r21 - r20,
            }
        )
    return pd.DataFrame(rows)


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("regime_id", sort=True)
        .agg(
            n=("cycle", "count"),
            overload_number_mean=("overload_number", "mean"),
            dimensionless_loop_number_mean=("dimensionless_loop_number", "mean"),
            peak_normalized_gain=("peak_normalized_gain", "first"),
            antisymmetric_strength=("antisymmetric_strength", "first"),
            absolute_circulation_mean=("absolute_circulation", "mean"),
            phase_speed_mean=("phase_speed", "mean"),
        )
        .reset_index()
    )


def draw_figure(tests: pd.DataFrame, inc: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel(ax_a, "a", x=-0.08)
    order = [
        "force tail",
        "loop activation",
        "dimensionless loop",
        "loop + non-normal",
        "loop + geometric flow",
        "full audited model",
    ]
    t = tests[(tests["validation"] == "leave_one_route_out")].set_index("model").loc[order]
    vals = t["r2_vs_training_mean"].to_numpy(float)
    colors = [MUTED, MUTED, ACCENT, GOLD, COOL, "#6F7C8A"]
    y = np.arange(len(order))
    labels = ["force tail", "loop\nactivation", "loop\nnumber", "loop +\nnon-normal", "loop +\ngeometry", "full\naudit"]
    ax_a.barh(y, vals, color=colors, height=0.56)
    ax_a.axvline(0, color="#AEB6C0", lw=0.7)
    for yi, v in zip(y, vals):
        if v >= 0:
            ax_a.text(v + 0.025, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.3)
        else:
            ax_a.text(v + 0.025, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.3, color=INK)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlabel(r"leave-one-route-out $R^2$")
    ax_a.set_xlim(-0.33, 0.92)
    ax_a.set_title("the route-weighted loop is the transfer backbone", loc="left", pad=4)
    finish(ax_a, "x")

    panel(ax_b, "b")
    x = np.arange(len(inc))
    ax_b.bar(x, inc["delta_r2"], color=[ACCENT if v >= 0 else MUTED for v in inc["delta_r2"]], width=0.62)
    ax_b.axhline(0, color="#AEB6C0", lw=0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["force\ntail", "non-\nnormal", "geometry", "all"], rotation=0)
    for xi, v in zip(x, inc["delta_r2"]):
        ax_b.text(xi, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.2)
    ax_b.set_ylabel(r"$\Delta R^2$ after loop number")
    ax_b.set_title("auxiliary metrics do not replace the loop number", loc="left", pad=4)
    finish(ax_b, "y")

    panel(ax_c, "c")
    ax_c.scatter(
        summary["dimensionless_loop_number_mean"],
        summary["overload_number_mean"],
        s=45 + 180 * summary["antisymmetric_strength"],
        color=ACCENT,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    for _, row in summary.iterrows():
        ax_c.text(
            row["dimensionless_loop_number_mean"] + 0.004,
            row["overload_number_mean"],
            row["regime_id"],
            fontsize=6.4,
            va="center",
            color=INK,
        )
    ax_c.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_c.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_c.set_xlabel(r"route-mean loop number $\langle\Psi\rangle$")
    ax_c.set_ylabel(r"route-mean overload $\langle\widehat{\Omega}\rangle$")
    ax_c.set_title("auxiliary physics modulates a loop backbone", loc="left", pad=4)
    finish(ax_c)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig20_mechanism_hierarchy_audit.{ext}", **kwargs)
    plt.close(fig)


def write_report(tests: pd.DataFrame, inc: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Mechanism hierarchy audit",
        "",
        "This audit compares the mechanism variables introduced in the manuscript in one prediction framework. The target is the asinh-transformed overload number, which keeps long-tail positive overloads visible without truncation.",
        "",
        "## Model tests",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Incremental tests after the dimensionless loop number",
        "",
        inc.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Manuscript-safe interpretation",
        "",
        "- The dimensionless loop number remains the primary transferable mechanism coordinate in the leave-one-route-out audit.",
        "- Non-normal gain and geometric phase-flow metrics are useful mechanistic modifiers and boundaries, but they do not replace the route-weighted force-loop coordinate.",
        "- This supports a cleaner narrative: force-loop embedding sets the overload coordinate; route controls set the gain; non-normal and geometric-flow diagnostics explain how a dissipative map can still transiently amplify selected directions.",
    ]
    (ROOT / "nature_physics_mechanism_hierarchy_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = build_dataset()
    tests = model_tests(df)
    inc = incremental_tests(df)
    summary = route_summary(df)
    df.to_csv(SRC / "nphys_mechanism_hierarchy_cycle_metrics.csv", index=False)
    tests.to_csv(SRC / "nphys_mechanism_hierarchy_model_tests.csv", index=False)
    inc.to_csv(SRC / "nphys_mechanism_hierarchy_incremental_tests.csv", index=False)
    summary.to_csv(SRC / "nphys_mechanism_hierarchy_route_summary.csv", index=False)
    draw_figure(tests, inc, summary)
    write_report(tests, inc, summary)
    print("Wrote mechanism hierarchy audit")
    print(tests.round(3).to_string(index=False))
    print(inc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
