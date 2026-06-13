#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_breathing_parameter_effects_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
RED = "#B6423E"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
VIOLET = "#7E6AAE"
NEUTRAL = "#8D99A6"
COLORS = {"R1": BLUE, "R3": GOLD, "R6": RED}
MARKERS = {"R1": "o", "R3": "s", "R6": "^"}


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


def within_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def within_spearman(df: pd.DataFrame, x: str, y: str) -> dict[str, float | int | str]:
    d = df[["regime_id", x, y]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    raw = spearmanr(d[x], d[y]) if len(d) >= 6 else None
    d[x + "_wc"] = within_center(d, x)
    d[y + "_wc"] = within_center(d, y)
    wc = spearmanr(d[x + "_wc"], d[y + "_wc"]) if len(d) >= 6 else None
    return {
        "predictor": x,
        "target": y,
        "n": int(len(d)),
        "spearman_raw": float(raw.statistic) if raw else np.nan,
        "p_raw": float(raw.pvalue) if raw else np.nan,
        "spearman_within_route": float(wc.statistic) if wc else np.nan,
        "p_within_route": float(wc.pvalue) if wc else np.nan,
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train[target].to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    if len(features) == 1:
        model = LinearRegression().fit(x_train, y_train)
    else:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, y_train)
    return model.predict(x_test)


def r2_vs_baseline(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(baseline)
    y = y[ok]
    yhat = yhat[ok]
    baseline = baseline[ok]
    if len(y) == 0:
        return np.nan
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def forward_model_test(df: pd.DataFrame, features: list[str], target: str) -> tuple[float, float, int]:
    y_all: list[float] = []
    yhat_all: list[float] = []
    base_all: list[float] = []
    for _, g in df.groupby("regime_id", sort=True):
        train = g[g["cycle"] <= 18]
        test = g[g["cycle"] > 18]
        if len(train) < 5 or len(test) < 5:
            continue
        y = test[target].to_numpy(float)
        yhat = fit_predict(train, test, features, target)
        baseline = np.repeat(float(train[target].mean()), len(test))
        y_all.extend(y)
        yhat_all.extend(yhat)
        base_all.extend(baseline)
    y = np.asarray(y_all)
    yhat = np.asarray(yhat_all)
    r2 = r2_vs_baseline(y, yhat, np.asarray(base_all))
    rho = float(spearmanr(y, yhat).statistic) if len(y) > 2 else np.nan
    return r2, rho, int(len(y))


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).copy()
    df = df.sort_values(["regime_id", "cycle"])
    df["irreversibility_area"] = df["breathing_triangle_area"].abs()
    df["directed_area"] = df["signed_breathing_area"]
    df["area_per_amplitude"] = df["irreversibility_area"] / (df["breath_amplitude"].abs() + 1e-12)
    df["positive_area"] = df["directed_area"] > 0
    df["overload_number_local"] = df["force_p99_hot_minus_cold"] / (
        df.groupby("regime_id")["force_p99_cold"].transform("median").abs() + 1e-12
    )
    df["overload_asinh"] = np.arcsinh(df["overload_number_local"] / 2.0)
    return df


def build_correlations(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("irreversibility_area", "overload_asinh"),
        ("area_per_amplitude", "overload_asinh"),
        ("loop_cost", "overload_asinh"),
        ("overload_cost", "imprint_efficiency"),
        ("irreversibility_area", "imprint_efficiency"),
        ("breath_irregularity", "overload_asinh"),
        ("area_per_amplitude", "force_h1_birth_force_share_next_cold_minus_current"),
        ("directed_area", "force_h1_birth_force_share_next_cold_minus_current"),
    ]
    return pd.DataFrame([within_spearman(df, x, y) for x, y in pairs])


def build_model_tests(df: pd.DataFrame) -> pd.DataFrame:
    target = "overload_asinh"
    models = {
        "amplitude": ["breath_amplitude"],
        "area": ["irreversibility_area"],
        "loop cost": ["loop_cost"],
        "area + amplitude": ["irreversibility_area", "breath_amplitude"],
        "loop + area": ["loop_cost", "irreversibility_area"],
        "irreversibility budget": ["loop_cost", "irreversibility_area", "imprint_efficiency", "breath_irregularity"],
    }
    rows = []
    for name, features in models.items():
        d = df.dropna(subset=[target, *features]).copy()
        r2, rho, n = forward_model_test(d, features, target)
        rows.append(
            {
                "target": target,
                "model": name,
                "features": ";".join(features),
                "validation": "within_route_forward_60_40",
                "n": n,
                "r2_vs_training_mean": r2,
                "spearman_y_yhat": rho,
            }
        )
    return pd.DataFrame(rows)


def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    def segment(cycle: int) -> str:
        if cycle <= 10:
            return "early"
        if cycle <= 20:
            return "middle"
        return "late"

    d = df.copy()
    d["segment"] = d["cycle"].astype(int).map(segment)
    out = (
        d.groupby(["regime_id", "segment"], observed=True)
        .agg(
            n=("cycle", "count"),
            irreversibility_area=("irreversibility_area", "mean"),
            area_per_amplitude=("area_per_amplitude", "mean"),
            loop_cost=("loop_cost", "mean"),
            overload_cost=("overload_cost", "mean"),
            imprint_efficiency=("imprint_efficiency", "mean"),
            overload_asinh=("overload_asinh", "mean"),
            positive_area_fraction=("positive_area", "mean"),
        )
        .reset_index()
    )
    return out


def make_figure(df: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, seg: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    x = "area_per_amplitude"
    y = "overload_asinh"
    plot = df[[x, y, "regime_id"]].dropna().copy()
    plot[x + "_wc"] = within_center(plot, x)
    plot[y + "_wc"] = within_center(plot, y)
    for rid, g in plot.groupby("regime_id", sort=True):
        ax.scatter(g[x + "_wc"], g[y + "_wc"], s=23, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", lw=0.35, alpha=0.88, label=rid)
    row = corr[(corr["predictor"] == x) & (corr["target"] == y)].iloc[0]
    ax.text(0.05, 0.96, rf"$\rho_{{within}}={row.spearman_within_route:.2f}$" + f"\nP={row.p_within_route:.1e}", transform=ax.transAxes, ha="left", va="top")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("regime-centred area per inhale")
    ax.set_ylabel("regime-centred overload number")
    ax.set_title("irreversibility area is a weak overload proxy")
    ax.legend(ncol=3, loc="lower right", handlelength=1.0, columnspacing=0.75)
    panel(ax, "a")
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    x = "loop_cost"
    y = "overload_asinh"
    plot = df[[x, y, "regime_id"]].dropna().copy()
    plot[x + "_wc"] = within_center(plot, x)
    plot[y + "_wc"] = within_center(plot, y)
    for rid, g in plot.groupby("regime_id", sort=True):
        ax.scatter(g[x + "_wc"], g[y + "_wc"], s=23, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", lw=0.35, alpha=0.88)
    xx = plot[x + "_wc"].to_numpy(float)
    yy = plot[y + "_wc"].to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    b = np.polyfit(xx[ok], yy[ok], 1)
    line_x = np.linspace(xx[ok].min(), xx[ok].max(), 100)
    ax.plot(line_x, b[0] * line_x + b[1], color=INK, lw=0.8)
    row = corr[(corr["predictor"] == x) & (corr["target"] == y)].iloc[0]
    ax.text(0.05, 0.96, rf"$\rho_{{within}}={row.spearman_within_route:.2f}$" + f"\nP={row.p_within_route:.1e}", transform=ax.transAxes, ha="left", va="top")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("regime-centred loop cost")
    ax.set_ylabel("regime-centred overload number")
    ax.set_title("loop cost carries the irreversible loss")
    panel(ax, "b")
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    order = ["amplitude", "area", "loop cost", "area + amplitude", "loop + area", "irreversibility budget"]
    t = tests.set_index("model").loc[order]
    vals = t["r2_vs_training_mean"].to_numpy(float)
    y_pos = np.arange(len(order))
    colors = [NEUTRAL, GOLD, RED, VIOLET, INK, BLUE]
    ax.barh(y_pos, vals, color=colors, height=0.64, zorder=3)
    ax.axvline(0, color="#AEB6C0", lw=0.65)
    ax.set_yticks(y_pos, ["amp.", "area", "loop\ncost", "area+\namp.", "loop+\narea", "budget"])
    ax.set_xlabel(r"within-route forward $R^2$")
    ax.set_title("prediction budget separates area from loop loss")
    panel(ax, "c")
    finish(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    heat = seg.copy()
    cols = ["area_per_amplitude", "loop_cost", "overload_cost", "imprint_efficiency", "positive_area_fraction"]
    for col in cols:
        s = heat[col]
        heat[col] = (s - s.mean()) / s.std(ddof=0)
    heat["row"] = heat["regime_id"] + "-" + heat["segment"].str[0]
    image = heat[cols].to_numpy(float)
    im = ax.imshow(image, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(np.arange(len(heat)), heat["row"])
    ax.set_xticks(np.arange(len(cols)), ["area/\ninhale", "loop\ncost", "overload\ncost", "imprint\neff.", "directed\narea"])
    ax.set_title("route state is an irreversible budget")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(size=2, width=0.5)
    cbar.set_label("z-score", labelpad=2)
    panel(ax, "d")

    out = FIG / "nphys_fig24_irreversibility_budget"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(corr: pd.DataFrame, tests: pd.DataFrame, seg: pd.DataFrame) -> None:
    def r(pred: str, target: str) -> pd.Series:
        return corr[(corr["predictor"] == pred) & (corr["target"] == target)].iloc[0]

    loop = r("loop_cost", "overload_asinh")
    area = r("area_per_amplitude", "overload_asinh")
    imprint = r("overload_cost", "imprint_efficiency")
    t = tests.set_index("model")
    report = ROOT / "nature_physics_irreversibility_budget.md"
    report.write_text(
        "# Irreversibility-budget audit\n\n"
        "This audit treats the measured breathing loop as a non-equilibrium cycle map. The quantities are proxies, not thermodynamic entropy production: `breathing_triangle_area` measures reduced-state hysteresis area, `loop_cost` measures force-loop activation per inhale, and `imprint_efficiency` measures retained next-cold imprint per inhale.\n\n"
        "## Main findings\n\n"
        f"- Regime-centred loop cost correlates with hot overload (`rho={loop.spearman_within_route:.3f}`, `P={loop.p_within_route:.2e}`), whereas area per inhale is weaker (`rho={area.spearman_within_route:.3f}`, `P={area.p_within_route:.2e}`).\n"
        f"- Imprint efficiency is anticorrelated with overload cost (`rho={imprint.spearman_within_route:.3f}`, `P={imprint.p_within_route:.2e}`), supporting the interpretation that efficient exhalation buffers the dangerous hot branch.\n"
        f"- Within-route forward prediction of overload gives R2={t.loc['loop cost','r2_vs_training_mean']:.3f} from loop cost, R2={t.loc['area','r2_vs_training_mean']:.3f} from hysteresis area alone and R2={t.loc['irreversibility budget','r2_vs_training_mean']:.3f} from the combined budget.\n\n"
        "## Manuscript-safe interpretation\n\n"
        "The breathing loop has a measurable irreversibility budget, but total hysteresis area is not the primary overload variable. The damaging branch is the part of the irreversible excursion that is spent on loop activation. This supports a non-equilibrium statistical-physics reading of the bed as a trained, dissipative network while preserving the stronger force-loop mechanism as the central claim.\n\n"
        "## Boundary\n\n"
        "Do not call this an entropy-production measurement. It is a reduced-state hysteresis and loss-budget proxy extracted from DEM cycle coordinates.\n",
        encoding="utf-8",
    )


def main() -> None:
    df = prepare()
    corr = build_correlations(df)
    tests = build_model_tests(df)
    seg = build_segment_summary(df)
    corr.to_csv(SRC / "nphys_irreversibility_budget_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_irreversibility_budget_prediction.csv", index=False)
    seg.to_csv(SRC / "nphys_irreversibility_budget_route_segments.csv", index=False)
    df.to_csv(SRC / "nphys_irreversibility_budget_cycle_metrics.csv", index=False)
    make_figure(df, corr, tests, seg)
    write_report(corr, tests, seg)
    print(corr.to_string(index=False))
    print(tests.to_string(index=False))
    print(seg.to_string(index=False))


if __name__ == "__main__":
    main()
