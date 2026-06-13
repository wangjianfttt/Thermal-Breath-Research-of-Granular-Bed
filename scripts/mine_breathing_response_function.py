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
INFILE = SRC / "nphys_irreversibility_budget_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
RED = "#B6423E"
VIOLET = "#7E6AAE"
MUTED = "#8D99A6"
COLORS = {"R1": BLUE, "R3": GOLD, "R6": RED}
MARKERS = {"R1": "o", "R3": "s", "R6": "^"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def within_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    amp = df["breath_amplitude"].abs() + 1e-12
    eta = df["exhalation_imprint_norm"] / amp
    loop_positive = np.maximum(df["loop_cost"].to_numpy(float), 0.0)
    loop_activation = amp * loop_positive
    dissipation = df["breathing_triangle_area"].abs() / (amp**2)
    overload_number = df["force_p99_hot_minus_cold"] / (
        df.groupby("regime_id")["force_p99_cold"].transform("median").abs() + 1e-12
    )
    df["response_amplitude"] = amp
    df["response_imprint_efficiency"] = eta
    df["response_loop_cost_positive"] = loop_positive
    df["response_loop_activation"] = loop_activation
    df["response_dissipation_density"] = dissipation
    df["response_hazard_number"] = loop_activation / (eta + 1e-6)
    df["response_quality_number"] = eta / (loop_activation + 1e-6)
    df["response_overload_number"] = overload_number
    df["response_overload_asinh"] = np.arcsinh(overload_number / 2.0)
    df["response_mode"] = pd.cut(
        df["response_hazard_number"],
        bins=[-np.inf, 0.1, 0.6, np.inf],
        labels=["buffered", "lossy", "overload-prone"],
    )
    return df


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


def build_correlations(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("response_amplitude", "response_overload_asinh"),
        ("response_loop_cost_positive", "response_overload_asinh"),
        ("response_imprint_efficiency", "response_overload_asinh"),
        ("response_loop_activation", "response_overload_asinh"),
        ("response_hazard_number", "response_overload_asinh"),
        ("response_quality_number", "response_overload_asinh"),
        ("response_dissipation_density", "response_overload_asinh"),
        ("response_hazard_number", "force_p99_hot_minus_cold"),
    ]
    return pd.DataFrame([within_spearman(df, x, y) for x, y in pairs])


def r2_vs_baseline(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(baseline)
    y = y[ok]
    yhat = yhat[ok]
    baseline = baseline[ok]
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def forward_model(df: pd.DataFrame, features: list[str], target: str) -> tuple[float, float, int]:
    y_all: list[float] = []
    yh_all: list[float] = []
    base_all: list[float] = []
    for _, g in df.groupby("regime_id", sort=True):
        train = g[g["cycle"] <= 18].replace([np.inf, -np.inf], np.nan).dropna(subset=[target, *features])
        test = g[g["cycle"] > 18].replace([np.inf, -np.inf], np.nan).dropna(subset=[target, *features])
        if len(train) < 5 or len(test) < 5:
            continue
        x_train = train[features].to_numpy(float)
        x_test = test[features].to_numpy(float)
        y_train = train[target].to_numpy(float)
        y_test = test[target].to_numpy(float)
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        model = LinearRegression() if len(features) == 1 else RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        y_all.extend(y_test)
        yh_all.extend(pred)
        base_all.extend(np.repeat(float(y_train.mean()), len(y_test)))
    y = np.asarray(y_all)
    yh = np.asarray(yh_all)
    rho = float(spearmanr(y, yh).statistic) if len(y) >= 3 else np.nan
    return r2_vs_baseline(y, yh, np.asarray(base_all)), rho, int(len(y))


def build_model_tests(df: pd.DataFrame) -> pd.DataFrame:
    target = "response_overload_asinh"
    models = {
        "amplitude": ["response_amplitude"],
        "positive loop cost": ["response_loop_cost_positive"],
        "imprint efficiency": ["response_imprint_efficiency"],
        "loop activation": ["response_loop_activation"],
        "hazard number": ["response_hazard_number"],
        "amplitude + loop + imprint": [
            "response_amplitude",
            "response_loop_cost_positive",
            "response_imprint_efficiency",
        ],
    }
    rows = []
    for name, features in models.items():
        r2, rho, n = forward_model(df, features, target)
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


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    def seg(cycle: int) -> str:
        if cycle <= 10:
            return "early"
        if cycle <= 20:
            return "middle"
        return "late"

    d = df.copy()
    d["segment"] = d["cycle"].astype(int).map(seg)
    return (
        d.groupby(["regime_id", "segment", "response_mode"], observed=True)
        .agg(
            n=("cycle", "count"),
            amplitude=("response_amplitude", "mean"),
            imprint_efficiency=("response_imprint_efficiency", "mean"),
            positive_loop_activation=("response_loop_activation", "mean"),
            dissipation_density=("response_dissipation_density", "mean"),
            hazard_number=("response_hazard_number", "mean"),
            quality_number=("response_quality_number", "mean"),
            overload_asinh=("response_overload_asinh", "mean"),
            hot_overload=("force_p99_hot_minus_cold", "mean"),
        )
        .reset_index()
    )


def plot_figure(df: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.25, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    x = "response_loop_activation"
    y = "response_overload_asinh"
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g[x],
            g[y],
            s=22,
            marker=MARKERS[rid],
            color=COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.86,
            label=rid,
        )
    row = corr[(corr["predictor"] == x) & (corr["target"] == y)].iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman_raw:.2f}$" + f"\nP={row.p_raw:.1e}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xscale("log")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("positive loop activation, A K+")
    ax.set_ylabel("overload number, asinh scaled")
    ax.set_title("loop activation orders the hot branch", loc="left", pad=2)
    ax.legend(ncol=3, loc="lower right", handlelength=1.0, columnspacing=0.75)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    x = "response_hazard_number"
    y = "response_overload_asinh"
    d = df[["regime_id", x, y]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d[x + "_wc"] = within_center(d, x)
    d[y + "_wc"] = within_center(d, y)
    for rid, g in d.groupby("regime_id", sort=True):
        ax.scatter(
            g[x + "_wc"],
            g[y + "_wc"],
            s=22,
            marker=MARKERS[rid],
            color=COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.86,
        )
    xx = d[x + "_wc"].to_numpy(float)
    yy = d[y + "_wc"].to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    b = np.polyfit(xx[ok], yy[ok], 1)
    line_x = np.linspace(xx[ok].min(), xx[ok].max(), 100)
    ax.plot(line_x, b[0] * line_x + b[1], color=INK, lw=0.8)
    row = corr[(corr["predictor"] == x) & (corr["target"] == y)].iloc[0]
    ax.text(
        0.05,
        0.95,
        rf"$\rho_{{within}}={row.spearman_within_route:.2f}$" + f"\nP={row.p_within_route:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("route-centred hazard number")
    ax.set_ylabel("route-centred overload")
    ax.set_title("buffer-weighted hazard remains predictive", loc="left", pad=2)
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    x = df["response_imprint_efficiency"]
    y = df["response_loop_activation"]
    hazard = df["response_hazard_number"]
    sc = ax.scatter(
        x,
        y,
        c=np.log10(hazard + 1e-4),
        s=24 + 8 * df["response_amplitude"],
        cmap="magma_r",
        edgecolor="white",
        lw=0.35,
        alpha=0.88,
    )
    xx = np.linspace(max(0.002, x.min()), x.quantile(0.98), 80)
    for h, ls in [(0.1, ":"), (0.6, "--"), (2.5, "-")]:
        ax.plot(xx, h * xx, color="#606872", lw=0.65, ls=ls)
        ax.text(xx[-1], h * xx[-1], rf"$\Xi={h:g}$", fontsize=5.8, color="#606872", va="center")
    ax.set_xlabel(r"imprint efficiency $\eta$")
    ax.set_ylabel(r"positive loop activation $A K_+$")
    ax.set_title("storage competes with loop activation", loc="left", pad=2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("log10 hazard", labelpad=2)
    cbar.ax.tick_params(size=2, width=0.5)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    order = ["amplitude", "positive loop cost", "imprint efficiency", "loop activation", "hazard number"]
    test = tests.set_index("model").loc[order].reset_index()
    ypos = np.arange(len(test))
    colors = [BLUE, VIOLET, GOLD, RED, INK]
    ax.barh(ypos, test["spearman_y_yhat"], color=colors, alpha=0.92, height=0.58)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["amplitude", "loop cost", "imprint eff.", "loop act.", r"hazard $\Xi$"])
    ax.set_xlim(0, 1.0)
    for y0, val, r2 in zip(ypos, test["spearman_y_yhat"], test["r2_vs_training_mean"]):
        ax.text(val + 0.025, y0, f"{val:.2f}", va="center", fontsize=6.2)
        ax.text(0.03, y0, rf"$R^2={r2:.2f}$", va="center", fontsize=5.9, color="white" if val > 0.35 else INK)
    ax.set_xlabel("forward rank correlation")
    ax.set_title("activation dominates; buffering sets risk", loc="left", pad=2)
    finish(ax, axis="x")

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig26_breathing_response_function.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    hazard = corr[(corr["predictor"] == "response_hazard_number") & (corr["target"] == "response_overload_asinh")].iloc[0]
    activation = corr[(corr["predictor"] == "response_loop_activation") & (corr["target"] == "response_overload_asinh")].iloc[0]
    amp = corr[(corr["predictor"] == "response_amplitude") & (corr["target"] == "response_overload_asinh")].iloc[0]
    loop = corr[(corr["predictor"] == "response_loop_cost_positive") & (corr["target"] == "response_overload_asinh")].iloc[0]
    eta = corr[(corr["predictor"] == "response_imprint_efficiency") & (corr["target"] == "response_overload_asinh")].iloc[0]
    ltest = tests[tests["model"] == "loop activation"].iloc[0]
    htest = tests[tests["model"] == "hazard number"].iloc[0]
    atest = tests[tests["model"] == "amplitude"].iloc[0]
    lines = [
        "# Breathing response-function audit",
        "",
        "This audit asks whether the breathing metaphor can be reduced to a falsifiable response function. The analysis uses the existing three-route, cycle-resolved breathing table; no new DEM result is invented.",
        "",
        "## Definitions",
        "",
        r"- Inhale amplitude: \(A_n=\|\mathbf{y}^{hot}_n-\mathbf{y}^{cold}_n\|\).",
        r"- Imprint efficiency: \(\eta_n=\|\mathbf{y}^{cold}_{n+1}-\mathbf{y}^{cold}_n\|/(A_n+\epsilon)\).",
        r"- Positive loop cost: \(K_{+,n}=\max(\Delta L_{f,n}/A_n,0)\).",
        r"- Breathing hazard: \(\Xi_n=A_nK_{+,n}/(\eta_n+\epsilon)\). This compares loop activation written during the hot branch with the efficiency of exhaling that branch into the next cold memory.",
        "",
        "## Main findings",
        "",
        f"- Positive loop activation orders the hot overload branch across routes with Spearman rho={activation.spearman_raw:.3f} (P={activation.p_raw:.2e}, n={int(activation.n)}) and remains strong after route centring (rho={activation.spearman_within_route:.3f}, P={activation.p_within_route:.2e}).",
        f"- The buffer-weighted hazard number remains predictive after subtracting route means (rho={hazard.spearman_within_route:.3f}, P={hazard.p_within_route:.2e}).",
        f"- Hazard is stronger than amplitude alone (rho={amp.spearman_raw:.3f}), positive loop cost alone (rho={loop.spearman_raw:.3f}) or imprint efficiency alone (rho={eta.spearman_raw:.3f}), but it does not outperform positive loop activation.",
        f"- In later-cycle forward ranking, loop activation gives Spearman rho={ltest.spearman_y_yhat:.3f}; the hazard gives rho={htest.spearman_y_yhat:.3f}; amplitude gives rho={atest.spearman_y_yhat:.3f}. This should be presented as a response hierarchy, not a universal one-variable law.",
        "",
        "## Manuscript-safe interpretation",
        "",
        "The bed behaves like a driven network with a measurable response balance. A large inhale is dangerous only when positive loop activation is high and the next-cold imprint efficiency is low. Efficient exhalation buffers overload; inefficient exhalation leaves the hot branch to appear as loop-mediated overload. This makes thermal breathing an operational response function rather than a metaphor.",
        "",
        "## Boundary",
        "",
        "The response hierarchy is derived from three long-cycle routes and should be used as a rank-ordering and interpretation tool. It is not a universal constitutive law and does not replace the five-route dimensionless loop-number evidence.",
        "",
        "## Correlations",
        "",
        corr.to_markdown(index=False),
        "",
        "## Forward tests",
        "",
        tests.to_markdown(index=False),
        "",
        "## Regime summary",
        "",
        summary.to_markdown(index=False),
    ]
    (ROOT / "nature_physics_breathing_response_function.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare()
    corr = build_correlations(df)
    tests = build_model_tests(df)
    summary = build_summary(df)

    df.to_csv(SRC / "nphys_breathing_response_function_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_response_function_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_breathing_response_function_forward_tests.csv", index=False)
    summary.to_csv(SRC / "nphys_breathing_response_function_regime_summary.csv", index=False)
    plot_figure(df, corr, tests, summary)
    write_report(df, corr, tests, summary)
    print("wrote breathing response-function audit, source data and figures/nphys_fig26_breathing_response_function.*")


if __name__ == "__main__":
    main()
