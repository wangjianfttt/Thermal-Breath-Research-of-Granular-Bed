#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from itertools import permutations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_mechanism_hierarchy_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
VIOLET = "#7E6AAE"
RED = "#B6423E"
DARK_RED = "#8C2F2C"
MUTED = "#8D99A6"
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


def panel(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def exact_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rho = float(spearmanr(x_arr, y_arr).statistic)
    n = len(x_arr)
    if n > 8:
        return rho, float(spearmanr(x_arr, y_arr).pvalue)
    ranks = np.argsort(np.argsort(x_arr))
    y_sorted = y_arr[np.argsort(x_arr)]
    observed = abs(float(spearmanr(np.arange(n), y_sorted).statistic))
    count = 0
    total = 0
    for perm in permutations(range(n)):
        total += 1
        r = abs(float(spearmanr(np.arange(n), np.asarray(perm)).statistic))
        if r >= observed - 1e-12:
            count += 1
    return rho, count / total


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["route_severity"] = df["frictional_expansion"] * df["boundary_attenuation"]
    df["loop_x_severity"] = df["loop_activation"] * df["route_severity"]
    df["overload_asinh"] = df["overload_number_asinh"]
    return df


def fit_route_kernels(df: pd.DataFrame, n_boot: int = 3000, seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    boots = []
    for rid, g in df.groupby("regime_id", sort=True):
        x = g["loop_activation"].to_numpy(float)
        y = g["overload_asinh"].to_numpy(float)
        X = np.column_stack([np.ones(len(g)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        slope_samples = []
        for i in range(n_boot):
            idx = rng.integers(0, len(g), len(g))
            xb = x[idx]
            yb = y[idx]
            Xb = np.column_stack([np.ones(len(idx)), xb])
            try:
                bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
                slope_samples.append(float(bb[1]))
                boots.append({"regime_id": rid, "bootstrap": i, "slope_loop_to_overload": float(bb[1])})
            except np.linalg.LinAlgError:
                continue
        slope_samples_arr = np.asarray(slope_samples, dtype=float)
        rows.append(
            {
                "regime_id": rid,
                "n": int(len(g)),
                "route_severity": float(g["route_severity"].iloc[0]),
                "frictional_expansion": float(g["frictional_expansion"].iloc[0]),
                "boundary_attenuation": float(g["boundary_attenuation"].iloc[0]),
                "mean_loop_activation": float(g["loop_activation"].mean()),
                "mean_overload_asinh": float(g["overload_asinh"].mean()),
                "susceptibility_slope": float(beta[1]),
                "intercept": float(beta[0]),
                "r2_loop_to_overload": float(r2),
                "slope_ci_low": float(np.quantile(slope_samples_arr, 0.025)),
                "slope_ci_high": float(np.quantile(slope_samples_arr, 0.975)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boots)


def r2_vs_baseline(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(baseline)
    y = y[ok]
    yhat = yhat[ok]
    baseline = baseline[ok]
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def leave_one_route_tests(df: pd.DataFrame) -> pd.DataFrame:
    target = "overload_asinh"
    models = {
        "loop only": ["loop_activation"],
        "route severity": ["route_severity"],
        "loop + severity": ["loop_activation", "route_severity"],
        "multiplicative kernel": ["loop_x_severity"],
        "loop + severity + kernel": ["loop_activation", "route_severity", "loop_x_severity"],
        "dimensionless loop number": ["dimensionless_loop_number"],
        "top-5 force tail": ["top5_activation"],
    }
    rows = []
    predictions = []
    for name, features in models.items():
        ys: list[float] = []
        yh: list[float] = []
        base: list[float] = []
        for rid, test in df.groupby("regime_id", sort=True):
            train = df[df["regime_id"] != rid].dropna(subset=[target, *features])
            test = test.dropna(subset=[target, *features])
            x_train = train[features].to_numpy(float)
            x_test = test[features].to_numpy(float)
            y_train = train[target].to_numpy(float)
            y_test = test[target].to_numpy(float)
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            model = LinearRegression() if len(features) == 1 else RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            ys.extend(y_test)
            yh.extend(pred)
            base.extend(np.repeat(float(y_train.mean()), len(y_test)))
            for cycle, yy, pp in zip(test["cycle"], y_test, pred):
                predictions.append(
                    {
                        "model": name,
                        "left_out_route": rid,
                        "cycle": int(cycle),
                        "observed": float(yy),
                        "predicted": float(pp),
                    }
                )
        y = np.asarray(ys)
        pred = np.asarray(yh)
        baseline = np.asarray(base)
        rows.append(
            {
                "target": target,
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": int(len(y)),
                "r2_vs_training_mean": r2_vs_baseline(y, pred, baseline),
                "spearman_y_yhat": float(spearmanr(y, pred).statistic),
            }
        )
    pd.DataFrame(predictions).to_csv(SRC / "nphys_loop_susceptibility_kernel_predictions.csv", index=False)
    return pd.DataFrame(rows)


def draw_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "a", x=-0.05)

    def box(x: float, y: float, w: float, h: float, label: str, edge: str, face: str) -> None:
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
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.0, color=INK)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color, shrinkA=2, shrinkB=2))

    box(0.06, 0.70, 0.30, 0.16, "thermal route\nseverity S", BLUE, "#F4F8FC")
    box(0.58, 0.70, 0.32, 0.16, "force-loop\nsusceptibility G(S)", RED, "#FFF4F1")
    box(0.06, 0.18, 0.30, 0.16, "loop activation\nDelta Lf", VIOLET, "#F8F5FC")
    box(0.58, 0.18, 0.32, 0.16, "hot overload\nOmega", RED, "#FFF4F1")
    arrow((0.37, 0.78), (0.57, 0.78), RED)
    arrow((0.74, 0.68), (0.74, 0.36), RED)
    arrow((0.37, 0.26), (0.57, 0.26), VIOLET)
    ax.text(0.50, 0.52, r"$\Omega_n \simeq G(S)\,\Delta L_{f,n}$", ha="center", va="center", fontsize=9.0, color=INK)
    ax.text(0.50, 0.44, r"$S=(\Pi_T/\Pi_{T,0})\mu/(1+\chi)$", ha="center", va="center", fontsize=7.2, color=MUTED)
    ax.text(0.06, 0.04, "diagnostic path, not proof of causality", ha="left", va="bottom", fontsize=6.2, color=MUTED)


def plot_figure(df: pd.DataFrame, route: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.25, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.25], height_ratios=[1.0, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    draw_schematic(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for rid, g in df.groupby("regime_id", sort=True):
        color = COLORS[rid]
        ax.scatter(g["loop_activation"], g["overload_asinh"], s=18, marker=MARKERS[rid], color=color, edgecolor="white", lw=0.3, alpha=0.82, label=rid, zorder=3)
        row = route[route["regime_id"] == rid].iloc[0]
        xx = np.linspace(g["loop_activation"].min(), g["loop_activation"].max(), 100)
        yy = row["intercept"] + row["susceptibility_slope"] * xx
        ax.plot(xx, yy, color=color, lw=0.85, alpha=0.95)
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("force-loop activation, Delta Lf")
    ax.set_ylabel("hot overload, asinh scaled")
    ax.set_title("the same loop activation has route-dependent gain", loc="left", pad=2)
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0.0, -0.18), handlelength=0.9, columnspacing=0.7)
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    route = route.sort_values("route_severity")
    ax.errorbar(
        route["route_severity"],
        route["susceptibility_slope"],
        yerr=[
            route["susceptibility_slope"] - route["slope_ci_low"],
            route["slope_ci_high"] - route["susceptibility_slope"],
        ],
        fmt="none",
        ecolor="#7B838C",
        elinewidth=0.8,
        capsize=2,
        zorder=1,
    )
    for _, row in route.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["route_severity"], row["susceptibility_slope"], s=42, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", lw=0.45, zorder=3)
        ax.text(row["route_severity"] + 0.018, row["susceptibility_slope"], rid, color=COLORS[rid], fontsize=6.4, va="center")
    rho, p_exact = exact_spearman(route["route_severity"], route["susceptibility_slope"])
    ax.text(0.05, 0.95, rf"$\rho={rho:.2f}$" + f"\nexact P={p_exact:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("route severity S")
    ax.set_ylabel("susceptibility G(S)")
    ax.set_title("boundary-friction severity sets loop gain", loc="left", pad=2)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    order = ["top-5 force tail", "loop only", "route severity", "multiplicative kernel", "dimensionless loop number", "loop + severity + kernel"]
    plot = tests.set_index("model").loc[order].reset_index()
    ypos = np.arange(len(plot))
    colors = [MUTED, VIOLET, BLUE, RED, DARK_RED, INK]
    ax.barh(ypos, plot["r2_vs_training_mean"], color=colors, alpha=0.92, height=0.58)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["top-5 tail", "loop", "severity", "S x loop", "Psi", "full kernel"])
    ax.set_xlim(0, 1.0)
    for y0, r2, rho in zip(ypos, plot["r2_vs_training_mean"], plot["spearman_y_yhat"]):
        ax.text(r2 + 0.025, y0, f"{r2:.2f}", va="center", fontsize=6.2)
        ax.text(0.03, y0, rf"$\rho={rho:.2f}$", va="center", fontsize=5.9, color="white" if r2 > 0.35 else INK)
    ax.set_xlabel("leave-one-route-out R2")
    ax.set_title("multiplicative severity improves transfer", loc="left", pad=2)
    finish(ax, axis="x")

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig27_loop_susceptibility_kernel.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_report(route: pd.DataFrame, tests: pd.DataFrame) -> None:
    rho, p_exact = exact_spearman(route["route_severity"], route["susceptibility_slope"])
    best = tests.sort_values("r2_vs_training_mean", ascending=False).iloc[0]
    loop = tests[tests["model"] == "loop only"].iloc[0]
    mult = tests[tests["model"] == "multiplicative kernel"].iloc[0]
    lines = [
        "# Loop-susceptibility kernel audit",
        "",
        "Figure contract:",
        "",
        "- Core conclusion: route severity sets the susceptibility with which force-loop activation is converted into hot overload.",
        "- Figure archetype: schematic-led composite.",
        "- Backend: Python/matplotlib only.",
        "- Evidence hierarchy: route-wise loop-overload slopes, slope-versus-severity ordering, and leave-one-route-out transfer tests.",
        "- Reviewer risk: route severity has only five route levels, so this is a diagnostic path decomposition rather than a causal proof.",
        "",
        "## Main findings",
        "",
        f"- The fitted loop-to-overload susceptibility G(S) increases monotonically with route severity S (Spearman rho={rho:.3f}, exact permutation P={p_exact:.3f}, n=5 routes).",
        f"- Leave-one-route-out prediction improves from R2={loop.r2_vs_training_mean:.3f} for loop activation alone to R2={mult.r2_vs_training_mean:.3f} for the multiplicative S x loop kernel.",
        f"- The best tested transfer model is `{best.model}` with R2={best.r2_vs_training_mean:.3f} and rank correlation rho={best.spearman_y_yhat:.3f}.",
        "",
        "## Manuscript-safe interpretation",
        "",
        "The same amount of force-loop activation is not equally dangerous in all routes. Thermal expansion, friction and boundary clearance set a route-level susceptibility, and force loops supply the cycle-level activation. The overload branch is therefore a multiplicative response, not simply a scalar pressure law or a force-tail law.",
        "",
        "## Boundary",
        "",
        "This analysis uses five route levels and repeated cycles within each route. It should be described as a route-conditioned susceptibility diagnostic. It supports the dimensionless-loop formulation but does not prove a universal constitutive kernel.",
        "",
        "## Route kernels",
        "",
        route.to_markdown(index=False),
        "",
        "## Transfer tests",
        "",
        tests.to_markdown(index=False),
    ]
    (ROOT / "nature_physics_loop_susceptibility_kernel.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare()
    route, boot = fit_route_kernels(df)
    tests = leave_one_route_tests(df)
    df.to_csv(SRC / "nphys_loop_susceptibility_kernel_cycle_metrics.csv", index=False)
    route.to_csv(SRC / "nphys_loop_susceptibility_kernel_route_kernels.csv", index=False)
    boot.to_csv(SRC / "nphys_loop_susceptibility_kernel_bootstrap.csv", index=False)
    tests.to_csv(SRC / "nphys_loop_susceptibility_kernel_model_tests.csv", index=False)
    plot_figure(df, route, tests)
    write_report(route, tests)
    print("wrote loop-susceptibility kernel audit, source data and figures/nphys_fig27_loop_susceptibility_kernel.*")


if __name__ == "__main__":
    main()
