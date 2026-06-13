#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import auc, brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_mechanism_hierarchy_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
RED = "#B6423E"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
VIOLET = "#7E6AAE"
NEUTRAL = "#8D99A6"
COLORS = {"R1": BLUE, "R3": GOLD, "R5": VIOLET, "R6": "#C95F3F", "R6c": "#8D3138"}
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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).copy()
    df = df.sort_values(["regime_id", "cycle"])
    df["overload_binary"] = (df["overload_number"] > 0).astype(int)
    df["psi"] = df["dimensionless_loop_number"]
    df["tail"] = df["dimensionless_top5_number"]
    df["loop"] = df["loop_activation"]
    df["controls"] = df["alpha_mult"] * df["friction"] / (1.0 + df["lid_gap_radii"])
    return df


def best_threshold(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    order = np.argsort(x)
    xs = x[order]
    candidates = np.r_[xs[0] - 1e-9, (xs[:-1] + xs[1:]) / 2.0, xs[-1] + 1e-9]
    best = {"threshold": np.nan, "balanced_accuracy": -np.inf, "sensitivity": np.nan, "specificity": np.nan}
    for th in candidates:
        pred = x >= th
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        bal = 0.5 * (sens + spec)
        if bal > best["balanced_accuracy"]:
            best = {"threshold": float(th), "balanced_accuracy": float(bal), "sensitivity": float(sens), "specificity": float(spec)}
    return best


def logistic_leave_one_route(df: pd.DataFrame, features: list[str]) -> dict[str, float | str | int]:
    y_all: list[int] = []
    p_all: list[float] = []
    for rid in sorted(df["regime_id"].unique()):
        train = df[df["regime_id"] != rid].dropna(subset=features + ["overload_binary"])
        test = df[df["regime_id"] == rid].dropna(subset=features + ["overload_binary"])
        if train["overload_binary"].nunique() < 2:
            continue
        scaler = StandardScaler().fit(train[features])
        x_train = scaler.transform(train[features])
        x_test = scaler.transform(test[features])
        model = LogisticRegression(C=1.0, solver="lbfgs").fit(x_train, train["overload_binary"])
        p = model.predict_proba(x_test)[:, 1]
        y_all.extend(test["overload_binary"].astype(int).tolist())
        p_all.extend(p.tolist())
    y = np.asarray(y_all, dtype=int)
    p = np.asarray(p_all, dtype=float)
    pred = p >= 0.5
    return {
        "features": ";".join(features),
        "validation": "leave_one_route_out",
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "accuracy": float(np.mean(pred == y)) if len(y) else np.nan,
        "spearman_y_prob": float(spearmanr(y, p).statistic) if len(y) > 2 else np.nan,
    }


def build_logistic_tests(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "dimensionless_loop": ["psi"],
        "raw_loop": ["loop"],
        "force_tail": ["tail"],
        "route_controls": ["controls"],
        "loop_plus_tail": ["psi", "tail"],
        "loop_plus_controls": ["psi", "controls"],
    }
    rows = []
    for name, features in feature_sets.items():
        row = logistic_leave_one_route(df, features)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_threshold(df: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    routes = sorted(df["regime_id"].unique())
    for i in range(n_boot):
        parts = []
        for rid in routes:
            g = df[df["regime_id"] == rid]
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        y = sample["overload_binary"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        th = best_threshold(sample["psi"].to_numpy(float), y)
        th["bootstrap"] = i
        rows.append(th)
    return pd.DataFrame(rows)


def hinge_fit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=["psi", "overload_number_asinh"]).copy()
    x = d["psi"].to_numpy(float)
    y = d["overload_number_asinh"].to_numpy(float)
    candidates = np.linspace(np.quantile(x, 0.08), np.quantile(x, 0.92), 120)
    rows = []
    for c in candidates:
        X = np.column_stack([np.ones(len(x)), x, np.maximum(0, x - c)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        sse = float(np.sum((y - yhat) ** 2))
        rows.append({"breakpoint": float(c), "sse": sse, "intercept": beta[0], "slope_low": beta[1], "slope_high_increment": beta[2]})
    scan = pd.DataFrame(rows)
    best = scan.loc[scan["sse"].idxmin()].copy()
    c = float(best["breakpoint"])
    Xh = np.column_stack([np.ones(len(x)), x, np.maximum(0, x - c)])
    bh, *_ = np.linalg.lstsq(Xh, y, rcond=None)
    yh = Xh @ bh
    Xl = np.column_stack([np.ones(len(x)), x])
    bl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
    yl = Xl @ bl
    sse_h = float(np.sum((y - yh) ** 2))
    sse_l = float(np.sum((y - yl) ** 2))
    r2_gain = 1.0 - sse_h / sse_l if sse_l > 0 else np.nan
    summary = pd.DataFrame(
        [
            {
                "breakpoint": c,
                "sse_hinge": sse_h,
                "sse_linear": sse_l,
                "r2_gain_vs_linear": r2_gain,
                "intercept": bh[0],
                "slope_low": bh[1],
                "slope_high": bh[1] + bh[2],
            }
        ]
    )
    return scan, summary


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("regime_id", sort=True)
        .agg(
            n=("cycle", "count"),
            mean_psi=("psi", "mean"),
            sd_psi=("psi", "std"),
            mean_overload=("overload_number", "mean"),
            overload_probability=("overload_binary", "mean"),
            mean_tail=("tail", "mean"),
            mean_loop=("loop", "mean"),
        )
        .reset_index()
    )


def make_figure(df: pd.DataFrame, tests: pd.DataFrame, boot: pd.DataFrame, hinge_summary: pd.DataFrame, routes: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(g["psi"], g["overload_number_asinh"], s=22, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", lw=0.35, alpha=0.86, label=rid)
    threshold = float(boot["threshold"].median())
    ax.axvline(threshold, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.text(threshold, ax.get_ylim()[1], rf"$\Psi_c\approx{threshold:.3f}$", ha="left", va="top", fontsize=6.6)
    ax.set_xlabel(r"dimensionless loop number $\Psi$")
    ax.set_ylabel(r"overload number, asinh")
    ax.set_title("overload onset is organised by loop number")
    ax.legend(ncol=3, loc="upper left", handlelength=1.0, columnspacing=0.7)
    panel(ax, "a")
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    x = routes["mean_psi"].to_numpy(float)
    y = routes["overload_probability"].to_numpy(float)
    for _, row in routes.iterrows():
        ax.errorbar(row["mean_psi"], row["overload_probability"], xerr=row["sd_psi"], fmt=MARKERS[row["regime_id"]], color=COLORS[row["regime_id"]], mec="white", mew=0.35, ms=5.2, lw=0.8)
        ax.text(row["mean_psi"] + 0.004, row["overload_probability"], row["regime_id"], color=COLORS[row["regime_id"]], va="center", fontsize=6.4)
    xx = np.linspace(min(df["psi"]) - 0.005, max(df["psi"]) + 0.005, 240)
    scaler = StandardScaler().fit(df[["psi"]])
    model = LogisticRegression(C=1.0, solver="lbfgs").fit(scaler.transform(df[["psi"]]), df["overload_binary"])
    prob = model.predict_proba(scaler.transform(pd.DataFrame({"psi": xx})))[:, 1]
    ax.plot(xx, prob, color=INK, lw=0.9)
    ax.axvline(threshold, color=INK, lw=0.7, ls=(0, (4, 3)))
    ax.set_xlabel(r"$\Psi$")
    ax.set_ylabel("probability of positive overload")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("route means show an onset crossover")
    panel(ax, "b")
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    order = ["dimensionless_loop", "raw_loop", "force_tail", "route_controls", "loop_plus_tail", "loop_plus_controls"]
    labels = [r"$\Psi$", r"$\Delta L_f$", r"$q_5$", "controls", r"$\Psi+q_5$", r"$\Psi+$ctrl"]
    colors = [RED, VIOLET, GOLD, NEUTRAL, INK, BLUE]
    t = tests.set_index("model").loc[order]
    y_pos = np.arange(len(order))
    ax.barh(y_pos, t["auc"], color=colors, height=0.64, zorder=3)
    ax.axvline(0.5, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlim(0.0, 1.02)
    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("leave-one-route-out AUC")
    ax.set_title("loop onset beats force-tail alternatives")
    panel(ax, "c")
    finish(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(boot["threshold"], bins=36, color=RED, alpha=0.82, edgecolor="white", lw=0.25)
    q = boot["threshold"].quantile([0.025, 0.5, 0.975])
    for val, ls in [(q.loc[0.5], "-"), (q.loc[0.025], (0, (3, 3))), (q.loc[0.975], (0, (3, 3)))]:
        ax.axvline(val, color=INK, lw=0.8, ls=ls)
    bp = float(hinge_summary["breakpoint"].iloc[0])
    ax.axvline(bp, color=BLUE, lw=0.8, ls=(0, (5, 2)))
    ax.text(0.97, 0.94, f"bootstrap median {q.loc[0.5]:.3f}\n95% {q.loc[0.025]:.3f}-{q.loc[0.975]:.3f}\nhinge {bp:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.6)
    ax.set_xlabel(r"onset threshold $\Psi_c$")
    ax.set_ylabel("bootstrap count")
    ax.set_title("threshold is finite but not a critical point")
    panel(ax, "d")
    finish(ax, "y")

    out = FIG / "nphys_fig25_loop_onset_susceptibility"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(tests: pd.DataFrame, boot: pd.DataFrame, hinge_summary: pd.DataFrame, routes: pd.DataFrame) -> None:
    q = boot["threshold"].quantile([0.025, 0.5, 0.975])
    by_model = tests.set_index("model")
    psi = by_model.loc["dimensionless_loop"]
    raw_loop = by_model.loc["raw_loop"]
    tail = tests.set_index("model").loc["force_tail"]
    controls = tests.set_index("model").loc["route_controls"]
    hinge = hinge_summary.iloc[0]
    report = ROOT / "nature_physics_loop_onset_susceptibility.md"
    report.write_text(
        "# Loop-onset susceptibility audit\n\n"
        "This audit tests whether the hot-overload branch can be described as an onset controlled by the dimensionless loop number, rather than as a vague correlation. It uses the five-route true-force dataset with 150 hot-minus-cold pairs.\n\n"
        "## Main findings\n\n"
        f"- Route-balanced bootstrap thresholding gives an overload-onset coordinate `Psi_c={q.loc[0.5]:.4f}` with 95% interval `{q.loc[0.025]:.4f}` to `{q.loc[0.975]:.4f}`.\n"
        f"- Leave-one-route-out logistic prediction gives AUC={raw_loop.auc:.3f} for raw loop activation and AUC={psi.auc:.3f} for the dimensionless loop number, compared with AUC={tail.auc:.3f} for the force-tail surrogate and AUC={controls.auc:.3f} for route controls alone.\n"
        f"- A hinge fit to the asinh overload number selects a breakpoint `Psi={hinge.breakpoint:.4f}` and improves SSE over a single linear response by `R2_gain={hinge.r2_gain_vs_linear:.3f}`.\n"
        "- Route means move from an almost always buffered R1 state to near-certain overload in R5/R6/R6c, with R3 occupying the crossover sector.\n\n"
        "## Manuscript-safe interpretation\n\n"
        "The data support an onset crossover controlled by force-loop activation. Raw loop activation is the sharper binary onset coordinate, whereas the dimensionless loop number is the better route-aware coordinate for organising overload amplitude. Both outperform a force-tail surrogate. This gives an experimentally testable threshold-like coordinate, but it should not be called a critical point or phase transition in the strict thermodynamic sense because the dataset has five routes, finite cycles and no finite-size scaling.\n\n"
        "## Route summary\n\n"
        + routes.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    df = prepare()
    tests = build_logistic_tests(df)
    boot = bootstrap_threshold(df)
    scan, hinge_summary = hinge_fit(df)
    routes = route_summary(df)
    tests.to_csv(SRC / "nphys_loop_onset_logistic_tests.csv", index=False)
    boot.to_csv(SRC / "nphys_loop_onset_threshold_bootstrap.csv", index=False)
    scan.to_csv(SRC / "nphys_loop_onset_hinge_scan.csv", index=False)
    hinge_summary.to_csv(SRC / "nphys_loop_onset_hinge_summary.csv", index=False)
    routes.to_csv(SRC / "nphys_loop_onset_route_summary.csv", index=False)
    make_figure(df, tests, boot, hinge_summary, routes)
    write_report(tests, boot, hinge_summary, routes)
    print(tests.to_string(index=False))
    print(boot["threshold"].quantile([0.025, 0.5, 0.975]).to_string())
    print(hinge_summary.to_string(index=False))
    print(routes.to_string(index=False))


if __name__ == "__main__":
    main()
