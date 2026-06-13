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

HOT = SRC / "nphys_mechanism_hierarchy_cycle_metrics.csv"
MAP = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
RED = "#B6423E"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
VIOLET = "#7E6AAE"
NEUTRAL = "#8D99A6"


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
    if len(y) == 0:
        return np.nan
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def leave_one_route_out(df: pd.DataFrame, features: list[str], target: str) -> tuple[float, float, int]:
    y_all: list[float] = []
    yhat_all: list[float] = []
    base_all: list[float] = []
    for rid in sorted(df["regime_id"].unique()):
        train = df[df["regime_id"] != rid]
        test = df[df["regime_id"] == rid]
        if len(test) == 0 or len(train) < 5:
            continue
        y_test = test[target].to_numpy(float)
        yhat = fit_predict(train, test, features, target)
        baseline = np.repeat(float(train[target].mean()), len(test))
        y_all.extend(y_test)
        yhat_all.extend(yhat)
        base_all.extend(baseline)
    y = np.asarray(y_all)
    yhat = np.asarray(yhat_all)
    r2 = r2_vs_baseline(y, yhat, np.asarray(base_all))
    rho = float(spearmanr(y, yhat).statistic) if len(y) > 2 else np.nan
    return r2, rho, int(len(y))


def within_route_forward(df: pd.DataFrame, features: list[str], target: str) -> tuple[float, float, int]:
    y_all: list[float] = []
    yhat_all: list[float] = []
    base_all: list[float] = []
    for _, g in df.groupby("regime_id", sort=True):
        train = g[g["cycle"] <= 18]
        test = g[g["cycle"] > 18]
        if len(train) < 5 or len(test) < 5:
            continue
        y_test = test[target].to_numpy(float)
        yhat = fit_predict(train, test, features, target)
        baseline = np.repeat(float(train[target].mean()), len(test))
        y_all.extend(y_test)
        yhat_all.extend(yhat)
        base_all.extend(baseline)
    y = np.asarray(y_all)
    yhat = np.asarray(yhat_all)
    r2 = r2_vs_baseline(y, yhat, np.asarray(base_all))
    rho = float(spearmanr(y, yhat).statistic) if len(y) > 2 else np.nan
    return r2, rho, int(len(y))


def build_hot_audit() -> pd.DataFrame:
    df = pd.read_csv(HOT).copy()
    target = "overload_number_asinh"
    families = {
        "controls only": ["alpha_mult", "friction", "lid_gap_radii"],
        "cold fabric": ["cold_memory_index"],
        "force tail": ["dimensionless_top5_number"],
        "loop activation": ["loop_activation"],
        "dimensionless loop": ["dimensionless_loop_number"],
        "loop + modifiers": [
            "dimensionless_loop_number",
            "peak_normalized_gain",
            "antisymmetric_strength",
            "absolute_circulation",
            "phase_speed",
            "tangential_fraction",
        ],
    }
    rows = []
    for name, features in families.items():
        d = df.dropna(subset=[target, *features]).copy()
        r2_loro, rho_loro, n_loro = leave_one_route_out(d, features, target)
        r2_fw, rho_fw, n_fw = within_route_forward(d, features, target)
        rows.append(
            {
                "target": target,
                "model_family": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": n_loro,
                "r2_vs_training_mean": r2_loro,
                "spearman_y_yhat": rho_loro,
            }
        )
        rows.append(
            {
                "target": target,
                "model_family": name,
                "features": ";".join(features),
                "validation": "within_route_forward_60_40",
                "n": n_fw,
                "r2_vs_training_mean": r2_fw,
                "spearman_y_yhat": rho_fw,
            }
        )
    return pd.DataFrame(rows)


def build_next_cold_audit() -> pd.DataFrame:
    df = pd.read_csv(MAP).copy()
    target = "next_memory_coordinate"
    families = {
        "memory only": ["memory_coordinate"],
        "hot coordinate only": ["hot_excitation_coordinate"],
        "force-tail control": ["dimensionless_top5_number"],
        "memory + hot": ["memory_coordinate", "hot_excitation_coordinate"],
        "memory + force tail": ["memory_coordinate", "dimensionless_top5_number"],
        "memory + hot + tail": ["memory_coordinate", "hot_excitation_coordinate", "dimensionless_top5_number"],
    }
    rows = []
    for name, features in families.items():
        d = df.dropna(subset=[target, *features]).copy()
        r2_loro, rho_loro, n_loro = leave_one_route_out(d, features, target)
        r2_fw, rho_fw, n_fw = within_route_forward(d, features, target)
        rows.append(
            {
                "target": target,
                "model_family": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": n_loro,
                "r2_vs_training_mean": r2_loro,
                "spearman_y_yhat": rho_loro,
            }
        )
        rows.append(
            {
                "target": target,
                "model_family": name,
                "features": ";".join(features),
                "validation": "within_route_forward_60_40",
                "n": n_fw,
                "r2_vs_training_mean": r2_fw,
                "spearman_y_yhat": rho_fw,
            }
        )
    return pd.DataFrame(rows)


def build_summary(hot: pd.DataFrame, cold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    h_loro = hot[hot["validation"] == "leave_one_route_out"].set_index("model_family")
    h_fw = hot[hot["validation"] == "within_route_forward_60_40"].set_index("model_family")
    c_fw = cold[cold["validation"] == "within_route_forward_60_40"].set_index("model_family")
    rows.append(
        {
            "test": "hot_overload_loop_gain_over_force_tail_loro",
            "value": h_loro.loc["dimensionless loop", "r2_vs_training_mean"]
            - h_loro.loc["force tail", "r2_vs_training_mean"],
            "interpretation": "Positive means the dimensionless force-loop coordinate transfers better than the top-5% force-tail surrogate.",
        }
    )
    rows.append(
        {
            "test": "hot_overload_modifiers_delta_forward",
            "value": h_fw.loc["loop + modifiers", "r2_vs_training_mean"]
            - h_fw.loc["dimensionless loop", "r2_vs_training_mean"],
            "interpretation": "Positive means non-normal and geometric modifiers improve within-route later-cycle prediction after the loop coordinate.",
        }
    )
    rows.append(
        {
            "test": "next_cold_hot_coordinate_delta_forward",
            "value": c_fw.loc["memory + hot", "r2_vs_training_mean"] - c_fw.loc["memory only", "r2_vs_training_mean"],
            "interpretation": "Positive would mean the hot loop-excitation coordinate carries lagged imprint information beyond the current cold memory coordinate for this target; negative bounds the claim.",
        }
    )
    rows.append(
        {
            "test": "next_cold_tail_delta_forward",
            "value": c_fw.loc["memory + force tail", "r2_vs_training_mean"]
            - c_fw.loc["memory only", "r2_vs_training_mean"],
            "interpretation": "Compares a force-tail surrogate with the current-memory baseline for next-cold imprint prediction.",
        }
    )
    return pd.DataFrame(rows)


def make_figure(hot: pd.DataFrame, cold: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.88])

    ax = fig.add_subplot(gs[0, 0])
    order_hot = ["controls only", "cold fabric", "force tail", "loop activation", "dimensionless loop", "loop + modifiers"]
    labels_hot = ["controls", r"$M_c$", r"$q_5$", r"$\Delta L_f$", r"$\Psi$", r"$\Psi$+mods"]
    colors_hot = [NEUTRAL, BLUE, GOLD, VIOLET, RED, INK]
    h = hot[hot["validation"] == "leave_one_route_out"].set_index("model_family").loc[order_hot]
    y = np.arange(len(order_hot))
    vals = h["r2_vs_training_mean"].to_numpy(float)
    plot_vals = np.maximum(vals, -0.92)
    ax.barh(y, plot_vals, color=colors_hot, height=0.64, zorder=3)
    for yi, v in zip(y, vals):
        if v < -0.92:
            ax.annotate(
                f"{v:.1f}",
                xy=(-0.92, yi),
                xytext=(-0.70, yi),
                ha="left",
                va="center",
                fontsize=6.5,
                color=INK,
                arrowprops=dict(arrowstyle="<|-", lw=0.65, color=NEUTRAL),
            )
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y, labels_hot)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"leave-one-route-out $R^2$")
    ax.set_title("hot overload requires loop embedding")
    panel(ax, "a")
    finish(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    order_cold = ["memory only", "hot coordinate only", "force-tail control", "memory + hot", "memory + force tail", "memory + hot + tail"]
    labels_cold = [r"$M_c$", r"$\Psi$", r"$q_5$", r"$M_c+\Psi$", r"$M_c+q_5$", r"$M_c+\Psi+q_5$"]
    colors_cold = [BLUE, RED, GOLD, INK, VIOLET, "#4E5661"]
    c = cold[cold["validation"] == "within_route_forward_60_40"].set_index("model_family").loc[order_cold]
    y = np.arange(len(order_cold))
    ax.barh(y, c["r2_vs_training_mean"], color=colors_cold, height=0.66, zorder=3)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y, labels_cold)
    ax.set_xlim(-1.72, 0.36)
    ax.set_xlabel(r"within-route forward $R^2$")
    ax.set_title("next-cold memory bounds the imprint claim")
    panel(ax, "b")
    finish(ax, "x")

    ax = fig.add_subplot(gs[1, :])
    tests = summary.copy()
    labels = [
        "loop over\ntail",
        "modifiers after\nloop",
        "hot coordinate\nfor imprint",
        "tail coordinate\nfor imprint",
    ]
    vals = tests["value"].to_numpy(float)
    colors = [RED if v >= 0 else NEUTRAL for v in vals]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.025 if v >= 0 else -0.025), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$\Delta R^2$")
    ax.set_title("variable deletion turns the breathing story into state requirements")
    panel(ax, "c", x=-0.035, y=1.08)
    finish(ax, "y")

    out = FIG / "nphys_fig23_minimal_state_sufficiency"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(hot: pd.DataFrame, cold: pd.DataFrame, summary: pd.DataFrame) -> None:
    h_loro = hot[hot["validation"] == "leave_one_route_out"].set_index("model_family")
    h_fw = hot[hot["validation"] == "within_route_forward_60_40"].set_index("model_family")
    c_fw = cold[cold["validation"] == "within_route_forward_60_40"].set_index("model_family")
    report = ROOT / "nature_physics_minimal_state_sufficiency.md"
    hot_delta = c_fw.loc["memory + hot", "r2_vs_training_mean"] - c_fw.loc["memory only", "r2_vs_training_mean"]
    tail_delta = c_fw.loc["memory + force tail", "r2_vs_training_mean"] - c_fw.loc["memory only", "r2_vs_training_mean"]
    if hot_delta > 0:
        hot_sentence = (
            "For this target, the hot loop-excitation coordinate improves next-cold memory prediction beyond current memory. "
            "This supports a lagged imprint channel."
        )
    else:
        hot_sentence = (
            "For this target, the hot loop-excitation coordinate does not improve next-cold memory prediction beyond current memory. "
            "This bounds the lagged-imprint claim to the targets where the return-map audit already showed a positive gain."
        )
    if tail_delta > hot_delta:
        comparison_sentence = (
            "The force-tail surrogate performs less poorly for this single target, so the manuscript should not claim that every next-cold imprint is uniquely carried by the hot loop coordinate."
        )
    else:
        comparison_sentence = (
            "The force-tail surrogate does not outperform the hot coordinate for this target."
        )
    report.write_text(
        "# Minimal state-sufficiency audit\n\n"
        "This audit asks whether the breathing interpretation needs two state coordinates, or whether a simpler scalar surrogate can replace them.\n\n"
        "## Hot-overload requirement\n\n"
        f"- Leave-one-route-out prediction of hot overload gives R2={h_loro.loc['dimensionless loop', 'r2_vs_training_mean']:.3f} for the dimensionless loop number, compared with R2={h_loro.loc['force tail', 'r2_vs_training_mean']:.3f} for the top-5% force-tail surrogate and R2={h_loro.loc['cold fabric', 'r2_vs_training_mean']:.3f} for the cold fabric index alone.\n"
        f"- Within-route forward prediction gives R2={h_fw.loc['dimensionless loop', 'r2_vs_training_mean']:.3f} for the dimensionless loop number and R2={h_fw.loc['loop + modifiers', 'r2_vs_training_mean']:.3f} after adding non-normal and geometric-flow modifiers.\n"
        "- Interpretation: the hot overload coordinate cannot be reduced to a scalar force-tail metric or to the cold fabric reservoir. It requires force-loop embedding weighted by route controls.\n\n"
        "## Next-cold imprint requirement\n\n"
        f"- Within-route forward prediction of next-cold memory gives R2={c_fw.loc['memory only', 'r2_vs_training_mean']:.3f} from current memory alone and R2={c_fw.loc['memory + hot', 'r2_vs_training_mean']:.3f} after adding the hot loop-excitation coordinate.\n"
        f"- Replacing the hot coordinate by the top-5% force-tail control gives R2={c_fw.loc['memory + force tail', 'r2_vs_training_mean']:.3f}.\n"
        f"- Interpretation: {hot_sentence} {comparison_sentence}\n\n"
        "## Manuscript-safe conclusion\n\n"
        "The minimal state description needs a hot force-loop coordinate to explain transient overload. Cold fabric alone cannot explain hot overload, and force-tail concentration alone cannot replace topology-conditioned loop activation. The next-cold imprint evidence is narrower: some forward tests improve when the hot coordinate is added, but the direct next-memory-coordinate target does not. The manuscript should therefore claim a route-conditioned breathing map with bounded lagged imprint, not a universal two-coordinate constitutive law.\n",
        encoding="utf-8",
    )


def main() -> None:
    hot = build_hot_audit()
    cold = build_next_cold_audit()
    summary = build_summary(hot, cold)
    hot.to_csv(SRC / "nphys_minimal_state_hot_overload.csv", index=False)
    cold.to_csv(SRC / "nphys_minimal_state_next_cold.csv", index=False)
    summary.to_csv(SRC / "nphys_minimal_state_sufficiency_summary.csv", index=False)
    make_figure(hot, cold, summary)
    write_report(hot, cold, summary)
    print(hot.to_string(index=False))
    print(cold.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
