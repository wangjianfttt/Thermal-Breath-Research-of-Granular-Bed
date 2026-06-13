#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INK = "#252A31"
GRID = "#E7EAEE"
BUFFERED = "#3D6B9C"
MILD = "#D98C3A"
SEVERE = "#B6423E"
NEUTRAL = "#8B929A"
ROUTE_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R6": "#C95F3F"}
ROUTE_MARKERS = {"R1": "o", "R3": "s", "R6": "^"}


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


def panel(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SRC / "nphys_breathing_response_function_cycle_metrics.csv")
    cols = [
        "regime_id",
        "cycle",
        "response_amplitude",
        "response_loop_cost_positive",
        "response_loop_activation",
        "response_imprint_efficiency",
        "response_hazard_number",
        "response_quality_number",
        "response_overload_asinh",
        "force_p99_hot_minus_cold",
    ]
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df["activation_eps"] = df["response_loop_activation"] + 1e-5
    df["imprint_eps"] = df["response_imprint_efficiency"] + 1e-5
    df["log_activation"] = np.log10(df["activation_eps"])
    df["log_hazard"] = np.log10(df["response_hazard_number"] + 1e-5)
    df["buffer_number"] = df["response_imprint_efficiency"] / (df["response_loop_activation"] + 1e-5)
    df["log_buffer_number"] = np.log10(df["buffer_number"] + 1e-5)

    bins = [-np.inf, 0.0, 1.0, np.inf]
    labels = ["negative/buffered", "mild overload", "severe overload"]
    df["measured_overload_class"] = pd.cut(df["response_overload_asinh"], bins=bins, labels=labels)
    df["positive_overload"] = (df["response_overload_asinh"] > 0.0).astype(int)
    df["strong_overload"] = (df["response_overload_asinh"] > 1.0).astype(int)
    return df


def correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        "response_loop_activation",
        "response_imprint_efficiency",
        "response_hazard_number",
        "response_amplitude",
        "buffer_number",
    ]
    targets = ["response_overload_asinh", "positive_overload", "strong_overload"]
    for target in targets:
        for pred in predictors:
            d = df[[pred, target]].replace([np.inf, -np.inf], np.nan).dropna()
            sp = spearmanr(d[pred], d[target])
            rows.append(
                {
                    "target": target,
                    "predictor": pred,
                    "n": len(d),
                    "spearman_rho": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                }
            )
    return pd.DataFrame(rows)


def forward_logistic_tests(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "activation": ["response_loop_activation"],
        "imprint": ["response_imprint_efficiency"],
        "hazard": ["response_hazard_number"],
        "activation+imprint": ["response_loop_activation", "response_imprint_efficiency"],
        "amplitude+cost+imprint": [
            "response_amplitude",
            "response_loop_cost_positive",
            "response_imprint_efficiency",
        ],
    }
    targets = {"positive_overload": 0.0, "strong_overload": 1.0}
    rows = []
    for target, threshold in targets.items():
        for model, features in models.items():
            y_all: list[int] = []
            p_all: list[float] = []
            pred_all: list[int] = []
            for rid, group in df.groupby("regime_id", sort=True):
                train = group[group["cycle"] <= 18].dropna(subset=[*features, "response_overload_asinh"])
                test = group[group["cycle"] > 18].dropna(subset=[*features, "response_overload_asinh"])
                if len(train) < 6 or len(test) < 3:
                    continue
                y_train = (train["response_overload_asinh"] > threshold).astype(int).to_numpy()
                y_test = (test["response_overload_asinh"] > threshold).astype(int).to_numpy()
                if len(np.unique(y_train)) < 2:
                    continue
                scaler = StandardScaler().fit(train[features])
                clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=0)
                clf.fit(scaler.transform(train[features]), y_train)
                prob = clf.predict_proba(scaler.transform(test[features]))[:, 1]
                y_all.extend(y_test.tolist())
                p_all.extend(prob.tolist())
                pred_all.extend((prob >= 0.5).astype(int).tolist())
            y = np.asarray(y_all)
            prob = np.asarray(p_all)
            pred = np.asarray(pred_all)
            auc = roc_auc_score(y, prob) if len(y) and len(np.unique(y)) == 2 else np.nan
            bal = balanced_accuracy_score(y, pred) if len(y) and len(np.unique(y)) == 2 else np.nan
            rho = spearmanr(y, prob).statistic if len(y) and len(np.unique(y)) == 2 else np.nan
            rows.append(
                {
                    "target": target,
                    "threshold_overload_asinh": threshold,
                    "model": model,
                    "features": ";".join(features),
                    "validation": "within_route_forward_60_40",
                    "n_test": int(len(y)),
                    "auc": float(auc) if np.isfinite(auc) else np.nan,
                    "balanced_accuracy": float(bal) if np.isfinite(bal) else np.nan,
                    "spearman_y_prob": float(rho) if np.isfinite(rho) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def route_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["regime_id", "measured_overload_class"], observed=True)
        .agg(
            n=("cycle", "count"),
            mean_activation=("response_loop_activation", "mean"),
            mean_imprint_efficiency=("response_imprint_efficiency", "mean"),
            mean_hazard=("response_hazard_number", "mean"),
            mean_overload_asinh=("response_overload_asinh", "mean"),
            cycle_min=("cycle", "min"),
            cycle_max=("cycle", "max"),
        )
        .reset_index()
    )


def build_figure(df: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 4.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", x=-0.12)
    class_colors = {
        "negative/buffered": BUFFERED,
        "mild overload": MILD,
        "severe overload": SEVERE,
    }
    for cls, group in df.groupby("measured_overload_class", observed=True):
        ax.scatter(
            group["response_loop_activation"] + 1e-5,
            group["response_imprint_efficiency"] + 1e-5,
            s=28,
            color=class_colors[str(cls)],
            edgecolor="white",
            lw=0.35,
            alpha=0.86,
            label=str(cls),
            zorder=4,
        )
    xgrid = np.logspace(-5, -0.8, 200)
    for hazard, label in [(0.1, "buffer boundary"), (0.6, "overload boundary"), (3.0, "severe branch")]:
        y = xgrid / hazard
        ax.plot(xgrid, y, color=NEUTRAL, lw=0.75, ls=(0, (3, 3)))
        if hazard in {0.6, 3.0}:
            idx = 145 if hazard == 0.6 else 162
            ax.text(xgrid[idx], y[idx] * 1.08, rf"$\Xi={hazard:g}$", fontsize=5.9, color=NEUTRAL, rotation=22)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e-6, 2e-1)
    ax.set_ylim(7e-3, 2.0)
    ax.set_xlabel(r"positive loop activation, $A K_+$")
    ax.set_ylabel(r"imprint efficiency, $\eta$")
    ax.set_title("breathing phase diagram", fontsize=7.6, pad=5)
    ax.legend(loc="lower left", fontsize=5.8, handletextpad=0.25)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for rid, group in df.groupby("regime_id", sort=True):
        ax.plot(group["cycle"], group["response_loop_activation"], color=ROUTE_COLORS[rid], lw=1.0)
        ax.scatter(group["cycle"], group["response_loop_activation"], marker=ROUTE_MARKERS[rid], color=ROUTE_COLORS[rid], s=13, edgecolor="white", lw=0.25, label=rid)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$A K_+$")
    ax.set_title("activation rhythm", fontsize=7.5, pad=5)
    ax.legend(ncol=3, loc="upper left", fontsize=5.7, handletextpad=0.25, columnspacing=0.7)
    finish(ax)

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    for rid, group in df.groupby("regime_id", sort=True):
        ax.plot(group["cycle"], group["response_imprint_efficiency"], color=ROUTE_COLORS[rid], lw=1.0)
        ax.scatter(group["cycle"], group["response_imprint_efficiency"], marker=ROUTE_MARKERS[rid], color=ROUTE_COLORS[rid], s=13, edgecolor="white", lw=0.25)
    ax.axhline(0.1, color=NEUTRAL, lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\eta$")
    ax.set_title("buffering efficiency", fontsize=7.5, pad=5)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    pred_order = ["response_loop_activation", "response_hazard_number", "response_amplitude", "response_imprint_efficiency"]
    labels = ["activation", "hazard", "amplitude", "imprint"]
    sub = corr[corr["target"] == "strong_overload"].set_index("predictor").loc[pred_order]
    vals = sub["spearman_rho"].to_numpy(float)
    colors = [SEVERE if i < 2 else NEUTRAL for i in range(len(vals))]
    ax.axvline(0, color="#B8BDC4", lw=0.65)
    ax.barh(np.arange(len(vals)), vals, color=colors, height=0.68)
    ax.set_yticks(np.arange(len(vals)), labels)
    ax.set_xlabel("Spearman rho")
    ax.set_xlim(-0.45, 0.75)
    ax.set_title("severe-overload ranking", fontsize=7.5, pad=5)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    test_sub = tests[tests["target"] == "strong_overload"].copy()
    order = ["activation", "hazard", "activation+imprint", "imprint"]
    test_sub = test_sub.set_index("model").loc[order].reset_index()
    ax.bar(np.arange(len(test_sub)), test_sub["auc"], color=[SEVERE, SEVERE, "#7E6AAE", NEUTRAL], width=0.68)
    ax.set_xticks(np.arange(len(test_sub)), ["activation", "hazard", "both", "imprint"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("forward AUC")
    ax.set_title("late-cycle transfer test", fontsize=7.5, pad=5)
    ax.text(0.05, 0.08, f"n={int(test_sub['n_test'].max())}", transform=ax.transAxes, fontsize=6.1, color=INK)
    finish(ax, axis="y")

    out = FIG / "nphys_fig30_breathing_phase_diagram"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(corr: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame) -> None:
    def rho(target: str, pred: str) -> float:
        return float(corr.query("target == @target and predictor == @pred")["spearman_rho"].iloc[0])

    def auc(target: str, model: str) -> float:
        return float(tests.query("target == @target and model == @model")["auc"].iloc[0])

    severe_amp = rho("strong_overload", "response_amplitude")
    report = f"""# Breathing phase-diagram audit

Date: 2026-06-12

## Question

The response-function audit showed that positive loop activation orders hot overload. This audit asks whether the breathing variables can be organised as a two-axis phase diagram: activation on one axis and imprint efficiency on the other. To avoid circularity, points are coloured by measured overload intensity rather than by the hazard-number bins used in the earlier response-mode summary.

## Main result

The phase map supports a bounded interpretation. Positive loop activation remains the primary coordinate for positive and moderate overload. For severe overload, defined here as response overload asinh greater than 1, the buffer-weighted hazard number is more rank-informative than activation alone (rho = {rho('strong_overload','response_hazard_number'):.3f} versus {rho('strong_overload','response_loop_activation'):.3f}), but response amplitude is also strong (rho = {severe_amp:.3f}). Imprint efficiency alone is negative for severe overload, rho = {rho('strong_overload','response_imprint_efficiency'):.3f}, consistent with the idea that poor exhalation buffers the dangerous branch less effectively. Thus the phase map explains where severe events sit in the activation-buffering plane; it does not replace the activation or amplitude hierarchy.

Within-route late-cycle forward tests are small because only three routes have complete breathing-state joins. They should be read as diagnostics, not definitive validation. In that limited test, activation gives severe-overload AUC = {auc('strong_overload','activation'):.3f}, the hazard number gives AUC = {auc('strong_overload','hazard'):.3f}, and imprint alone gives AUC = {auc('strong_overload','imprint'):.3f}.

## Interpretation allowed in the manuscript

Allowed: the most severe hot excursions occupy a high-activation, low-buffering sector of the breathing plane. This makes "breathing efficiency" operational: an inhale becomes dangerous when positive loop activation and large amplitude are not efficiently written into the next cold imprint.

Not allowed: the phase diagram does not replace the five-route force-loop mechanism, and it is not a universal phase boundary. For ordinary positive overload, loop activation remains the cleaner coordinate.

## Generated files

- `figures/nphys_fig30_breathing_phase_diagram.*`
- `source_data/nphys_breathing_phase_diagram_cycle_metrics.csv`
- `source_data/nphys_breathing_phase_diagram_correlations.csv`
- `source_data/nphys_breathing_phase_diagram_forward_tests.csv`
- `source_data/nphys_breathing_phase_diagram_route_summary.csv`
"""
    (ROOT / "nature_physics_breathing_phase_diagram.md").write_text(report, encoding="utf-8")


def main() -> None:
    df = load_data()
    corr = correlation_tests(df)
    tests = forward_logistic_tests(df)
    summary = route_state_summary(df)
    df.to_csv(SRC / "nphys_breathing_phase_diagram_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_phase_diagram_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_breathing_phase_diagram_forward_tests.csv", index=False)
    summary.to_csv(SRC / "nphys_breathing_phase_diagram_route_summary.csv", index=False)
    build_figure(df, corr, tests, summary)
    write_report(corr, tests, summary)
    print("Breathing phase diagram complete.")


if __name__ == "__main__":
    main()
