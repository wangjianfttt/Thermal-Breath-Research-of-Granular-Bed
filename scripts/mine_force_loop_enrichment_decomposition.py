#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E7EAEE"
RED = "#B6423E"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
GREEN = "#4F8B67"
VIOLET = "#7E6AAE"
ROUTE_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
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


def route_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def safe_spearman(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> tuple[float, float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 4 or np.nanstd(xx[ok]) == 0 or np.nanstd(yy[ok]) == 0:
        return np.nan, np.nan
    out = spearmanr(xx[ok], yy[ok])
    return float(out.statistic), float(out.pvalue)


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    eps = 1e-9
    df["loop_abundance_cold"] = df["force_h1_birth_fraction_cold"].clip(lower=eps)
    df["loop_abundance_hot"] = df["force_h1_birth_fraction_hot"].clip(lower=eps)
    df["loop_force_enrichment_cold"] = df["force_h1_birth_force_share_cold"] / df["loop_abundance_cold"]
    df["loop_force_enrichment_hot"] = df["force_h1_birth_force_share_hot"] / df["loop_abundance_hot"]
    df["loop_abundance_delta"] = df["loop_abundance_hot"] - df["loop_abundance_cold"]
    df["loop_enrichment_delta"] = df["loop_force_enrichment_hot"] - df["loop_force_enrichment_cold"]
    df["loop_force_share_delta"] = df["force_h1_birth_force_share_hot_minus_cold"]
    df["log_loop_enrichment_ratio"] = np.log(
        (df["loop_force_enrichment_hot"].clip(lower=eps)) / (df["loop_force_enrichment_cold"].clip(lower=eps))
    )
    df["top5_delta"] = df["force_share_top5_edges_hot_minus_cold"]
    df["overload_asinh"] = np.arcsinh(df["overload_number"].to_numpy(float))
    for col in [
        "loop_abundance_delta",
        "loop_enrichment_delta",
        "log_loop_enrichment_ratio",
        "loop_force_share_delta",
        "top5_delta",
        "overload_asinh",
    ]:
        df[col + "_rc"] = route_center(df, col)
    return df


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        ("loop abundance change", "loop_abundance_delta"),
        ("loop-force enrichment change", "loop_enrichment_delta"),
        ("log loop-enrichment ratio", "log_loop_enrichment_ratio"),
        ("loop force-share change", "loop_force_share_delta"),
        ("top-5% force-tail change", "top5_delta"),
    ]
    for name, col in predictors:
        raw_r, raw_p = safe_spearman(df[col], df["overload_asinh"])
        rc_r, rc_p = safe_spearman(df[col + "_rc"], df["overload_asinh_rc"])
        rows.append(
            {
                "predictor": name,
                "n": int(df[[col, "overload_asinh"]].dropna().shape[0]),
                "spearman_raw": raw_r,
                "p_raw": raw_p,
                "spearman_route_centered": rc_r,
                "p_route_centered": rc_p,
            }
        )
    return pd.DataFrame(rows)


def leave_one_route_models(df: pd.DataFrame) -> pd.DataFrame:
    target = "overload_asinh"
    models = {
        "abundance only": ["loop_abundance_delta"],
        "enrichment only": ["loop_enrichment_delta"],
        "enrichment ratio": ["log_loop_enrichment_ratio"],
        "force share": ["loop_force_share_delta"],
        "abundance + enrichment": ["loop_abundance_delta", "loop_enrichment_delta"],
        "top5 tail": ["top5_delta"],
        "top5 + enrichment": ["top5_delta", "loop_enrichment_delta"],
    }
    rows = []
    for name, features in models.items():
        y_all = []
        yh_all = []
        base_all = []
        for rid in sorted(df["regime_id"].unique()):
            train = df[df["regime_id"] != rid].dropna(subset=[target, *features]).copy()
            test = df[df["regime_id"] == rid].dropna(subset=[target, *features]).copy()
            scaler = StandardScaler().fit(train[features].to_numpy(float))
            x_train = scaler.transform(train[features].to_numpy(float))
            x_test = scaler.transform(test[features].to_numpy(float))
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, train[target].to_numpy(float))
            yh = model.predict(x_test)
            y = test[target].to_numpy(float)
            y_all.extend(y)
            yh_all.extend(yh)
            base_all.extend(np.repeat(float(train[target].mean()), len(test)))
        y = np.asarray(y_all)
        yh = np.asarray(yh_all)
        base = np.asarray(base_all)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": int(len(y)),
                "r2_vs_training_mean": 1 - np.sum((y - yh) ** 2) / np.sum((y - base) ** 2),
                "spearman_y_yhat": safe_spearman(y, yh)[0],
            }
        )
    return pd.DataFrame(rows)


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("regime_id", as_index=False)
        .agg(
            n=("cycle", "count"),
            mean_overload_asinh=("overload_asinh", "mean"),
            mean_loop_abundance_delta=("loop_abundance_delta", "mean"),
            mean_loop_enrichment_delta=("loop_enrichment_delta", "mean"),
            mean_log_enrichment_ratio=("log_loop_enrichment_ratio", "mean"),
            mean_loop_force_share_delta=("loop_force_share_delta", "mean"),
            mean_top5_delta=("top5_delta", "mean"),
        )
    )


def make_figure(df: pd.DataFrame, corr: pd.DataFrame, models: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.0, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["loop_abundance_delta"],
            g["loop_enrichment_delta"],
            c=g["overload_asinh"],
            cmap="magma",
            vmin=df["overload_asinh"].quantile(0.05),
            vmax=df["overload_asinh"].quantile(0.95),
            marker=MARKERS.get(rid, "o"),
            s=25,
            edgecolor="white",
            lw=0.35,
            alpha=0.90,
            zorder=3,
        )
    im = ax.scatter([], [], c=[], cmap="magma", vmin=df["overload_asinh"].quantile(0.05), vmax=df["overload_asinh"].quantile(0.95))
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("asinh overload", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("change in loop abundance")
    ax.set_ylabel("change in loop-force enrichment")
    ax.set_title("topology count versus force loading", loc="left", pad=4)
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    order = [
        "loop abundance change",
        "loop-force enrichment change",
        "log loop-enrichment ratio",
        "loop force-share change",
        "top-5% force-tail change",
    ]
    g = corr.set_index("predictor").loc[order].reset_index()
    y = np.arange(len(g))
    colors = [BLUE, RED, RED, GOLD, VIOLET]
    ax.barh(y, g["spearman_route_centered"], color=colors, alpha=0.88)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(g["predictor"], fontsize=6.0)
    ax.set_xlabel("route-centred Spearman")
    ax.set_title("which component carries overload?", loc="left", pad=4)
    finish(ax, axis="x")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    s = summary.sort_values("mean_overload_asinh")
    for _, row in s.iterrows():
        rid = row["regime_id"]
        ax.scatter(
            row["mean_loop_abundance_delta"],
            row["mean_overload_asinh"],
            s=44,
            color=ROUTE_COLORS.get(rid, INK),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            lw=0.45,
            zorder=3,
        )
        ax.text(row["mean_loop_abundance_delta"] + 0.006, row["mean_overload_asinh"], rid, fontsize=6.2, color=ROUTE_COLORS.get(rid, INK), va="center")
    ax.set_xlabel("mean change in loop abundance")
    ax.set_ylabel("mean asinh overload")
    ax.set_title("route severity follows loop formation", loc="left", pad=4)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    m = models.sort_values("r2_vs_training_mean")
    y = np.arange(len(m))
    colors = [VIOLET if "top5" in model else RED if "enrichment" in model else BLUE for model in m["model"]]
    ax.barh(y, m["r2_vs_training_mean"], color=colors, alpha=0.88)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(m["model"], fontsize=6.0)
    ax.set_xlabel(r"leave-route $R^2$")
    ax.set_title("cross-route transfer", loc="left", pad=4)
    finish(ax, axis="x")
    panel(ax, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig43_force_loop_enrichment.{ext}", **kwargs)
    plt.close(fig)


def write_report(corr: pd.DataFrame, models: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Force-loop enrichment decomposition",
        "",
        "This audit asks whether force loops are dangerous because more cycle-closing contacts appear, or because existing cycle-closing contacts become selectively force-loaded.",
        "",
        "## Correlations",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## Leave-one-route transfer",
        "",
        models.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Mechanistic reading",
        "",
        "The decomposition favours loop formation and total loop-force share over per-loop force enrichment. Route-centred overload tracks loop-abundance change and loop-force-share change more strongly than the enrichment-only variables, and the enrichment terms are negatively signed. The dangerous inhale should therefore be described as the creation of many cycle-closing load paths that collectively carry force, not as selective loading of a fixed set of loops.",
        "",
        "Interpretation boundary: loop-force enrichment is a diagnostic decomposition of the force-loop coordinate. It does not replace the dimensionless loop number because route severity and boundary susceptibility still set the gain.",
    ]
    (ROOT / "nature_physics_force_loop_enrichment_decomposition.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare()
    corr = correlations(df)
    models = leave_one_route_models(df)
    summary = route_summary(df)
    df.to_csv(SRC / "nphys_force_loop_enrichment_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_force_loop_enrichment_correlations.csv", index=False)
    models.to_csv(SRC / "nphys_force_loop_enrichment_transfer_tests.csv", index=False)
    summary.to_csv(SRC / "nphys_force_loop_enrichment_route_summary.csv", index=False)
    make_figure(df, corr, models, summary)
    write_report(corr, models, summary)
    print("Wrote force-loop enrichment decomposition products")
    print(corr.round(3).to_string(index=False))
    print(models.sort_values("r2_vs_training_mean", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
