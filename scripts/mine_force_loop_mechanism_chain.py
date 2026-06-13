#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

REGIME_COLORS = {
    "R1": "#345995",
    "R3": "#D98C3A",
    "R5": "#7E6AAE",
    "R6": "#C95F3F",
    "R6c": "#9E3D34",
}
GRID = "#E7EAEE"
INK = "#252A31"
LOOP = "#B64342"
NEUTRAL = "#8B929A"


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


def panel(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def draw_mechanism_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "a", x=-0.04)

    def box(x: float, y: float, w: float, h: float, label: str, color: str, face: str = "#FFFFFF") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.8,
            edgecolor=color,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.2, color=INK)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = NEUTRAL) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                lw=0.8,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    box(0.08, 0.80, 0.32, 0.10, "thermal\ncycle", "#6F8FBF", "#F5F8FC")
    box(0.58, 0.80, 0.32, 0.10, "geometry\nrewritten", "#6F8FBF", "#F5F8FC")
    box(0.08, 0.10, 0.32, 0.10, "force\nconcentration", NEUTRAL, "#F7F7F7")
    box(0.58, 0.10, 0.32, 0.10, "hot\noverload", LOOP, "#FFF4F1")
    arrow((0.41, 0.85), (0.57, 0.85), "#6F8FBF")
    arrow((0.74, 0.78), (0.74, 0.63), LOOP)
    arrow((0.74, 0.37), (0.74, 0.23), LOOP)
    arrow((0.41, 0.15), (0.57, 0.15), NEUTRAL)

    nodes = {
        "A": (0.66, 0.59),
        "B": (0.82, 0.60),
        "C": (0.86, 0.44),
        "D": (0.70, 0.40),
        "E": (0.58, 0.51),
    }
    edges = [("A", "B", 0.95), ("B", "C", 0.85), ("C", "D", 0.75), ("D", "A", 0.65), ("A", "C", 1.0), ("E", "A", 0.35), ("E", "D", 0.30)]
    for u, v, weight in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        is_loop = (u, v) == ("A", "C")
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=LOOP if is_loop else "#6F8FBF",
            lw=0.8 + 2.0 * weight if is_loop else 0.5 + 1.2 * weight,
            alpha=0.95 if is_loop else 0.65,
            solid_capstyle="round",
        )
    for x, y in nodes.values():
        ax.add_patch(Circle((x, y), 0.016, facecolor="#FFFFFF", edgecolor=INK, lw=0.65, zorder=4))
    ax.text(0.50, 0.53, "cycle-closing\nforce edge", ha="center", va="center", fontsize=6.4, color=LOOP)
    ax.plot([0.57, 0.66], [0.53, 0.59], color=LOOP, lw=0.65)

    ax.text(0.08, 0.66, "not a scalar\npressure law", fontsize=6.6, color=NEUTRAL, ha="left")
    ax.text(0.08, 0.37, "not sufficient\nby itself", fontsize=6.6, color=NEUTRAL, ha="left")
    ax.text(0.58, 0.31, "topology-conditioned\nstress route", fontsize=6.8, color=LOOP, ha="left")


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0
    return (s - s.mean()) / std


def within_regime_center(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_within"] = out[col] - out.groupby("regime_id")[col].transform("mean")
    return out


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, resid, r2


def design_matrix(df: pd.DataFrame, predictors: list[str], regime_fe: bool = True, cycle: bool = True) -> np.ndarray:
    cols = [np.ones(len(df))]
    for p in predictors:
        cols.append(df[p].to_numpy(float))
    if cycle:
        cols.append(zscore(df["cycle"]).to_numpy(float))
    if regime_fe:
        for rid in sorted(str(v) for v in df["regime_id"].dropna().unique())[1:]:
            cols.append((df["regime_id"] == rid).astype(float).to_numpy())
    return np.column_stack(cols)


def permutation_r2_gain(df: pd.DataFrame, base: list[str], add: str, target: str, n_perm: int = 3000, seed: int = 31) -> dict[str, float | str | int]:
    X0 = design_matrix(df, base)
    _b0, _r0, r20 = ols(X0, df[target].to_numpy(float))
    X1 = design_matrix(df, base + [add])
    _b1, _r1, r21 = ols(X1, df[target].to_numpy(float))
    observed = r21 - r20
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    shuffled = df.copy()
    for i in range(n_perm):
        shuffled[add] = df.groupby("regime_id")[add].transform(lambda s: rng.permutation(s.to_numpy()))
        Xp = design_matrix(shuffled, base + [add])
        _bp, _rp, r2p = ols(Xp, shuffled[target].to_numpy(float))
        null[i] = r2p - r20
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {"target": target, "base": "+".join(base) if base else "FE+cycle", "added": add, "r2_base": r20, "r2_full": r21, "delta_r2": observed, "within_regime_permutation_p": p, "n": len(df)}


def spearman_table(df: pd.DataFrame, target: str, predictors: list[str]) -> pd.DataFrame:
    rows = []
    centered = within_regime_center(df, [target] + predictors)
    for p in predictors:
        raw = spearmanr(df[p], df[target], nan_policy="omit")
        within = spearmanr(centered[f"{p}_within"], centered[f"{target}_within"], nan_policy="omit")
        rows.append(
            {
                "target": target,
                "predictor": p,
                "spearman_raw": float(raw.statistic),
                "p_raw": float(raw.pvalue),
                "spearman_within_regime_centered": float(within.statistic),
                "p_within_regime_centered": float(within.pvalue),
                "n": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(SRC / "nphys_long_cycle_true_force_metrics.csv")
    delta = pd.read_csv(SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv")
    joined = pd.read_csv(SRC / "nphys_long_cycle_true_force_geometry_join.csv")
    return metrics, delta, joined


def build_force_loop_chain(delta: pd.DataFrame, joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = [
        "force_share_top5_edges_hot_minus_cold",
        "force_h1_birth_force_share_hot_minus_cold",
        "giant_fraction_after_top5_edges_hot_minus_cold",
        "bottom_side_percolation_edge_fraction_hot_minus_cold",
    ]
    corr = spearman_table(delta, "force_p99_hot_minus_cold", predictors)

    regression_rows = [
        permutation_r2_gain(delta, [], "force_share_top5_edges_hot_minus_cold", "force_p99_hot_minus_cold"),
        permutation_r2_gain(delta, [], "force_h1_birth_force_share_hot_minus_cold", "force_p99_hot_minus_cold"),
        permutation_r2_gain(delta, ["force_share_top5_edges_hot_minus_cold"], "force_h1_birth_force_share_hot_minus_cold", "force_p99_hot_minus_cold"),
        permutation_r2_gain(delta, ["force_h1_birth_force_share_hot_minus_cold"], "force_share_top5_edges_hot_minus_cold", "force_p99_hot_minus_cold"),
    ]
    regression = pd.DataFrame(regression_rows)

    hot = joined[joined["phase"] == "hot"].copy()
    geom_predictors = [
        "orientation_entropy",
        "cycle_birth_positive_fraction",
        "force_proxy_gini",
        "bottom_side_percolation_edge_fraction_geometry",
        "force_share_top5_edges",
        "force_h1_birth_force_share",
    ]
    geometry_corr = spearman_table(hot, "force_p99", geom_predictors)

    mediation_rows = []
    for geom in ["cycle_birth_positive_fraction", "orientation_entropy", "force_proxy_gini"]:
        mediator = "force_h1_birth_force_share"
        target = "force_p99"
        X_a = design_matrix(hot, [geom])
        beta_a, _res_a, r2_a = ols(X_a, hot[mediator].to_numpy(float))
        X_b = design_matrix(hot, [geom, mediator])
        beta_b, _res_b, r2_b = ols(X_b, hot[target].to_numpy(float))
        X_c = design_matrix(hot, [geom])
        beta_c, _res_c, r2_c = ols(X_c, hot[target].to_numpy(float))
        mediation_rows.append(
            {
                "geometry_predictor": geom,
                "mediator": mediator,
                "target": target,
                "a_geom_to_mediator": float(beta_a[1]),
                "b_mediator_to_target_cond_geom": float(beta_b[2]),
                "direct_geom_to_target_cond_mediator": float(beta_b[1]),
                "total_geom_to_target": float(beta_c[1]),
                "indirect_product_a_b": float(beta_a[1] * beta_b[2]),
                "r2_mediator_model": r2_a,
                "r2_target_with_mediator": r2_b,
                "r2_target_without_mediator": r2_c,
                "n": int(len(hot)),
            }
        )
    mediation = pd.DataFrame(mediation_rows)
    return corr, regression, geometry_corr, mediation


def build_regime_dynamics(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [
        "force_p99_hot_minus_cold",
        "force_share_top5_edges_hot_minus_cold",
        "force_h1_birth_force_share_hot_minus_cold",
        "giant_fraction_after_top5_edges_hot_minus_cold",
    ]
    for rid, group in delta.groupby("regime_id"):
        g = group.sort_values("cycle")
        row: dict[str, float | str] = {"regime_id": rid, "n_cycles": len(g)}
        for col in cols:
            x = g["cycle"].to_numpy(float)
            y = g[col].to_numpy(float)
            slope = float(np.polyfit(x, y, 1)[0])
            early = float(g[g["cycle"] <= 5][col].mean())
            late = float(g[g["cycle"] >= 26][col].mean())
            peak_cycle = int(g.loc[g[col].idxmax(), "cycle"])
            row[f"{col}_mean"] = float(np.mean(y))
            row[f"{col}_slope_per_cycle"] = slope
            row[f"{col}_early_mean"] = early
            row[f"{col}_late_mean"] = late
            row[f"{col}_late_minus_early"] = late - early
            row[f"{col}_peak_cycle"] = peak_cycle
        rows.append(row)
    return pd.DataFrame(rows)


def build_early_warning(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        "force_share_top5_edges_hot_minus_cold",
        "force_h1_birth_force_share_hot_minus_cold",
        "giant_fraction_after_top5_edges_hot_minus_cold",
    ]
    target = "force_p99_hot_minus_cold"
    for rid, group in delta.groupby("regime_id"):
        g = group.sort_values("cycle")
        early = g[g["cycle"] <= 5]
        late = g[g["cycle"] >= 26]
        row: dict[str, float | str] = {
            "regime_id": rid,
            "late_force_p99_delta": float(late[target].mean()),
            "early_force_p99_delta": float(early[target].mean()),
        }
        for p in predictors:
            row[f"early_{p}"] = float(early[p].mean())
            row[f"late_{p}"] = float(late[p].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out


def build_figure(delta: pd.DataFrame, corr: pd.DataFrame, regression: pd.DataFrame, dynamics: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    draw_mechanism_panel(ax)

    ax = fig.add_subplot(gs[0, 1])
    for rid, g in delta.groupby("regime_id"):
        color = REGIME_COLORS.get(rid, "#6F7C8A")
        ax.plot(g["cycle"], g["force_p99_hot_minus_cold"], marker="o", ms=2.2, lw=1.05, color=color)
        ax.text(g["cycle"].max() + 0.25, g["force_p99_hot_minus_cold"].iloc[-1], rid, color=color, fontsize=6.6, va="center")
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Delta f_{99}$")
    ax.set_title("route-specific overload", fontsize=7.5, pad=5)
    ax.set_xlim(0.5, 31.8)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    for rid, g in delta.groupby("regime_id"):
        color = REGIME_COLORS.get(rid, "#6F7C8A")
        ax.plot(g["cycle"], g["force_h1_birth_force_share_hot_minus_cold"], marker="o", ms=2.2, lw=1.05, color=color)
        ax.text(g["cycle"].max() + 0.25, g["force_h1_birth_force_share_hot_minus_cold"].iloc[-1], rid, color=color, fontsize=6.6, va="center")
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Delta$ force-loop share")
    ax.set_title("loop activation memory", fontsize=7.5, pad=5)
    ax.set_xlim(0.5, 31.8)
    finish(ax)
    panel(ax, "e")

    ax = fig.add_subplot(gs[0, 2])
    for rid, g in delta.groupby("regime_id"):
        ax.scatter(g["force_h1_birth_force_share_hot_minus_cold"], g["force_p99_hot_minus_cold"], s=19, color=REGIME_COLORS.get(rid, "#6F7C8A"), edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.axvline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel(r"$\Delta$ force-loop share")
    ax.set_ylabel(r"$\Delta f_{99}$")
    rho = corr.loc[corr["predictor"] == "force_h1_birth_force_share_hot_minus_cold", "spearman_within_regime_centered"].iloc[0]
    ax.text(0.03, 0.93, rf"within-route $\rho={rho:.2f}$", transform=ax.transAxes, fontsize=6.4, color=LOOP, ha="left", va="top")
    ax.set_title("loop-overload link", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    for rid, g in delta.groupby("regime_id"):
        ax.scatter(g["force_share_top5_edges_hot_minus_cold"], g["force_p99_hot_minus_cold"], s=19, color=REGIME_COLORS.get(rid, "#6F7C8A"), edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.axvline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel(r"$\Delta$ top-5% force share")
    ax.set_ylabel(r"$\Delta f_{99}$")
    rho = corr.loc[corr["predictor"] == "force_share_top5_edges_hot_minus_cold", "spearman_within_regime_centered"].iloc[0]
    ax.text(0.03, 0.93, rf"within-route $\rho={rho:.2f}$", transform=ax.transAxes, fontsize=6.4, color=NEUTRAL, ha="left", va="top")
    ax.set_title("force concentration control", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "f")

    inset = fig.add_subplot(gs[1, 0])
    rows = [
        regression[(regression["base"] == "FE+cycle") & (regression["added"] == "force_share_top5_edges_hot_minus_cold")].iloc[0],
        regression[(regression["base"] == "FE+cycle") & (regression["added"] == "force_h1_birth_force_share_hot_minus_cold")].iloc[0],
        regression[(regression["base"] == "force_h1_birth_force_share_hot_minus_cold") & (regression["added"] == "force_share_top5_edges_hot_minus_cold")].iloc[0],
    ]
    labels = ["top-5%\nshare", "force\nloops", "top-5%\nafter loops"]
    vals = [r["delta_r2"] for r in rows]
    colors = [NEUTRAL, LOOP, "#D8A096"]
    inset.bar(np.arange(len(vals)), vals, color=colors, width=0.62)
    inset.set_xticks(np.arange(len(vals)), labels)
    inset.tick_params(axis="x", labelsize=6.2)
    inset.set_ylim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        inset.text(i, v + max(vals) * 0.035, f"{v:.3f}", ha="center", va="bottom", fontsize=6.0, color=colors[i])
    inset.set_ylabel(r"$\Delta R^2$")
    inset.set_title("added explanatory power", fontsize=7.5, pad=5)
    finish(inset, "y")
    panel(inset, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig9_force_loop_mechanism_chain.{ext}", **kwargs)
    plt.close(fig)


def write_report(corr: pd.DataFrame, regression: pd.DataFrame, geometry_corr: pd.DataFrame, mediation: pd.DataFrame, dynamics: pd.DataFrame, early_warning: pd.DataFrame) -> None:
    loop_row = corr[corr["predictor"] == "force_h1_birth_force_share_hot_minus_cold"].iloc[0]
    top_row = corr[corr["predictor"] == "force_share_top5_edges_hot_minus_cold"].iloc[0]
    gain_loop = regression[regression["added"] == "force_h1_birth_force_share_hot_minus_cold"].iloc[0]
    gain_top = regression[regression["added"] == "force_share_top5_edges_hot_minus_cold"].iloc[0]
    r6 = dynamics[dynamics["regime_id"] == "R6"].iloc[0]
    n_routes = int(dynamics["regime_id"].nunique())
    n_cycles = int(dynamics["n_cycles"].sum())
    optional_lines = []
    if "R5" in set(dynamics["regime_id"]):
        r5 = dynamics[dynamics["regime_id"] == "R5"].iloc[0]
        optional_lines.append(f"- R5: high expansion with intermediate friction has mean overload {r5['force_p99_hot_minus_cold_mean']:.4g} and mean force-loop activation {r5['force_h1_birth_force_share_hot_minus_cold_mean']:.4g}, testing whether expansion alone is sufficient.")
    if "R6c" in set(dynamics["regime_id"]):
        r6c = dynamics[dynamics["regime_id"] == "R6c"].iloc[0]
        optional_lines.append(f"- R6c: the closed high-friction route has mean overload {r6c['force_p99_hot_minus_cold_mean']:.4g} and mean force-loop activation {r6c['force_h1_birth_force_share_hot_minus_cold_mean']:.4g}, isolating the effect of lid clearance in the high-friction sector.")
    lines = [
        "# Force-loop mechanism chain",
        "",
        "## Question",
        "",
        "Does thermal cycling merely concentrate contact forces, or does it activate graph-cycle force loops that convert geometric rewriting into overload?",
        "",
        "## Evidence from completed long-cycle true-force rerun",
        "",
        f"- Data status: {n_routes} targeted routes, {n_cycles} complete hot/cold cycle pairs, {2 * n_cycles} true pair-force states.",
        f"- Within-regime-centered Spearman for force-loop activation vs overload: rho = {loop_row['spearman_within_regime_centered']:.2f}.",
        f"- Within-regime-centered Spearman for top-5% force concentration vs overload: rho = {top_row['spearman_within_regime_centered']:.2f}.",
        f"- Added R2 over regime fixed effects plus cycle trend: force-loop activation {gain_loop['delta_r2']:.3f}, top-5% concentration {gain_top['delta_r2']:.3f}.",
        f"- R6 late-minus-early overload change: {r6['force_p99_hot_minus_cold_late_minus_early']:.4g}; R6 force-loop late-minus-early change: {r6['force_h1_birth_force_share_hot_minus_cold_late_minus_early']:.4g}.",
        "",
        "## Mechanistic reading",
        "",
        "The all-cycle true-force rerun separates two stress-sector modes. Top-5% force concentration is strongest in the stable/low-expansion route and does not by itself imply overload. Overload is instead aligned with force carried by cycle-closing edges during the force-ordered filtration. In graph terms, the dangerous state is not a single spanning force chain, but the activation of high-force redundant loops that can store and release stress under thermal cycling.",
        "",
        "## Regime-specific interpretation",
        "",
        "- R1: hot states concentrate force into fewer contacts, but force-loop share is reduced and the p99 tail is lower than cold. This is a concentrated-but-buffered memory state.",
        "- R3: moderate positive p99 and loop activation indicate a transitional route where geometric rewriting starts to enter the force network.",
        "- R6: large early overload and positive force-loop activation identify the high-expansion/high-friction route as loop-mediated overload. The late-cycle decay means the extreme response is partly transient, not a simple monotonic ratchet.",
        *optional_lines,
        "",
        "## What remains conservative",
        "",
        "The analysis supports a force-loop activation mechanism, but not a universal force-percolation threshold. It should be framed as a topology-conditioned stress-memory route, with regime-resolved dynamics shown explicitly.",
        "",
    ]
    (ROOT / "nature_physics_force_loop_mechanism_chain.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    _metrics, delta, joined = load_data()
    corr, regression, geometry_corr, mediation = build_force_loop_chain(delta, joined)
    dynamics = build_regime_dynamics(delta)
    early_warning = build_early_warning(delta)

    corr.to_csv(SRC / "nphys_force_loop_chain_correlations.csv", index=False)
    regression.to_csv(SRC / "nphys_force_loop_chain_regression_gain.csv", index=False)
    geometry_corr.to_csv(SRC / "nphys_force_loop_chain_geometry_correlations.csv", index=False)
    mediation.to_csv(SRC / "nphys_force_loop_chain_mediation.csv", index=False)
    dynamics.to_csv(SRC / "nphys_force_loop_chain_regime_dynamics.csv", index=False)
    early_warning.to_csv(SRC / "nphys_force_loop_chain_early_late.csv", index=False)

    build_figure(delta, corr, regression, dynamics)
    write_report(corr, regression, geometry_corr, mediation, dynamics, early_warning)
    print("correlations")
    print(corr)
    print("regression gains")
    print(regression)
    print("dynamics")
    print(dynamics[["regime_id", "force_p99_hot_minus_cold_mean", "force_h1_birth_force_share_hot_minus_cold_mean"]])


if __name__ == "__main__":
    main()
