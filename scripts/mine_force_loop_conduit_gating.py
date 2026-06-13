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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_force_loop_enrichment_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
RED = "#B6423E"
VIOLET = "#7E6AAE"
GREEN = "#4F8B67"
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


def panel(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float, int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 4 or np.nanstd(xx[ok]) == 0 or np.nanstd(yy[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    out = spearmanr(xx[ok], yy[ok])
    return float(out.statistic), float(out.pvalue), int(ok.sum())


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    if "overload_asinh" not in df:
        df["overload_asinh"] = np.arcsinh(df["overload_number"].to_numpy(float))
    df["loop_activation"] = df["force_h1_birth_force_share_hot_minus_cold"]
    df["loop_abundance_activation"] = df["force_h1_birth_fraction_hot_minus_cold"]
    df["wall_conduit_activation"] = df["bottom_side_percolation_edge_fraction_hot_minus_cold"]
    df["giant_force_backbone_activation"] = df["giant_fraction_after_top5_edges_hot_minus_cold"]
    df["top5_tail_activation"] = df["force_share_top5_edges_hot_minus_cold"]
    df["loop_conduit_product"] = df["loop_activation"] * df["wall_conduit_activation"]
    df["loop_giant_product"] = df["loop_activation"] * df["giant_force_backbone_activation"]

    for col in [
        "overload_asinh",
        "loop_activation",
        "loop_abundance_activation",
        "wall_conduit_activation",
        "giant_force_backbone_activation",
        "top5_tail_activation",
        "loop_conduit_product",
        "loop_giant_product",
    ]:
        df[col + "_rc"] = route_center(df, col)

    rare = []
    for _rid, g in df.groupby("regime_id", sort=False):
        q = g["overload_asinh"].quantile(0.80)
        rare.extend((g["overload_asinh"] >= q).to_numpy(bool))
    df["route_local_top20_overload"] = rare
    df["high_loop"] = df["loop_activation_rc"] > 0
    df["high_wall_conduit"] = df["wall_conduit_activation_rc"] > 0
    df["quadrant"] = np.select(
        [
            (~df["high_loop"]) & (~df["high_wall_conduit"]),
            (~df["high_loop"]) & df["high_wall_conduit"],
            df["high_loop"] & (~df["high_wall_conduit"]),
            df["high_loop"] & df["high_wall_conduit"],
        ],
        ["low loop / closed conduit", "conduit only", "loop only", "loop + conduit"],
        default="unclassified",
    )
    return df


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        ("loop force share", "loop_activation"),
        ("loop abundance", "loop_abundance_activation"),
        ("wall conduit", "wall_conduit_activation"),
        ("giant top-5 backbone", "giant_force_backbone_activation"),
        ("top-5 force tail", "top5_tail_activation"),
        ("loop x wall conduit", "loop_conduit_product"),
    ]
    for name, col in predictors:
        raw_r, raw_p, n = safe_spearman(df[col], df["overload_asinh"])
        rc_r, rc_p, _n = safe_spearman(df[col + "_rc"], df["overload_asinh_rc"])
        rows.append(
            {
                "predictor": name,
                "column": col,
                "n": n,
                "spearman_raw": raw_r,
                "p_raw": raw_p,
                "spearman_route_centered": rc_r,
                "p_route_centered": rc_p,
            }
        )
    return pd.DataFrame(rows)


def leave_one_route_models(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "loop force share": ["loop_activation"],
        "loop abundance": ["loop_abundance_activation"],
        "wall conduit": ["wall_conduit_activation"],
        "giant backbone": ["giant_force_backbone_activation"],
        "top-5 force tail": ["top5_tail_activation"],
        "loop + conduit": ["loop_activation", "wall_conduit_activation"],
        "loop + conduit + interaction": ["loop_activation", "wall_conduit_activation", "loop_conduit_product"],
        "loop + giant backbone": ["loop_activation", "giant_force_backbone_activation"],
    }
    rows = []
    for name, features in models.items():
        y_all: list[float] = []
        yh_all: list[float] = []
        base_all: list[float] = []
        for rid in sorted(df["regime_id"].unique()):
            train = df[df["regime_id"] != rid].dropna(subset=["overload_asinh", *features])
            test = df[df["regime_id"] == rid].dropna(subset=["overload_asinh", *features])
            scaler = StandardScaler().fit(train[features].to_numpy(float))
            x_train = scaler.transform(train[features].to_numpy(float))
            x_test = scaler.transform(test[features].to_numpy(float))
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, train["overload_asinh"].to_numpy(float))
            pred = model.predict(x_test)
            y_all.extend(test["overload_asinh"].to_numpy(float))
            yh_all.extend(pred)
            base_all.extend(np.repeat(float(train["overload_asinh"].mean()), len(test)))
        y = np.asarray(y_all, dtype=float)
        yh = np.asarray(yh_all, dtype=float)
        base = np.asarray(base_all, dtype=float)
        rho, p, _n = safe_spearman(y, yh)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": int(len(y)),
                "r2_vs_training_mean": float(1 - np.sum((y - yh) ** 2) / np.sum((y - base) ** 2)),
                "spearman_y_yhat": rho,
                "spearman_p": p,
            }
        )
    return pd.DataFrame(rows)


def conditional_risk(df: pd.DataFrame, n_perm: int = 10000, seed: int = 47) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = ["low loop / closed conduit", "conduit only", "loop only", "loop + conduit"]
    rows = []
    for q in order:
        g = df[df["quadrant"] == q]
        rows.append(
            {
                "quadrant": q,
                "n": int(len(g)),
                "rare_event_count": int(g["route_local_top20_overload"].sum()),
                "rare_event_probability": float(g["route_local_top20_overload"].mean()) if len(g) else np.nan,
                "base_rate": float(df["route_local_top20_overload"].mean()),
            }
        )
    risk = pd.DataFrame(rows)

    high_loop = df["high_loop"].to_numpy(bool)
    high_cond = df["high_wall_conduit"].to_numpy(bool)
    rare = df["route_local_top20_overload"].to_numpy(float)
    observed = float(rare[high_loop & high_cond].mean() - rare[high_loop & ~high_cond].mean())

    rng = np.random.default_rng(seed)
    null = []
    route = df["regime_id"].to_numpy(str)
    for i in range(n_perm):
        perm_cond = high_cond.copy()
        for rid in np.unique(route):
            idx = np.where(route == rid)[0]
            perm_cond[idx] = rng.permutation(perm_cond[idx])
        if np.any(high_loop & perm_cond) and np.any(high_loop & ~perm_cond):
            delta = rare[high_loop & perm_cond].mean() - rare[high_loop & ~perm_cond].mean()
            null.append(delta)
    null_arr = np.asarray(null, dtype=float)
    p = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))
    auc_rows = []
    for name, col in [
        ("loop force share", "loop_activation"),
        ("wall conduit", "wall_conduit_activation"),
        ("loop x wall conduit", "loop_conduit_product"),
        ("top-5 force tail", "top5_tail_activation"),
    ]:
        auc_rows.append(
            {
                "predictor": name,
                "auc_for_route_local_top20_overload": float(roc_auc_score(df["route_local_top20_overload"], df[col])),
            }
        )
    gate = pd.DataFrame(
        [
            {
                "test": "high_loop_wall_conduit_gate",
                "observed_risk_difference": observed,
                "routewise_conduit_shuffle_p_one_sided": p,
                "n_permutations": int(len(null_arr)),
                "null_q025": float(np.quantile(null_arr, 0.025)),
                "null_q975": float(np.quantile(null_arr, 0.975)),
            }
        ]
    )
    auc = pd.DataFrame(auc_rows)
    return risk, gate, auc


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("regime_id", as_index=False)
        .agg(
            n=("cycle", "count"),
            mean_overload_asinh=("overload_asinh", "mean"),
            mean_loop_activation=("loop_activation", "mean"),
            mean_wall_conduit=("wall_conduit_activation", "mean"),
            mean_giant_backbone=("giant_force_backbone_activation", "mean"),
            top20_overload_rate=("route_local_top20_overload", "mean"),
        )
    )


def make_figure(df: pd.DataFrame, corr: pd.DataFrame, models: pd.DataFrame, risk: pd.DataFrame, gate: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 0.92], height_ratios=[1.05, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    vmax = np.nanquantile(np.abs(df["overload_asinh_rc"]), 0.96)
    for rid, g in df.groupby("regime_id", sort=True):
        sc = ax.scatter(
            g["loop_activation_rc"],
            g["wall_conduit_activation_rc"],
            c=g["overload_asinh_rc"],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            marker=MARKERS.get(rid, "o"),
            s=np.where(g["route_local_top20_overload"], 42, 22),
            edgecolor=np.where(g["route_local_top20_overload"], INK, "white"),
            linewidth=np.where(g["route_local_top20_overload"], 0.75, 0.35),
            alpha=0.92,
            zorder=3,
        )
    ax.axhline(0, color="#AEB6C0", lw=0.75, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.75, ls=(0, (3, 3)))
    ax.text(0.97, 0.95, "loop + conduit", transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=RED)
    ax.text(0.03, 0.08, "closed / weak", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color=MUTED)
    ax.set_xlabel("route-centred force-loop activation")
    ax.set_ylabel("route-centred wall-conduit activation")
    ax.set_title("where dangerous breaths sit", loc="left", pad=4)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("route-centred overload", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    risk_order = ["low loop / closed conduit", "conduit only", "loop only", "loop + conduit"]
    rr = risk.set_index("quadrant").loc[risk_order].reset_index()
    colors = ["#B8C2CC", "#9BB7C7", GOLD, RED]
    x = np.arange(len(rr))
    ax.bar(x, rr["rare_event_probability"], color=colors, width=0.68)
    ax.axhline(rr["base_rate"].iloc[0], color="#6F7C8A", lw=0.8, ls=(0, (3, 3)))
    for i, row in rr.iterrows():
        ax.text(i, row["rare_event_probability"] + 0.025, f"n={int(row['n'])}", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    ax.set_xticks(x, ["low/closed", "conduit\nonly", "loop\nonly", "loop+\nconduit"], fontsize=6.1)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("top-20% overload probability")
    p = gate.loc[gate["test"] == "high_loop_wall_conduit_gate", "routewise_conduit_shuffle_p_one_sided"].iloc[0]
    ax.set_title(f"suggestive tail-risk gate, p={p:.3f}", loc="left", pad=4)
    finish(ax, "y")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    order = ["loop force share", "loop abundance", "giant top-5 backbone", "wall conduit", "top-5 force tail", "loop x wall conduit"]
    cc = corr.set_index("predictor").loc[order].reset_index()
    y = np.arange(len(cc))
    bar_cols = [RED, RED, VIOLET, BLUE, "#6F7C8A", GOLD]
    ax.barh(y, cc["spearman_route_centered"], color=bar_cols, alpha=0.88)
    ax.axvline(0, color="#AEB6C0", lw=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(cc["predictor"], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlim(-0.62, 0.98)
    ax.set_xlabel("route-centred Spearman with overload")
    ax.set_title("main coordinate versus gate", loc="left", pad=4)
    finish(ax, "x")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    model_order = ["loop force share", "wall conduit", "top-5 force tail", "loop + conduit", "loop + conduit + interaction"]
    mm = models.set_index("model").loc[model_order].reset_index()
    x = np.arange(len(mm))
    ax.bar(x, mm["r2_vs_training_mean"], color=[RED, BLUE, "#6F7C8A", GOLD, GOLD], width=0.68)
    ax.axhline(0, color="#AEB6C0", lw=0.75)
    ax.set_xticks(x, ["loop", "conduit", "top-5%", "loop+\nconduit", "loop+\ninteraction"], fontsize=5.8)
    ax.set_ylim(-0.16, 0.95)
    ax.set_ylabel(r"leave-route-out $R^2$")
    ax.set_title("conduit modulates, not replaces", loc="left", pad=4)
    finish(ax, "y")
    panel(ax, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig46_force_loop_conduit_gating.{ext}", **kwargs)
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    models: pd.DataFrame,
    risk: pd.DataFrame,
    gate: pd.DataFrame,
    auc: pd.DataFrame,
    routes: pd.DataFrame,
) -> None:
    def grab_corr(name: str) -> float:
        return float(corr.loc[corr["predictor"] == name, "spearman_route_centered"].iloc[0])

    def grab_model(name: str) -> float:
        return float(models.loc[models["model"] == name, "r2_vs_training_mean"].iloc[0])

    high = risk.set_index("quadrant")
    gate_row = gate.iloc[0]
    lines = [
        "# Force-loop conduit-gating audit",
        "",
        "This reserve audit asks whether the force-loop mechanism has a spatial gate: are hot overload tails largest when route-centred cycle-closing force loops coincide with an increased bottom-to-side force conduit?",
        "",
        "## Cycle-level diagnostics",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## Leave-one-route transfer",
        "",
        models.round(4).to_markdown(index=False),
        "",
        "## Conditional rare-event risk",
        "",
        risk.round(4).to_markdown(index=False),
        "",
        gate.round(4).to_markdown(index=False),
        "",
        "## Rare-event ranking AUC",
        "",
        auc.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        routes.round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"The supported mechanism remains force-loop activation: route-centred overload has rho={grab_corr('loop force share'):.3f} with loop force share and rho={grab_corr('loop abundance'):.3f} with loop abundance. The wall-conduit variable is weak by itself (rho={grab_corr('wall conduit'):.3f}) and fails as a leave-one-route overload coordinate (R2={grab_model('wall conduit'):.3f}).",
        "",
        f"The conduit gives a suggestive but not decisive rare-event gate. Within high-loop states, adding a positive wall-conduit excursion raises route-local top-20% overload probability from {high.loc['loop only', 'rare_event_probability']:.3f} to {high.loc['loop + conduit', 'rare_event_probability']:.3f}; however, a route-wise conduit shuffle gives one-sided p={gate_row['routewise_conduit_shuffle_p_one_sided']:.4f}. This should be used only as a spatial modulation hypothesis or reserve diagnostic, not as a main-text proof.",
        "",
        f"Leave-one-route transfer also enforces the boundary: loop force share alone gives R2={grab_model('loop force share'):.3f}, whereas adding conduit or an interaction gives R2={grab_model('loop + conduit + interaction'):.3f}. Thus the conduit gate explains where the dangerous branch sits in state space, while the transferable overload amplitude remains the dimensionless force-loop coordinate.",
        "",
        "Allowed wording: the dangerous hot-breath sector is enriched when many force-carrying cycle-closing paths are activated and the force subgraph also opens a wall-coupled conduit. Not allowed: the conduit is not a universal pressure law, not a standalone predictor, not statistically decisive under the present route-wise shuffle, and not evidence that a single spanning chain replaces loop activation.",
    ]
    (ROOT / "nature_physics_force_loop_conduit_gating.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = prepare()
    corr = correlations(df)
    models = leave_one_route_models(df)
    risk, gate, auc = conditional_risk(df)
    routes = route_summary(df)
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df.to_csv(SRC / "nphys_force_loop_conduit_gating_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_force_loop_conduit_gating_correlations.csv", index=False)
    models.to_csv(SRC / "nphys_force_loop_conduit_gating_transfer_tests.csv", index=False)
    risk.to_csv(SRC / "nphys_force_loop_conduit_gating_conditional_risk.csv", index=False)
    gate.to_csv(SRC / "nphys_force_loop_conduit_gating_gate_tests.csv", index=False)
    auc.to_csv(SRC / "nphys_force_loop_conduit_gating_auc.csv", index=False)
    routes.to_csv(SRC / "nphys_force_loop_conduit_gating_route_summary.csv", index=False)
    make_figure(df, corr, models, risk, gate)
    write_report(df, corr, models, risk, gate, auc, routes)
    print("Wrote force-loop conduit-gating products")
    print(corr[["predictor", "spearman_route_centered", "p_route_centered"]].round(3).to_string(index=False))
    print(models[["model", "r2_vs_training_mean", "spearman_y_yhat"]].round(3).to_string(index=False))
    print(risk.round(3).to_string(index=False))
    print(gate.round(4).to_string(index=False))
    print(auc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
