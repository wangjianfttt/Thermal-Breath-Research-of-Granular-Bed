#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E8EBEF"
MUTED = "#8A929C"
LOOP = "#B6423E"
PSI = "#C95F3F"
TAIL = "#3D6B9C"
COLD = "#7E6AAE"
ROUTE_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}

PREDICTORS = {
    "lagged Psi": ("dimensionless_loop_number", PSI),
    "lagged loop": ("loop_activation", LOOP),
    "lagged top-5% tail": ("top5_activation", TAIL),
    "lagged cold loop": ("force_h1_birth_force_share_cold", COLD),
}


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


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=INK)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_iqr_score(df: pd.DataFrame, col: str) -> pd.Series:
    center = df.groupby("regime_id")[col].transform("median")
    scale = df.groupby("regime_id")[col].transform(lambda s: np.subtract(*np.percentile(s, [75, 25])))
    return (df[col] - center) / scale.replace(0, np.nan)


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["overload_asinh"] = np.arcsinh(df["overload_number"] / 2.0)
    threshold = df.groupby("regime_id")["overload_asinh"].transform(lambda s: s.quantile(0.80))
    df["rare_overload"] = (df["overload_asinh"] >= threshold).astype(int)
    for label, (col, _) in PREDICTORS.items():
        df[f"{col}_z"] = route_iqr_score(df, col)
    parts = []
    for _, g in df.groupby("regime_id", sort=True):
        g = g.sort_values("cycle").copy()
        for _, (col, _) in PREDICTORS.items():
            for lag in range(1, 7):
                g[f"{col}_z_lag{lag}"] = g[f"{col}_z"].shift(lag)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def auc_pooled(df: pd.DataFrame, feature: str) -> float:
    d = df[["rare_overload", feature]].dropna()
    y = d["rare_overload"].to_numpy(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, d[feature].to_numpy(float)))


def route_local_auc(df: pd.DataFrame, feature: str) -> tuple[float, float, float]:
    vals = []
    for _, g in df.dropna(subset=[feature]).groupby("regime_id", sort=True):
        y = g["rare_overload"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        vals.append(float(roc_auc_score(y, g[feature].to_numpy(float))))
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def circular_shift_null_auc(df: pd.DataFrame, feature: str, n_null: int = 3000, seed: int = 61) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed + sum(ord(c) for c in feature))
    d = df[["regime_id", "cycle", "rare_overload", feature]].dropna().copy()
    observed = auc_pooled(d, feature)
    null = np.empty(n_null, dtype=float)
    groups = [(rid, g.sort_values("cycle").copy()) for rid, g in d.groupby("regime_id", sort=True)]
    for i in range(n_null):
        y_perm_parts = []
        score_parts = []
        for _, g in groups:
            y = g["rare_overload"].to_numpy(int)
            shift = int(rng.integers(0, len(y)))
            y_perm_parts.append(np.roll(y, shift))
            score_parts.append(g[feature].to_numpy(float))
        y_perm = np.concatenate(y_perm_parts)
        score = np.concatenate(score_parts)
        null[i] = roc_auc_score(y_perm, score) if len(np.unique(y_perm)) == 2 else np.nan
    p = float((1 + np.sum(np.abs(null - 0.5) >= abs(observed - 0.5))) / (len(null) + 1))
    return null, p


def lag_scan(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    null_rows = []
    for label, (col, color) in PREDICTORS.items():
        for lag in range(1, 7):
            feature = f"{col}_z_lag{lag}"
            d = df.dropna(subset=[feature]).copy()
            auc = auc_pooled(d, feature)
            mean_auc, min_auc, max_auc = route_local_auc(d, feature)
            null, p = circular_shift_null_auc(d, feature)
            for idx, value in enumerate(null):
                null_rows.append(
                    {
                        "predictor": label,
                        "lag_cycles": lag,
                        "null_index": idx,
                        "null_auc": float(value),
                        "observed_auc": auc,
                    }
                )
            rows.append(
                {
                    "predictor": label,
                    "predictor_column": col,
                    "feature": feature,
                    "lag_cycles": lag,
                    "n": int(len(d)),
                    "rare_event_count": int(d["rare_overload"].sum()),
                    "auc": auc,
                    "route_mean_auc": mean_auc,
                    "route_min_auc": min_auc,
                    "route_max_auc": max_auc,
                    "circular_shift_p": p,
                    "null_q025": float(np.nanquantile(null, 0.025)),
                    "null_q975": float(np.nanquantile(null, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def conditional_risk(df: pd.DataFrame, feature: str, predictor: str) -> pd.DataFrame:
    d = df[["regime_id", "rare_overload", feature]].dropna().copy()
    d["quintile"] = pd.qcut(d[feature], q=5, labels=False, duplicates="drop") + 1
    rows = []
    base = float(d["rare_overload"].mean())
    for q, g in d.groupby("quintile", sort=True):
        rows.append(
            {
                "predictor": predictor,
                "feature": feature,
                "quintile": int(q),
                "n": int(len(g)),
                "rare_event_count": int(g["rare_overload"].sum()),
                "rare_event_probability": float(g["rare_overload"].mean()),
                "base_rate": base,
                "risk_ratio_vs_base": float(g["rare_overload"].mean() / base) if base else np.nan,
            }
        )
    return pd.DataFrame(rows)


def route_timeline(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows = []
    for rid, g in df.dropna(subset=[feature]).groupby("regime_id", sort=True):
        cutoff = g[feature].quantile(0.80)
        out = g[["regime_id", "cycle", "rare_overload", feature, "overload_asinh"]].copy()
        out["high_precursor"] = (out[feature] >= cutoff).astype(int)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def make_figure(df: pd.DataFrame, scan: pd.DataFrame, risk: pd.DataFrame, route_auc: pd.DataFrame, timeline: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.08, 5.35))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    for label, (_, color) in PREDICTORS.items():
        d = scan[scan["predictor"] == label]
        ax.plot(d["lag_cycles"], d["auc"], marker="o", ms=3.8, lw=1.15, color=color, label=label)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("lag before overload cycle")
    ax.set_ylabel("AUC for rare overload")
    ax.set_title("Extreme-event precursors are lag-selective", loc="left", pad=4)
    finish(ax)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.35)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for label, (_, color) in PREDICTORS.items():
        d = risk[risk["predictor"] == label]
        ax.plot(d["quintile"], d["rare_event_probability"], marker="o", ms=3.8, lw=1.15, color=color, label=label)
    base = risk["base_rate"].iloc[0]
    ax.axhline(base, color=MUTED, lw=0.8, ls=(0, (3, 2)), label="base rate")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("lag-2 predictor quintile")
    ax.set_ylabel("P(rare overload two cycles later)")
    ax.set_title("Lag-2 loop state raises conditional risk", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    order = ["lagged Psi", "lagged loop", "lagged top-5% tail", "lagged cold loop"]
    colors = [PREDICTORS[o][1] for o in order]
    summary = route_auc.set_index("predictor").loc[order].reset_index()
    x = np.arange(len(order))
    ax.bar(x, summary["mean_auc"], color=colors, alpha=0.86, width=0.58)
    for i, row in summary.iterrows():
        ax.plot([i, i], [row["min_auc"], row["max_auc"]], color=INK, lw=0.9)
        vals = row["values"]
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, s=14, color="white", edgecolor=colors[i], linewidth=0.75, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(["Psi", "loop", "top-5%", "cold loop"])
    ax.set_ylabel("route-local AUC at lag 2")
    ax.set_title("Route-wise tests bound forecasting strength", loc="left", pad=4)
    finish(ax, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    routes = sorted(timeline["regime_id"].unique())
    ybase = {rid: i for i, rid in enumerate(routes)}
    for rid in routes:
        g = timeline[timeline["regime_id"] == rid]
        y = np.full(len(g), ybase[rid])
        ax.scatter(g["cycle"], y, s=18, color="#D5DADF", edgecolor="none", zorder=1)
        hp = g[g["high_precursor"] == 1]
        ax.scatter(hp["cycle"], np.full(len(hp), ybase[rid]), s=34, facecolor="none", edgecolor=LOOP, linewidth=1.0, zorder=2)
        rr = g[g["rare_overload"] == 1]
        ax.scatter(rr["cycle"], np.full(len(rr), ybase[rid]), s=22, color=LOOP, marker="s", edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels(routes)
    ax.set_xlim(1, 30.5)
    ax.set_xlabel("cycle")
    ax.set_title("High lag-2 loop states precede tail events intermittently", loc="left", pad=4)
    finish(ax, axis="x")
    h1 = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#D5DADF", markersize=5, label="cycle")
    h2 = plt.Line2D([0], [0], marker="o", color=LOOP, markerfacecolor="none", markersize=6, label="high lag-2 loop")
    h3 = plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=LOOP, markeredgecolor="white", markersize=5, label="rare overload")
    ax.legend(handles=[h1, h2, h3], loc="upper left", fontsize=5.8, handletextpad=0.3)

    fig.suptitle("Force-loop memory gives a bounded early warning for rare hot overload", x=0.02, y=0.995, ha="left", fontsize=10.2, fontweight="bold", color=INK)
    fig.savefig(FIG / "nphys_fig41_force_loop_rare_event_precursor.pdf", bbox_inches="tight")
    fig.savefig(FIG / "nphys_fig41_force_loop_rare_event_precursor.svg", bbox_inches="tight")
    fig.savefig(FIG / "nphys_fig41_force_loop_rare_event_precursor.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG / "nphys_fig41_force_loop_rare_event_precursor.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(scan: pd.DataFrame, risk: pd.DataFrame, route_auc_long: pd.DataFrame) -> None:
    lag2 = scan[(scan["lag_cycles"] == 2)].set_index("predictor")
    loop = lag2.loc["lagged loop"]
    psi = lag2.loc["lagged Psi"]
    tail = lag2.loc["lagged top-5% tail"]
    lag1_loop = scan[(scan["lag_cycles"] == 1) & (scan["predictor"] == "lagged loop")].iloc[0]
    risk_loop = risk[(risk["predictor"] == "lagged loop") & (risk["quintile"] == 5)].iloc[0]
    low_loop = risk[(risk["predictor"] == "lagged loop") & (risk["quintile"] == 1)].iloc[0]
    lines = [
        "# Force-loop rare-event precursor audit",
        "",
        "Purpose: test whether the route-local extreme hot-overload tail has a measurable force-loop precursor, without using same-cycle overload information.",
        "",
        "## Definition",
        "",
        "Rare overload is the top 20% asinh-overload tail within each route. Predictors are route-normalised and lagged by one to six cycles before the labelled overload cycle.",
        "",
        "## Main results",
        "",
        f"- Lag-2 force-loop activation predicts rare overload with AUC={loop['auc']:.3f}; route-preserving circular-shift P={loop['circular_shift_p']:.4f}.",
        f"- Lag-2 Psi gives AUC={psi['auc']:.3f}; lag-2 top-5% force-tail gives AUC={tail['auc']:.3f}.",
        f"- The top quintile of lag-2 force-loop activation has rare-event probability {risk_loop['rare_event_probability']:.3f}, compared with {low_loop['rare_event_probability']:.3f} in the bottom quintile and a base rate of {risk_loop['base_rate']:.3f}.",
        f"- Lag-1 loop activation has pooled AUC={lag1_loop['auc']:.3f} but a route-wise minimum AUC={lag1_loop['route_min_auc']:.3f}, so it is a weak alarm rather than a robust one-cycle warning.",
        "",
        "## Interpretation boundary",
        "",
        "This supports a bounded early-warning statement: force-loop memory two cycles earlier increases the risk of entering the route-local hot-overload tail. It is not a monotonic one-cycle alarm. Lag selectivity and route-to-route variability are part of the mechanism, consistent with a dissipative breathing return map rather than a universal forecasting law.",
        "",
        "## Lag scan",
        "",
        scan.to_markdown(index=False),
        "",
        "## Lag-2 conditional risk",
        "",
        risk.to_markdown(index=False),
        "",
        "## Route-wise lag-2 AUC",
        "",
        route_auc_long.to_markdown(index=False),
        "",
    ]
    (ROOT / "nature_physics_force_loop_rare_event_precursor.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    df = prepare()
    scan, nulls = lag_scan(df)

    risk_parts = []
    for label, (col, _) in PREDICTORS.items():
        risk_parts.append(conditional_risk(df, f"{col}_z_lag2", label))
    risk = pd.concat(risk_parts, ignore_index=True)

    route_rows = []
    route_long = []
    for label, (col, _) in PREDICTORS.items():
        feature = f"{col}_z_lag2"
        vals = []
        for rid, g in df.dropna(subset=[feature]).groupby("regime_id", sort=True):
            auc = roc_auc_score(g["rare_overload"].to_numpy(int), g[feature].to_numpy(float))
            vals.append(float(auc))
            route_long.append({"predictor": label, "regime_id": rid, "feature": feature, "route_local_auc": float(auc), "n": int(len(g)), "rare_event_count": int(g["rare_overload"].sum())})
        route_rows.append({"predictor": label, "mean_auc": float(np.mean(vals)), "min_auc": float(np.min(vals)), "max_auc": float(np.max(vals)), "values": vals})
    route_auc = pd.DataFrame(route_rows)
    route_long_df = pd.DataFrame(route_long)

    timeline = route_timeline(df, "loop_activation_z_lag2")

    df.to_csv(SRC / "nphys_force_loop_rare_event_precursor_cycle_metrics.csv", index=False)
    scan.to_csv(SRC / "nphys_force_loop_rare_event_precursor_lag_scan.csv", index=False)
    nulls.to_csv(SRC / "nphys_force_loop_rare_event_precursor_null.csv", index=False)
    risk.to_csv(SRC / "nphys_force_loop_rare_event_precursor_conditional_risk.csv", index=False)
    route_long_df.to_csv(SRC / "nphys_force_loop_rare_event_precursor_route_auc.csv", index=False)
    timeline.to_csv(SRC / "nphys_force_loop_rare_event_precursor_timeline.csv", index=False)

    make_figure(df, scan, risk, route_auc, timeline)
    write_report(scan, risk, route_long_df)

    print(scan[scan["lag_cycles"].isin([1, 2, 4])].to_string(index=False))
    print("\nWrote figures/nphys_fig41_force_loop_rare_event_precursor.*")


if __name__ == "__main__":
    main()
