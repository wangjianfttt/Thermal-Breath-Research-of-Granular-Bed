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


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INFILE = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"

COLORS = {"R1": "#3D6B9C", "R3": "#D98C3A", "R5": "#6BAFB0", "R6": "#B6423E", "R6c": "#7E6AAE"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
GRID = "#E7EAEE"
ACCENT = "#B6423E"
MUTED = "#8B929A"


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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_centered_spearman(df: pd.DataFrame, x: str, y: str) -> float:
    xx = df[x] - df.groupby("regime_id")[x].transform("mean")
    yy = df[y] - df.groupby("regime_id")[y].transform("mean")
    return float(spearmanr(xx, yy).statistic)


def route_centered_rho_xy(x: np.ndarray, y: np.ndarray, route_codes: np.ndarray) -> float:
    xc = x.copy().astype(float)
    yc = y.copy().astype(float)
    for code in np.unique(route_codes):
        mask = route_codes == code
        xc[mask] -= np.nanmean(xc[mask])
        yc[mask] -= np.nanmean(yc[mask])
    return float(spearmanr(xc, yc).statistic)


def load_table() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["overload_asinh"] = np.arcsinh(df["overload_number"] / 2.0)
    df["overload_severe"] = df["overload_asinh"] > 1.0
    parts = []
    for rid, g in df.groupby("regime_id", sort=True):
        g = g.sort_values("cycle").copy()
        for col in [
            "dimensionless_loop_number",
            "loop_activation",
            "top5_activation",
            "force_h1_birth_force_share_cold",
            "force_h1_birth_fraction_cold",
            "force_p99_cold",
        ]:
            for lag in range(1, 7):
                g[f"{col}_lag{lag}"] = g[col].shift(lag)
            g[f"{col}_prev3_mean"] = g[col].shift(1).rolling(3, min_periods=2).mean()
            g[f"{col}_prev3_std"] = g[col].shift(1).rolling(3, min_periods=2).std()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def lag_join(df: pd.DataFrame, predictor: str, lag: int) -> pd.DataFrame:
    target = df[["regime_id", "cycle", "overload_asinh"]].copy()
    pred = df[["regime_id", "cycle", predictor]].copy()
    pred["cycle"] = pred["cycle"] + lag
    pred = pred.rename(columns={predictor: "predictor_value"})
    return target.merge(pred, on=["regime_id", "cycle"], how="inner").dropna()


def circular_shift_null(df: pd.DataFrame, predictor: str, lag: int, n_null: int = 1200, seed: int = 21) -> tuple[np.ndarray, float, float, float]:
    rng = np.random.default_rng(seed + 37 * lag + len(predictor))
    observed_df = lag_join(df, predictor, lag)
    observed = route_centered_spearman(observed_df, "predictor_value", "overload_asinh")

    null = np.empty(n_null, dtype=float)
    groups = [(rid, g.sort_values("cycle").copy()) for rid, g in df.groupby("regime_id", sort=True)]
    for i in range(n_null):
        xs: list[float] = []
        ys: list[float] = []
        rs: list[str] = []
        for rid, g in groups:
            shift = int(rng.integers(0, len(g)))
            shifted = np.roll(g[predictor].to_numpy(float), shift)
            pred = g[["regime_id", "cycle"]].copy()
            pred["predictor_value"] = shifted
            pred["cycle"] = pred["cycle"] + lag
            target = g[["regime_id", "cycle", "overload_asinh"]].copy()
            joined = target.merge(pred, on=["regime_id", "cycle"], how="inner").dropna()
            xs.extend(joined["predictor_value"].to_numpy(float))
            ys.extend(joined["overload_asinh"].to_numpy(float))
            rs.extend(joined["regime_id"].astype(str).tolist())
        null[i] = route_centered_rho_xy(np.asarray(xs), np.asarray(ys), np.asarray(rs))
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return null, observed, p, float(len(observed_df))


def build_lag_scan(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag_path = SRC / "nphys_force_loop_precursor_lag_scan.csv"
    null_path = SRC / "nphys_force_loop_precursor_circular_shift_null.csv"
    if lag_path.exists() and null_path.exists():
        return pd.read_csv(lag_path), pd.read_csv(null_path)

    predictors = [
        ("dimensionless_loop_number", "Psi"),
        ("loop_activation", "loop activation"),
        ("top5_activation", "top-5% tail"),
        ("force_h1_birth_force_share_cold", "cold loop share"),
    ]
    rows = []
    null_rows = []
    for pred, label in predictors:
        for lag in range(0, 7):
            joined = lag_join(df, pred, lag) if lag else df[["regime_id", "cycle", "overload_asinh", pred]].rename(columns={pred: "predictor_value"}).dropna()
            rho = route_centered_spearman(joined, "predictor_value", "overload_asinh")
            p_shift = np.nan
            q025 = np.nan
            q975 = np.nan
            if lag > 0:
                null, rho, p_shift, n = circular_shift_null(df, pred, lag)
                q025 = float(np.quantile(null, 0.025))
                q975 = float(np.quantile(null, 0.975))
                for idx, value in enumerate(null):
                    null_rows.append({"predictor": label, "lag": lag, "null_index": idx, "rho_null": value})
            else:
                n = float(len(joined))
            rows.append(
                {
                    "predictor": label,
                    "predictor_column": pred,
                    "lag_cycles": lag,
                    "n": int(n),
                    "route_centered_spearman": rho,
                    "circular_shift_p": p_shift,
                    "null_q025": q025,
                    "null_q975": q975,
                }
            )
    lag_scan = pd.DataFrame(rows)
    null_df = pd.DataFrame(null_rows)
    lag_scan.to_csv(lag_path, index=False)
    null_df.to_csv(null_path, index=False)
    return lag_scan, null_df


def forward_tests(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "same-cycle Psi": ["dimensionless_loop_number"],
        "lag-1 Psi": ["dimensionless_loop_number_lag1"],
        "lag-2 Psi": ["dimensionless_loop_number_lag2"],
        "prev-3 Psi mean": ["dimensionless_loop_number_prev3_mean"],
        "lag-2 loop": ["loop_activation_lag2"],
        "lag-2 top-5 tail": ["top5_activation_lag2"],
        "lag-2 cold loop share": ["force_h1_birth_force_share_cold_lag2"],
    }
    rows = []
    prediction_rows = []
    for name, features in models.items():
        obs: list[float] = []
        pred_all: list[float] = []
        baseline_all: list[float] = []
        for rid, g in df.groupby("regime_id", sort=True):
            d = g.dropna(subset=features + ["overload_asinh"]).copy()
            train = d[d["cycle"] <= 18]
            test = d[d["cycle"] > 18]
            if len(train) < 5 or len(test) < 5:
                continue
            x_train = train[features].to_numpy(float)
            x_test = test[features].to_numpy(float)
            y_train = train["overload_asinh"].to_numpy(float)
            y_test = test["overload_asinh"].to_numpy(float)
            model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]).fit(x_train, y_train)
            yhat = model.predict(x_test)
            obs.extend(y_test)
            pred_all.extend(yhat)
            baseline_all.extend(np.repeat(float(y_train.mean()), len(y_test)))
            for cyc, yy, pp in zip(test["cycle"], y_test, yhat):
                prediction_rows.append({"model": name, "regime_id": rid, "cycle": int(cyc), "observed": float(yy), "predicted": float(pp)})
        y = np.asarray(obs)
        yhat = np.asarray(pred_all)
        base = np.asarray(baseline_all)
        sse = float(np.sum((y - yhat) ** 2))
        sse0 = float(np.sum((y - base) ** 2))
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "within_route_forward_60_40",
                "n": int(len(y)),
                "r2_vs_training_mean": 1.0 - sse / sse0 if sse0 > 0 else np.nan,
                "spearman_observed_predicted": float(spearmanr(y, yhat).statistic),
            }
        )
    tests = pd.DataFrame(rows)
    preds = pd.DataFrame(prediction_rows)
    tests.to_csv(SRC / "nphys_force_loop_precursor_forward_tests.csv", index=False)
    preds.to_csv(SRC / "nphys_force_loop_precursor_predictions.csv", index=False)
    return tests


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("regime_id", sort=True):
        rows.append(
            {
                "regime_id": rid,
                "n": len(g),
                "mean_overload_asinh": float(g["overload_asinh"].mean()),
                "mean_dimensionless_loop_number": float(g["dimensionless_loop_number"].mean()),
                "lag2_psi_to_overload_rho": route_centered_spearman(g.dropna(subset=["dimensionless_loop_number_lag2"]), "dimensionless_loop_number_lag2", "overload_asinh"),
                "lag2_loop_to_overload_rho": route_centered_spearman(g.dropna(subset=["loop_activation_lag2"]), "loop_activation_lag2", "overload_asinh"),
                "prev3_psi_mean": float(g["dimensionless_loop_number_prev3_mean"].mean()),
                "prev3_psi_std": float(g["dimensionless_loop_number_prev3_mean"].std()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "nphys_force_loop_precursor_route_summary.csv", index=False)
    return out


def draw_figure(df: pd.DataFrame, lag_scan: pd.DataFrame, forward: pd.DataFrame, route: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.45, 4.95), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.22, 1.0, 1.04], height_ratios=[1, 1], wspace=0.28, hspace=0.36)

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", x=-0.11)
    for rid, g in df.groupby("regime_id", sort=True):
        color = COLORS.get(rid, MUTED)
        ax.plot(g["cycle"], g["dimensionless_loop_number"], color=color, lw=1.0, alpha=0.95, label=rid)
        severe = g["overload_asinh"] > 1.0
        ax.scatter(g.loc[severe, "cycle"], g.loc[severe, "dimensionless_loop_number"], s=28, marker=MARKERS.get(rid, "o"), color=color, edgecolor="white", lw=0.4, zorder=4)
    ax.axhline(0, color="#B8BFC7", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"dimensionless loop number, $\Psi$")
    ax.set_title("route-periodic loop memory", loc="left", pad=4)
    ax.text(
        0.11,
        0.94,
        "filled markers: severe hot overload",
        transform=ax.transAxes,
        va="top",
        fontsize=6.1,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2),
        zorder=10,
    )
    ax.legend(loc="lower right", ncol=2, fontsize=5.8, handlelength=1.2, columnspacing=0.7)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for pred, color in [("Psi", ACCENT), ("loop activation", "#D98C3A"), ("top-5% tail", "#6F7C8A")]:
        g = lag_scan[lag_scan["predictor"].eq(pred)].copy()
        ax.plot(g["lag_cycles"], g["route_centered_spearman"], "o-", ms=3.2, lw=1.0, color=color, label=pred)
        if pred == "Psi":
            ax.fill_between(g["lag_cycles"], g["null_q025"], g["null_q975"], color=color, alpha=0.12, lw=0)
    ax.axhline(0, color="#B8BFC7", lw=0.65)
    ax.set_xlabel("predictor lag (cycles before overload)")
    ax.set_ylabel(r"route-centred $\rho$")
    ax.set_title("lag scan with circular-shift null", loc="left", pad=4)
    ax.legend(loc="lower right", fontsize=5.8)
    finish(ax)

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    sub = lag_scan[lag_scan["predictor"].isin(["Psi", "loop activation", "top-5% tail"]) & lag_scan["lag_cycles"].isin([1, 2, 4])].copy()
    order_pairs = [
        ("Psi", 1),
        ("Psi", 2),
        ("Psi", 4),
        ("loop activation", 2),
        ("loop activation", 4),
        ("top-5% tail", 2),
        ("top-5% tail", 4),
    ]
    sub["order"] = sub.apply(lambda r: order_pairs.index((r["predictor"], int(r["lag_cycles"]))) if (r["predictor"], int(r["lag_cycles"])) in order_pairs else 99, axis=1)
    sub = sub[sub["order"] < 99].sort_values("order").reset_index(drop=True)
    sub["label"] = [r"$\Psi_1$", r"$\Psi_2$", r"$\Psi_4$", r"$L_2$", r"$L_4$", r"$q_{5,2}$", r"$q_{5,4}$"]
    x = np.arange(len(sub))
    colors = [ACCENT if (p in ["Psi", "loop activation"] and l in [2, 4]) else MUTED for p, l in zip(sub["predictor"], sub["lag_cycles"])]
    ax.bar(x, -np.log10(sub["circular_shift_p"].clip(lower=1 / 1200)), color=colors, width=0.66)
    ax.axhline(-np.log10(0.05), color="#B8BFC7", lw=0.65, ls=(0, (3, 3)))
    ax.set_xticks(x, sub["label"])
    ax.set_ylabel(r"$-\log_{10}(P_{\rm shift})$")
    ax.set_title("lags exceeding route-periodic null", loc="left", pad=4)
    ax.text(0.02, 0.90, r"subscript = lag", transform=ax.transAxes, va="top", fontsize=5.8)
    finish(ax, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    order = ["same-cycle Psi", "lag-1 Psi", "lag-2 Psi", "prev-3 Psi mean", "lag-2 loop", "lag-2 top-5 tail", "lag-2 cold loop share"]
    f = forward.set_index("model").loc[order].reset_index()
    y = np.arange(len(f))
    ax.barh(y, f["r2_vs_training_mean"], color=[ACCENT if "Psi" in m or "loop" in m and "cold" not in m else MUTED for m in f["model"]], height=0.55)
    ax.axvline(0, color="#B8BFC7", lw=0.65)
    ax.set_yticks(y, f["model"])
    ax.invert_yaxis()
    ax.set_xlabel(r"late-cycle forward $R^2$")
    ax.set_title("late-cycle transfer", loc="left", pad=4)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    for _, row in route.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["mean_dimensionless_loop_number"], row["lag2_loop_to_overload_rho"], s=50, color=COLORS.get(rid, MUTED), marker=MARKERS.get(rid, "o"), edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["mean_dimensionless_loop_number"] + 0.002, row["lag2_loop_to_overload_rho"], rid, color=COLORS.get(rid, MUTED), fontsize=6.2, va="center")
    ax.axhline(0, color="#B8BFC7", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#B8BFC7", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel(r"mean $\Psi$")
    ax.set_ylabel(r"within-route lag-2 loop $\rho$")
    ax.set_title("route-conditioned precursor", loc="left", pad=4)
    finish(ax)

    out = FIG / "nphys_fig35_force_loop_precursor"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(lag_scan: pd.DataFrame, forward: pd.DataFrame, route: pd.DataFrame) -> None:
    psi2 = lag_scan[(lag_scan["predictor"].eq("Psi")) & (lag_scan["lag_cycles"].eq(2))].iloc[0]
    loop2 = lag_scan[(lag_scan["predictor"].eq("loop activation")) & (lag_scan["lag_cycles"].eq(2))].iloc[0]
    psi1 = lag_scan[(lag_scan["predictor"].eq("Psi")) & (lag_scan["lag_cycles"].eq(1))].iloc[0]
    same = forward[forward["model"].eq("same-cycle Psi")].iloc[0]
    lag2 = forward[forward["model"].eq("lag-2 Psi")].iloc[0]
    top5 = forward[forward["model"].eq("lag-2 top-5 tail")].iloc[0]
    lines = [
        "# Force-loop precursor audit",
        "",
        "Date: 2026-06-12",
        "",
        "## Question",
        "",
        "This audit asks whether hot overload is preceded by measurable structure in the force-loop sector. The analysis uses the existing five-route true-force long-cycle data and tests lagged loop variables against circular-shift nulls that preserve route-level cyclic structure.",
        "",
        "## Main result",
        "",
        f"The same-cycle loop coordinate remains the strongest overload coordinate, but it is not the whole story. The dimensionless loop number two cycles earlier predicts route-centred overload with rho = {psi2.route_centered_spearman:.3f} and exceeds a within-route circular-shift null (P = {psi2.circular_shift_p:.3g}); lag-2 raw loop activation gives rho = {loop2.route_centered_spearman:.3f} with P = {loop2.circular_shift_p:.3g}. By contrast, lag-1 Psi is not significant against the same null (rho = {psi1.route_centered_spearman:.3f}, P = {psi1.circular_shift_p:.3g}).",
        "",
        f"Late-cycle forward tests show the same hierarchy but keep the boundary clear. Same-cycle Psi gives R2 = {same.r2_vs_training_mean:.3f}, whereas lag-2 Psi gives R2 = {lag2.r2_vs_training_mean:.3f}; a lag-2 top-5% force-tail control gives R2 = {top5.r2_vs_training_mean:.3f}. Thus lagged force-loop structure contains precursor information, but it is route-periodic and weaker than phase-resolved same-cycle loop activation.",
        "",
        "## Lag scan",
        "",
        lag_scan.round(4).to_markdown(index=False),
        "",
        "## Forward tests",
        "",
        forward.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        route.round(4).to_markdown(index=False),
        "",
        "## Manuscript use",
        "",
        "Allowed: hot overload has a route-conditioned force-loop precursor at even cycle lags, consistent with a dissipative return map whose loop sector carries memory across breaths.",
        "",
        "Not allowed: do not claim monotonic early warning, universal forecasting or a one-cycle causal trigger. Lag-1 is not significant under circular-shift nulls, and same-cycle loop activation remains the primary mechanism.",
        "",
        "## Outputs",
        "",
        "- `figures/nphys_fig35_force_loop_precursor.*`",
        "- `source_data/nphys_force_loop_precursor_lag_scan.csv`",
        "- `source_data/nphys_force_loop_precursor_circular_shift_null.csv`",
        "- `source_data/nphys_force_loop_precursor_forward_tests.csv`",
        "- `source_data/nphys_force_loop_precursor_predictions.csv`",
        "- `source_data/nphys_force_loop_precursor_route_summary.csv`",
        "",
    ]
    (ROOT / "nature_physics_force_loop_precursor.md").write_text("\n".join(lines))


def main() -> None:
    df = load_table()
    lag_scan, _ = build_lag_scan(df)
    forward = forward_tests(df)
    route = route_summary(df)
    draw_figure(df, lag_scan, forward, route)
    write_report(lag_scan, forward, route)
    print(lag_scan.round(4).to_string(index=False))
    print(forward.round(4).to_string(index=False))
    print(route.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
