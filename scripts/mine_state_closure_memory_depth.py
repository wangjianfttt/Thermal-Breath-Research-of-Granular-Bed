#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"

INK = "#242A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
BLUE = "#3D6B9C"
RED = "#B6423E"
ORANGE = "#D98C3A"
PURPLE = "#7E6AAE"
GREEN = "#3F7F6B"

MODEL_COLORS = {
    "route": "#B9C0C8",
    "route+state": BLUE,
    "route+state+tail": ORANGE,
    "route+state+history": PURPLE,
    "route+state+tail+history": RED,
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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def within_route_z(df: pd.DataFrame, col: str) -> pd.Series:
    mean = df.groupby("regime_id")[col].transform("mean")
    std = df.groupby("regime_id")[col].transform("std").replace(0, np.nan)
    return (df[col] - mean) / std


def prepare_table() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["tail_control"] = within_route_z(df, "top5_activation")
    df["loop_control"] = within_route_z(df, "loop_activation")
    for col in ["memory_coordinate", "hot_excitation_coordinate", "overload_number_wz", "tail_control", "loop_control"]:
        df[f"{col}_lag1"] = df.groupby("regime_id")[col].shift(1)
    df["target_next_overload"] = df["next_overload_number_wz"]
    df["target_next_memory"] = df["next_memory_coordinate"]
    df["target_next_hot"] = df["next_hot_excitation_coordinate"]
    return df


def encode_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    route_train = enc.fit_transform(train[["regime_id"]])
    route_test = enc.transform(test[["regime_id"]])
    if not feature_cols:
        return route_train, route_test
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols].to_numpy(float))
    x_test = scaler.transform(test[feature_cols].to_numpy(float))
    return np.hstack([route_train, x_train]), np.hstack([route_test, x_test])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[np.ndarray, float, float]:
    x_train, x_test = encode_features(train, test, feature_cols)
    y_train = train[target].to_numpy(float)
    y_test = test[target].to_numpy(float)
    model = RidgeCV(alphas=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]).fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred, float(r2_score(y_test, pred)), float(np.sqrt(mean_squared_error(y_test, pred)))


def model_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = {
        "next overload": "target_next_overload",
        "next cold memory": "target_next_memory",
        "next hot excitation": "target_next_hot",
    }
    feature_sets = {
        "route": [],
        "route+state": ["memory_coordinate", "hot_excitation_coordinate"],
        "route+state+tail": ["memory_coordinate", "hot_excitation_coordinate", "tail_control"],
        "route+state+history": [
            "memory_coordinate",
            "hot_excitation_coordinate",
            "memory_coordinate_lag1",
            "hot_excitation_coordinate_lag1",
        ],
        "route+state+tail+history": [
            "memory_coordinate",
            "hot_excitation_coordinate",
            "tail_control",
            "memory_coordinate_lag1",
            "hot_excitation_coordinate_lag1",
            "tail_control_lag1",
        ],
    }
    rows: list[dict[str, float | str | int]] = []
    pred_rows: list[dict[str, float | str | int]] = []
    for target_label, target in targets.items():
        needed = [target, "regime_id", "cycle"]
        for cols in feature_sets.values():
            needed.extend(cols)
        d = df.dropna(subset=sorted(set(needed))).copy()
        train = d[d["cycle"] <= 20].copy()
        test = d[d["cycle"] > 20].copy()
        for model_name, cols in feature_sets.items():
            pred, r2, rmse = fit_predict(train, test, cols, target)
            rows.append(
                {
                    "target": target_label,
                    "target_column": target,
                    "model": model_name,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "split": "within_route_forward_cycle_20",
                    "r2": r2,
                    "rmse": rmse,
                }
            )
            for idx, yhat in zip(test.index, pred):
                pred_rows.append(
                    {
                        "target": target_label,
                        "model": model_name,
                        "regime_id": test.loc[idx, "regime_id"],
                        "cycle": int(test.loc[idx, "cycle"]),
                        "observed": float(test.loc[idx, target]),
                        "predicted": float(yhat),
                        "residual": float(test.loc[idx, target] - yhat),
                    }
                )
    perf = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    base = perf[perf["model"] == "route+state"].set_index("target")["r2"]
    inc_rows = []
    for _, row in perf.iterrows():
        if row["model"] == "route+state":
            continue
        inc_rows.append(
            {
                "target": row["target"],
                "model": row["model"],
                "delta_r2_vs_route_state": float(row["r2"] - base.loc[row["target"]]),
                "r2": float(row["r2"]),
            }
        )
    return perf, preds, pd.DataFrame(inc_rows)


def residual_memory(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    d = df.dropna(
        subset=[
            "memory_coordinate",
            "hot_excitation_coordinate",
            "next_memory_coordinate",
            "next_hot_excitation_coordinate",
        ]
    ).copy()
    for rid, g in d.groupby("regime_id", sort=True):
        g = g.sort_values("cycle")
        x = g[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
        y = g[["next_memory_coordinate", "next_hot_excitation_coordinate"]].to_numpy(float)
        fit = LinearRegression().fit(x, y)
        residual = y - fit.predict(x)
        for k, name in enumerate(["next cold-memory residual", "next hot-excitation residual"]):
            values = residual[:, k]
            raw = y[:, k]
            if len(values) > 3:
                res_r = pearsonr(values[1:], values[:-1]).statistic
                raw_r = pearsonr(raw[1:], raw[:-1]).statistic
            else:
                res_r = np.nan
                raw_r = np.nan
            rows.append(
                {
                    "regime_id": rid,
                    "series": name,
                    "n_pairs": int(max(len(values) - 1, 0)),
                    "raw_lag1_pearson": float(raw_r),
                    "route_map_residual_lag1_pearson": float(res_r),
                    "absolute_memory_removed": float(abs(raw_r) - abs(res_r)) if np.isfinite(raw_r) and np.isfinite(res_r) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def draw_performance(ax: plt.Axes, perf: pd.DataFrame) -> None:
    order_targets = ["next overload", "next cold memory", "next hot excitation"]
    models = ["route", "route+state", "route+state+tail", "route+state+history"]
    x = np.arange(len(order_targets))
    width = 0.18
    for i, model in enumerate(models):
        vals = [
            float(perf[(perf["target"] == target) & (perf["model"] == model)]["r2"].iloc[0])
            for target in order_targets
        ]
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=MODEL_COLORS[model], label=model, zorder=3)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["overload", "cold\nmemory", "hot\nexcitation"])
    ax.set_ylabel(r"forward $R^2$")
    ax.set_title("one-state closure fails in late-cycle transfer", loc="left", pad=4)
    ax.legend(fontsize=5.6, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    finish(ax, axis="y")


def draw_increment(ax: plt.Axes, inc: pd.DataFrame) -> None:
    show = inc[inc["model"].isin(["route", "route+state+tail", "route+state+history"])].copy()
    order_targets = ["next overload", "next cold memory", "next hot excitation"]
    order_models = ["route", "route+state+tail", "route+state+history"]
    mat = np.full((len(order_targets), len(order_models)), np.nan)
    for i, target in enumerate(order_targets):
        for j, model in enumerate(order_models):
            row = show[(show["target"] == target) & (show["model"] == model)]
            if not row.empty:
                mat[i, j] = row["delta_r2_vs_route_state"].iloc[0]
    vmax = max(0.25, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(order_targets)))
    ax.set_yticklabels(["overload", "cold memory", "hot excitation"])
    ax.set_xticks(np.arange(len(order_models)))
    ax.set_xticklabels(["route\nonly", "+ tail", "+ lag-1\nstate"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.1, color="white" if abs(v) > 0.55 * vmax else INK)
    ax.set_title(r"increment relative to route+$X_n$", loc="left", pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(r"$\Delta R^2$", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8, length=2)


def draw_residual_memory(ax: plt.Axes, res: pd.DataFrame) -> None:
    summary = (
        res.groupby("series")[["raw_lag1_pearson", "route_map_residual_lag1_pearson"]]
        .median()
        .reset_index()
    )
    x = np.arange(len(summary))
    width = 0.28
    ax.bar(x - width / 2, summary["raw_lag1_pearson"], width=width, color="#B9C0C8", label="raw target", zorder=3)
    ax.bar(x + width / 2, summary["route_map_residual_lag1_pearson"], width=width, color=GREEN, label="map residual", zorder=3)
    for i, row in res.iterrows():
        j = 0 if row["series"] == "next cold-memory residual" else 1
        ax.scatter(j - width / 2, row["raw_lag1_pearson"], s=12, color="white", edgecolor="#6C747D", lw=0.45, zorder=4)
        ax.scatter(j + width / 2, row["route_map_residual_lag1_pearson"], s=12, color="white", edgecolor=GREEN, lw=0.45, zorder=4)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["cold-memory\nreadout", "hot-excitation\nreadout"])
    ax.set_ylabel("within-route lag-1 correlation")
    ax.set_title("how much memory remains in map residuals?", loc="left", pad=4)
    ax.legend(fontsize=5.8, loc="lower left")
    finish(ax, axis="y")


def draw_prediction(ax: plt.Axes, preds: pd.DataFrame) -> None:
    model_name = "route+state+history"
    d = preds[(preds["target"] == "next overload") & (preds["model"] == model_name)].copy()
    route_colors = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
    for rid, g in d.groupby("regime_id", sort=True):
        ax.scatter(g["observed"], g["predicted"], s=22, color=route_colors.get(rid, BLUE), edgecolor="white", lw=0.45, label=rid, zorder=3)
    lim = np.nanmax(np.abs(d[["observed", "predicted"]].to_numpy(float))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color="#AEB6C0", lw=0.75, ls=(0, (3, 3)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("observed next overload")
    ax.set_ylabel(r"predicted from route + $X_n+X_{n-1}$")
    r2 = r2_score(d["observed"], d["predicted"])
    ax.text(0.05, 0.94, rf"$R^2={r2:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK)
    ax.set_title("phase-augmented overload readout", loc="left", pad=4)
    ax.legend(fontsize=5.4, ncol=2, loc="lower right")
    finish(ax)


def build_figure(perf: pd.DataFrame, inc: pd.DataFrame, res: pd.DataFrame, preds: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    draw_performance(ax_a, perf)
    panel(ax_a, "a")
    draw_increment(ax_b, inc)
    panel(ax_b, "b")
    draw_residual_memory(ax_c, res)
    panel(ax_c, "c")
    draw_prediction(ax_d, preds)
    panel(ax_d, "d")
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig62_state_closure_memory_depth.{ext}", **kwargs)
    plt.close(fig)


def write_report(perf: pd.DataFrame, inc: pd.DataFrame, res: pd.DataFrame) -> None:
    summary_rows = []
    for target in ["next overload", "next cold memory", "next hot excitation"]:
        state_r2 = float(perf[(perf["target"] == target) & (perf["model"] == "route+state")]["r2"].iloc[0])
        hist_dr2 = float(inc[(inc["target"] == target) & (inc["model"] == "route+state+history")]["delta_r2_vs_route_state"].iloc[0])
        tail_dr2 = float(inc[(inc["target"] == target) & (inc["model"] == "route+state+tail")]["delta_r2_vs_route_state"].iloc[0])
        summary_rows.append(
            {
                "target": target,
                "route_state_r2": state_r2,
                "history_delta_r2": hist_dr2,
                "tail_delta_r2": tail_dr2,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SRC / "nphys_state_closure_summary.csv", index=False)
    out = ROOT / "nature_physics_state_closure_memory_depth.md"
    lines = [
        "# State-closure and memory-depth audit",
        "",
        "Purpose: test whether the reduced route-conditioned state `X_n=(M_n,Psi_n)` behaves like a useful one-step state description, rather than requiring an unrestricted history of earlier cycles.",
        "",
        "## Forward model summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Full model tests",
        "",
        perf.round(4).to_markdown(index=False),
        "",
        "## Increment relative to route + current state",
        "",
        inc.round(4).to_markdown(index=False),
        "",
        "## Residual memory after route-map fit",
        "",
        res.round(4).to_markdown(index=False),
        "",
        "Interpretation: a strictly one-state closure is not supported by the late-cycle forward split. The current reduced state `X_n=(M_n,Psi_n)` is not enough by itself, whereas adding the previous reduced state `X_{n-1}` gives large gains for overload and hot excitation. This supports a finite-memory, phase-augmented breathing map rather than a first-order Markov constitutive law. The result is finite-route and target-dependent, so it should be used to bound the return-map language, not to claim a universal oscillator.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare_table()
    perf, preds, inc = model_tests(df)
    res = residual_memory(df)
    perf.to_csv(SRC / "nphys_state_closure_model_tests.csv", index=False)
    preds.to_csv(SRC / "nphys_state_closure_predictions.csv", index=False)
    inc.to_csv(SRC / "nphys_state_closure_incremental.csv", index=False)
    res.to_csv(SRC / "nphys_state_closure_residual_memory.csv", index=False)
    build_figure(perf, inc, res, preds)
    write_report(perf, inc, res)
    print("Wrote state-closure memory-depth audit")
    print(pd.read_csv(SRC / "nphys_state_closure_summary.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
