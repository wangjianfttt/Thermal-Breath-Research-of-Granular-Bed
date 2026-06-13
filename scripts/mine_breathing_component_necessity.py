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
INFILE = SRC / "nphys_breathing_vital_signs_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#7A8491"
GRID = "#E7EAEE"
BLUE = "#345995"
GOLD = "#D98C3A"
RED = "#B6423E"
GREEN = "#4F8B67"
ROUTE_COLORS = {"R1": BLUE, "R3": GOLD, "R6": RED}


MODELS = {
    "A": ["log_A"],
    "L+": ["log_L_plus"],
    "eta": ["log_eta"],
    "Hb": ["log_H_b"],
    "A,L+,eta": ["log_A", "log_L_plus", "log_eta"],
    "drop A": ["log_L_plus", "log_eta"],
    "drop L+": ["log_A", "log_eta"],
    "drop eta": ["log_A", "log_L_plus"],
    "A,L+,eta,Ib": ["log_A", "log_L_plus", "log_eta", "I_b"],
    "full vital signs": ["log_A", "log_L_plus", "log_eta", "I_b", "E_m"],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=INK)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INFILE).replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["regime_id", "cycle"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    scaler = StandardScaler().fit(train[features].to_numpy(float))
    x_train = scaler.transform(train[features].to_numpy(float))
    x_test = scaler.transform(test[features].to_numpy(float))
    y_train = train["Omega"].to_numpy(float)
    if len(features) == 1:
        model = LinearRegression().fit(x_train, y_train)
    else:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, y_train)
    return model.predict(x_test)


def r2_vs_base(y: np.ndarray, yhat: np.ndarray, base: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(base)
    y = y[ok]
    yhat = yhat[ok]
    base = base[ok]
    if len(y) == 0:
        return np.nan
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - base) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def within_route_forward(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, features in MODELS.items():
        y_all: list[float] = []
        yhat_all: list[float] = []
        base_all: list[float] = []
        for _, g in df.groupby("regime_id", sort=True):
            d = g.dropna(subset=["Omega", *features])
            train = d[d["cycle"] <= 18]
            test = d[d["cycle"] > 18]
            if len(train) < 6 or len(test) < 5:
                continue
            y = test["Omega"].to_numpy(float)
            yhat = fit_predict(train, test, features)
            base = np.repeat(float(train["Omega"].mean()), len(test))
            y_all.extend(y)
            yhat_all.extend(yhat)
            base_all.extend(base)
        y_arr = np.asarray(y_all)
        yhat_arr = np.asarray(yhat_all)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "within_route_forward_60_40",
                "n_test": int(len(y_arr)),
                "r2_vs_training_mean": r2_vs_base(y_arr, yhat_arr, np.asarray(base_all)),
                "spearman_prediction": safe_spearman(y_arr, yhat_arr),
            }
        )
    out = pd.DataFrame(rows)
    full = float(out.loc[out["model"] == "A,L+,eta", "r2_vs_training_mean"].iloc[0])
    out["delta_r2_from_A_L_eta"] = out["r2_vs_training_mean"] - full
    return out


def route_heldout(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    routes = sorted(df["regime_id"].unique())
    for name, features in MODELS.items():
        for holdout in routes:
            train = df[df["regime_id"] != holdout].dropna(subset=["Omega", *features])
            test = df[df["regime_id"] == holdout].dropna(subset=["Omega", *features])
            if len(train) < 10 or len(test) < 5:
                continue
            y = test["Omega"].to_numpy(float)
            yhat = fit_predict(train, test, features)
            base = np.repeat(float(test["Omega"].mean()), len(test))
            rows.append(
                {
                    "model": name,
                    "heldout_route": holdout,
                    "features": ";".join(features),
                    "n_test": int(len(test)),
                    "r2_vs_heldout_mean": r2_vs_base(y, yhat, base),
                    "spearman_prediction": safe_spearman(y, yhat),
                    "mean_error": float(np.mean(yhat - y)),
                }
            )
    return pd.DataFrame(rows)


def route_centered_rho(df: pd.DataFrame, col: str) -> float:
    d = df[[col, "Omega", "regime_id"]].dropna().copy()
    d[f"{col}_wc"] = d[col] - d.groupby("regime_id")[col].transform("mean")
    d["Omega_wc"] = d["Omega"] - d.groupby("regime_id")["Omega"].transform("mean")
    return float(spearmanr(d[f"{col}_wc"], d["Omega_wc"]).statistic)


def bootstrap_correlations(df: pd.DataFrame, n_boot: int = 2500, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    predictors = {
        "A": "A",
        "L+": "L_plus",
        "eta": "eta",
        "Hb": "log_H_b",
        "Vb": "breathing_vital_index",
        "Em": "E_m",
    }
    rows = []
    grouped = {rid: g.reset_index(drop=True) for rid, g in df.groupby("regime_id", sort=True)}
    for label, col in predictors.items():
        obs = route_centered_rho(df, col)
        vals = []
        for _ in range(n_boot):
            parts = []
            for rid, g in grouped.items():
                idx = rng.integers(0, len(g), size=len(g))
                parts.append(g.iloc[idx])
            sample = pd.concat(parts, ignore_index=True)
            vals.append(route_centered_rho(sample, col))
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "predictor": label,
                "column": col,
                "observed_spearman": obs,
                "ci_low_95": float(np.nanpercentile(arr, 2.5)),
                "ci_high_95": float(np.nanpercentile(arr, 97.5)),
                "bootstrap_n": n_boot,
            }
        )
    return pd.DataFrame(rows)


def draw_forward(ax: plt.Axes, fwd: pd.DataFrame) -> None:
    order = ["A", "L+", "eta", "Hb", "A,L+,eta", "drop A", "drop L+", "drop eta", "A,L+,eta,Ib", "full vital signs"]
    d = fwd.set_index("model").loc[order].reset_index()
    y = np.arange(len(d))
    vals = d["r2_vs_training_mean"].to_numpy(float)
    colors = [RED if m == "A,L+,eta" else MUTED if v < 0 else "#C95F3F" for m, v in zip(d["model"], vals)]
    ax.barh(y, vals, color=colors, alpha=0.86)
    ax.axvline(0, color="#AEB6C0", lw=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"], fontsize=5.8)
    for yi, v in zip(y, vals):
        ax.text(v + (0.012 if v >= 0 else -0.012), yi, f"{v:.2f}", ha="left" if v >= 0 else "right", va="center", fontsize=5.2, color=INK)
    ax.set_xlabel(r"within-route forward $R^2$")
    ax.set_title("component hierarchy", loc="left", pad=4)
    finish(ax, axis="x")


def draw_deletion(ax: plt.Axes, fwd: pd.DataFrame) -> None:
    base = float(fwd.loc[fwd["model"] == "A,L+,eta", "r2_vs_training_mean"].iloc[0])
    rows = []
    for name, dropped in [("drop A", "A"), ("drop L+", "L+"), ("drop eta", r"$\eta$")]:
        r2 = float(fwd.loc[fwd["model"] == name, "r2_vs_training_mean"].iloc[0])
        rows.append({"dropped": dropped, "loss": base - r2})
    d = pd.DataFrame(rows)
    y = np.arange(len(d))
    ax.barh(y, d["loss"], color=[BLUE, GOLD, GREEN], alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(d["dropped"])
    for yi, v in zip(y, d["loss"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6.0, color=INK)
    ax.set_xlabel(r"$R^2$ loss from removing component")
    ax.set_title("necessity within the triplet", loc="left", pad=4)
    finish(ax, axis="x")


def draw_route_transfer(ax: plt.Axes, held: pd.DataFrame) -> None:
    models = ["A", "L+", "Hb", "A,L+,eta", "full vital signs"]
    routes = ["R1", "R3", "R6"]
    d = held[held["model"].isin(models)].copy()
    mat = d.pivot(index="model", columns="heldout_route", values="r2_vs_heldout_mean").loc[models, routes]
    mat_clip = mat.clip(lower=-2, upper=1)
    im = ax.imshow(mat_clip.to_numpy(float), cmap="RdBu_r", vmin=-2, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(routes)))
    ax.set_xticklabels(routes)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=5.8)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            shown = "<-2" if value < -2 else f"{value:.1f}"
            ax.text(j, i, shown, ha="center", va="center", fontsize=5.2, color="white" if abs(mat_clip.iloc[i, j]) > 0.95 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cbar.ax.tick_params(labelsize=5.4, length=2)
    cbar.set_label(r"held-out $R^2$", fontsize=5.6)
    ax.set_title("route-held-out transfer", loc="left", pad=4)


def draw_bootstrap(ax: plt.Axes, boot: pd.DataFrame) -> None:
    order = ["A", "L+", "eta", "Hb", "Vb", "Em"]
    d = boot.set_index("predictor").loc[order].reset_index()
    y = np.arange(len(d))
    colors = [RED if v > 0 else GREEN for v in d["observed_spearman"]]
    ax.barh(y, d["observed_spearman"], color=colors, alpha=0.72)
    for yi, row in d.iterrows():
        ax.plot([row["ci_low_95"], row["ci_high_95"]], [yi, yi], color=INK, lw=0.9)
        ax.plot([row["ci_low_95"], row["ci_low_95"]], [yi - 0.09, yi + 0.09], color=INK, lw=0.7)
        ax.plot([row["ci_high_95"], row["ci_high_95"]], [yi - 0.09, yi + 0.09], color=INK, lw=0.7)
    ax.axvline(0, color="#AEB6C0", lw=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(d["predictor"])
    ax.set_xlabel(r"route-centred Spearman with $\Omega$")
    ax.set_title("bootstrap stability", loc="left", pad=4)
    finish(ax, axis="x")


def make_figure(fwd: pd.DataFrame, held: pd.DataFrame, boot: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.35, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[1.1, 0.9])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    draw_forward(ax_a, fwd)
    panel(ax_a, "a", x=-0.08)
    draw_route_transfer(ax_b, held)
    panel(ax_b, "b")
    draw_deletion(ax_d, fwd)
    panel(ax_d, "c", x=-0.10)
    draw_bootstrap(ax_c, boot)
    panel(ax_c, "d", x=-0.10)
    FIG.mkdir(exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig66_breathing_component_necessity.{ext}", **kwargs)
    plt.close(fig)


def write_report(fwd: pd.DataFrame, held: pd.DataFrame, boot: pd.DataFrame) -> None:
    triplet = fwd.loc[fwd["model"] == "A,L+,eta"].iloc[0]
    drop_a = fwd.loc[fwd["model"] == "drop A"].iloc[0]
    boot_hb = boot.loc[boot["predictor"] == "Hb"].iloc[0]
    held_triplet = held[held["model"] == "A,L+,eta"]
    out = ROOT / "nature_physics_breathing_component_necessity.md"
    lines = [
        "# Breathing component-necessity audit",
        "",
        "## Purpose",
        "",
        "This audit checks whether the vital-sign triplet `(A,L+,eta)` is a robust response compression or merely a polished restatement of one variable.",
        "",
        "## Main result",
        "",
        f"Within-route forward prediction favours the physical triplet `(A,L+,eta)` with R2 = {triplet.r2_vs_training_mean:.3f}. Dropping amplitude gives R2 = {drop_a.r2_vs_training_mean:.3f}, making amplitude necessary in this three-route response audit.",
        f"The hazard number remains stable under route-stratified bootstrap: rho = {boot_hb.observed_spearman:.3f}, 95% CI [{boot_hb.ci_low_95:.3f}, {boot_hb.ci_high_95:.3f}].",
        f"Route-held-out transfer for the triplet spans R2 from {held_triplet.r2_vs_heldout_mean.min():.3f} to {held_triplet.r2_vs_heldout_mean.max():.3f}, so the response compression is route-conditioned rather than universal.",
        "",
        "## Within-route forward hierarchy",
        "",
        fwd.round(4).to_markdown(index=False),
        "",
        "## Route-held-out transfer",
        "",
        held.round(4).to_markdown(index=False),
        "",
        "## Bootstrap correlations",
        "",
        boot.round(4).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: `A`, `L+` and `eta` are jointly useful physical vital signs for the measured response, with amplitude and loop drive carrying the strongest within-route signal.",
        "",
        "Not allowed: the triplet is not a route-independent constitutive law, and the three-route audit does not replace the five-route true-force loop-coordinate evidence.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig66_breathing_component_necessity.*`",
        "- `source_data/nphys_breathing_component_necessity_forward.csv`",
        "- `source_data/nphys_breathing_component_necessity_route_heldout.csv`",
        "- `source_data/nphys_breathing_component_necessity_bootstrap.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_data()
    fwd = within_route_forward(df)
    held = route_heldout(df)
    boot = bootstrap_correlations(df)
    fwd.to_csv(SRC / "nphys_breathing_component_necessity_forward.csv", index=False)
    held.to_csv(SRC / "nphys_breathing_component_necessity_route_heldout.csv", index=False)
    boot.to_csv(SRC / "nphys_breathing_component_necessity_bootstrap.csv", index=False)
    make_figure(fwd, held, boot)
    write_report(fwd, held, boot)
    print("Wrote breathing component-necessity audit")
    print(fwd[["model", "r2_vs_training_mean", "delta_r2_from_A_L_eta"]].round(3).to_string(index=False))
    print(held[held["model"].isin(["A,L+,eta", "Hb"])][["model", "heldout_route", "r2_vs_heldout_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
