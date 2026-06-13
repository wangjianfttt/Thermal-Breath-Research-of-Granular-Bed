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

QUALITY = SRC / "nphys_breathing_quality_factor_cycle_metrics.csv"
MEMCOST = SRC / "nphys_breathing_memory_cost_tradeoff_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#7A8491"
GRID = "#E7EAEE"
BLUE = "#345995"
GOLD = "#D98C3A"
RED = "#B6423E"
GREEN = "#4F8B67"
VIOLET = "#7E6AAE"
ROUTE_COLORS = {"R1": BLUE, "R3": GOLD, "R6": RED}
ROUTE_MARKERS = {"R1": "o", "R3": "s", "R6": "^"}
SEGMENT_ORDER = ["early", "middle", "late"]


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


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def within_route_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def prepare() -> pd.DataFrame:
    q = pd.read_csv(QUALITY).copy()
    m = pd.read_csv(MEMCOST)[
        [
            "regime_id",
            "cycle",
            "route_efficiency_score",
            "route_frontier_deficit",
            "memory_benefit",
            "overload_cost_positive",
        ]
    ].copy()
    df = q.merge(m, on=["regime_id", "cycle"], how="left", validate="one_to_one")
    df = df.sort_values(["regime_id", "cycle"]).reset_index(drop=True)
    eps = 1e-9
    df["A"] = df["response_amplitude"].clip(lower=0.0)
    df["L_plus"] = df["loop_activation_positive"].clip(lower=0.0)
    df["eta"] = df["buffer_efficiency"].clip(lower=0.0)
    df["H_b"] = df["breathing_hazard_number"].clip(lower=0.0)
    df["Q_b"] = df["breathing_quality_factor"].clip(lower=0.0)
    df["Omega"] = df["overload_asinh"]
    df["I_b"] = df["breath_irregularity"].fillna(0.0).clip(lower=0.0)
    df["E_m"] = df["route_efficiency_score"].fillna(0.0)
    df["log_A"] = np.log10(df["A"] + eps)
    df["log_L_plus"] = np.log10(df["L_plus"] + eps)
    df["log_eta"] = np.log10(df["eta"] + eps)
    df["log_H_b"] = np.log10(df["H_b"] + eps)
    df["log_Q_b"] = np.log10(df["Q_b"] + eps)
    # Operational diagnostic only: large inhale amplitude/loop drive/irregularity,
    # weak buffering and low memory-cost efficiency all raise the risk coordinate.
    df["breathing_vital_index"] = (
        zscore(df["log_A"])
        + zscore(df["log_L_plus"])
        - zscore(df["log_eta"])
        + zscore(df["I_b"])
        - zscore(df["E_m"])
    )
    df["vital_index_wc"] = within_route_center(df, "breathing_vital_index")
    df["Omega_wc"] = within_route_center(df, "Omega")
    return df


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "A": "mean",
        "L_plus": "mean",
        "eta": "mean",
        "H_b": "mean",
        "Q_b": "mean",
        "I_b": "mean",
        "E_m": "mean",
        "Omega": "mean",
        "breathing_vital_index": "mean",
    }
    return (
        df.groupby(["regime_id", "segment"], observed=True)
        .agg(n=("cycle", "count"), **{f"mean_{k}": (k, v) for k, v in metrics.items()})
        .reset_index()
    )


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("amplitude A", "A"),
        ("loop activation L+", "L_plus"),
        ("buffer efficiency eta", "eta"),
        ("hazard Hb", "log_H_b"),
        ("quality Qb", "log_Q_b"),
        ("irregularity Ib", "I_b"),
        ("memory-cost efficiency Em", "E_m"),
        ("vital index Vb", "breathing_vital_index"),
    ]
    rows = []
    for label, col in pairs:
        d = df[[col, "Omega", "regime_id"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        d[f"{col}_wc"] = within_route_center(d, col)
        d["Omega_wc"] = within_route_center(d, "Omega")
        raw = spearmanr(d[col], d["Omega"])
        wc = spearmanr(d[f"{col}_wc"], d["Omega_wc"])
        rows.append(
            {
                "predictor_label": label,
                "predictor": col,
                "target": "Omega",
                "n": int(len(d)),
                "spearman_raw": float(raw.statistic),
                "p_raw": float(raw.pvalue),
                "spearman_within_route": float(wc.statistic),
                "p_within_route": float(wc.pvalue),
            }
        )
    return pd.DataFrame(rows)


def fit_forward(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    scaler = StandardScaler().fit(train[features].to_numpy(float))
    x_train = scaler.transform(train[features].to_numpy(float))
    x_test = scaler.transform(test[features].to_numpy(float))
    y_train = train[target].to_numpy(float)
    if len(features) == 1:
        model = LinearRegression().fit(x_train, y_train)
    else:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, y_train)
    return model.predict(x_test)


def r2_vs_baseline(y: np.ndarray, yhat: np.ndarray, base: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(base)
    y = y[ok]
    yhat = yhat[ok]
    base = base[ok]
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - base) ** 2))
    return 1.0 - sse / sse0 if sse0 > 0 else np.nan


def forward_tests(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "A only": ["log_A"],
        "L+ only": ["log_L_plus"],
        "eta only": ["log_eta"],
        "Hb": ["log_H_b"],
        "Vb": ["breathing_vital_index"],
        "A,L+,eta": ["log_A", "log_L_plus", "log_eta"],
        "A,L+,eta,Ib": ["log_A", "log_L_plus", "log_eta", "I_b"],
        "full vital signs": ["log_A", "log_L_plus", "log_eta", "I_b", "E_m"],
    }
    rows = []
    for name, features in models.items():
        y_all: list[float] = []
        yhat_all: list[float] = []
        base_all: list[float] = []
        for rid, g in df.groupby("regime_id", sort=True):
            train = g[g["cycle"] <= 18].replace([np.inf, -np.inf], np.nan).dropna(subset=["Omega", *features])
            test = g[g["cycle"] > 18].replace([np.inf, -np.inf], np.nan).dropna(subset=["Omega", *features])
            if len(train) < 6 or len(test) < 5:
                continue
            y = test["Omega"].to_numpy(float)
            yhat = fit_forward(train, test, features, "Omega")
            base = np.repeat(float(train["Omega"].mean()), len(test))
            y_all.extend(y)
            yhat_all.extend(yhat)
            base_all.extend(base)
        y_arr = np.asarray(y_all, dtype=float)
        yhat_arr = np.asarray(yhat_all, dtype=float)
        base_arr = np.asarray(base_all, dtype=float)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "target": "Omega",
                "validation": "within_route_forward_60_40",
                "n_test": int(len(y_arr)),
                "r2_vs_training_mean": r2_vs_baseline(y_arr, yhat_arr, base_arr),
                "spearman_prediction": float(spearmanr(y_arr, yhat_arr).statistic) if len(y_arr) > 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def shift_null(df: pd.DataFrame, n_perm: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs = float(spearmanr(df["vital_index_wc"], df["Omega_wc"]).statistic)
    rows = []
    for i in range(n_perm):
        shifted = []
        for _, g in df.groupby("regime_id", sort=True):
            vals = g["vital_index_wc"].to_numpy(float)
            shift = int(rng.integers(0, len(vals)))
            shifted.extend(np.roll(vals, shift))
        rho = float(spearmanr(np.asarray(shifted), df["Omega_wc"].to_numpy(float)).statistic)
        rows.append({"permutation": i, "null_spearman": rho, "observed_spearman": obs})
    out = pd.DataFrame(rows)
    out["two_sided_p"] = (np.sum(np.abs(out["null_spearman"]) >= abs(obs)) + 1.0) / (len(out) + 1.0)
    return out


def draw_heatmap(ax: plt.Axes, seg: pd.DataFrame) -> None:
    cols = [
        ("A", "mean_A"),
        (r"$L_+$", "mean_L_plus"),
        (r"$\eta$", "mean_eta"),
        (r"$H_b$", "mean_H_b"),
        (r"$I_b$", "mean_I_b"),
        (r"$E_m$", "mean_E_m"),
        (r"$\Omega$", "mean_Omega"),
    ]
    rows = []
    labels = []
    for rid in ["R1", "R3", "R6"]:
        g = seg[seg["regime_id"] == rid].set_index("segment")
        for s in SEGMENT_ORDER:
            if s in g.index:
                rows.append([g.loc[s, col] for _, col in cols])
                labels.append(f"{rid} {s[0]}")
    mat = np.asarray(rows, dtype=float)
    mat_z = np.column_stack([zscore(pd.Series(mat[:, j])).to_numpy(float) for j in range(mat.shape[1])])
    mat_z = np.clip(mat_z, -2.0, 2.0)
    im = ax.imshow(mat_z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat_z.shape[0]):
        for j in range(mat_z.shape[1]):
            ax.text(j, i, f"{mat_z[i, j]:+.1f}", ha="center", va="center", fontsize=5.2, color="white" if abs(mat_z[i, j]) > 1.15 else INK)
    cbar = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.018)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    cbar.set_label("z across route-segments", fontsize=5.8)
    ax.set_title("breathing vital signs", loc="left", pad=4)


def draw_state_plane(ax: plt.Axes, df: pd.DataFrame, corr: pd.DataFrame, null: pd.DataFrame) -> None:
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["log_H_b"],
            g["Omega"],
            s=24,
            marker=ROUTE_MARKERS[rid],
            color=ROUTE_COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.88,
            label=rid,
            zorder=3,
        )
    stat = corr[corr["predictor"] == "log_H_b"].iloc[0]
    ax.text(
        0.04,
        0.94,
        rf"within-route $\rho={stat.spearman_within_route:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=INK,
    )
    ax.text(
        0.04,
        0.84,
        rf"shift-null $P={null['two_sided_p'].iloc[0]:.3f}$ for $V_b$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=MUTED,
    )
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel(r"$\log_{10} H_b$")
    ax.set_ylabel(r"overload, $\operatorname{asinh}\Omega$")
    ax.set_title("hazard state plane", loc="left", pad=4)
    finish(ax)


def draw_correlation_bars(ax: plt.Axes, corr: pd.DataFrame) -> None:
    order = ["amplitude A", "loop activation L+", "buffer efficiency eta", "hazard Hb", "quality Qb", "irregularity Ib", "memory-cost efficiency Em", "vital index Vb"]
    d = corr.set_index("predictor_label").loc[order].reset_index()
    y = np.arange(len(d))
    colors = [RED if v > 0 else GREEN for v in d["spearman_within_route"]]
    ax.barh(y, d["spearman_within_route"], color=colors, alpha=0.86)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["predictor_label"], fontsize=5.7)
    for yi, row in d.iterrows():
        x = row["spearman_within_route"]
        ax.text(0.05 if x >= 0 else -0.05, yi, f"{x:+.2f}", ha="left" if x >= 0 else "right", va="center", fontsize=5.4, color=INK)
    ax.set_xlim(-0.75, 0.95)
    ax.set_xlabel(r"within-route Spearman with $\Omega$")
    ax.set_title("which vital signs carry risk?", loc="left", pad=4)
    finish(ax, axis="x")


def draw_model_tests(ax: plt.Axes, tests: pd.DataFrame) -> None:
    order = ["A only", "L+ only", "eta only", "Hb", "Vb", "A,L+,eta", "A,L+,eta,Ib", "full vital signs"]
    d = tests.set_index("model").loc[order].reset_index()
    y = np.arange(len(d))
    vals = d["r2_vs_training_mean"].to_numpy(float)
    colors = [RED if v >= 0 else MUTED for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"], fontsize=5.8)
    for yi, row in d.iterrows():
        ax.text(row["r2_vs_training_mean"] + (0.015 if row["r2_vs_training_mean"] >= 0 else -0.015), yi, f"{row['r2_vs_training_mean']:.2f}", va="center", ha="left" if row["r2_vs_training_mean"] >= 0 else "right", fontsize=5.4, color=INK)
    ax.set_xlabel(r"forward $R^2$ vs route training mean")
    ax.set_title("compression test", loc="left", pad=4)
    finish(ax, axis="x")


def make_figure(df: pd.DataFrame, seg: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, null: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.35, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.28, 0.9, 0.9], height_ratios=[1.05, 0.95])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    draw_heatmap(ax_a, seg)
    panel(ax_a, "a", x=-0.08)
    draw_state_plane(ax_b, df, corr, null)
    panel(ax_b, "b")
    draw_correlation_bars(ax_c, corr)
    panel(ax_c, "c")
    draw_model_tests(ax_d, tests)
    panel(ax_d, "d")
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig65_breathing_vital_signs.{ext}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, null: pd.DataFrame) -> None:
    hb = corr[corr["predictor"] == "log_H_b"].iloc[0]
    vb = corr[corr["predictor"] == "breathing_vital_index"].iloc[0]
    best = tests.sort_values("r2_vs_training_mean", ascending=False).iloc[0]
    out = ROOT / "nature_physics_breathing_vital_signs.md"
    lines = [
        "# Breathing vital-signs audit",
        "",
        "## Purpose",
        "",
        "This audit asks whether the breathing language can be expressed as a small set of measurable response variables: amplitude `A`, positive loop activation `L+`, buffer efficiency `eta`, hazard `Hb`, irregularity `Ib`, memory-cost efficiency `Em` and overload `Omega`.",
        "",
        "## Main results",
        "",
        f"`log10(Hb)` correlates with overload within route with Spearman rho = {hb.spearman_within_route:.3f} (P = {hb.p_within_route:.4g}).",
        f"The operational vital index `Vb=z(log A)+z(log L+)-z(log eta)+z(Ib)-z(Em)` correlates with overload within route with rho = {vb.spearman_within_route:.3f} (P = {vb.p_within_route:.4g}); the route-preserving circular-shift null gives P = {null['two_sided_p'].iloc[0]:.4g}.",
        f"The best forward 60/40 route-local compression in this audit is `{best.model}` with R2 = {best.r2_vs_training_mean:.3f} relative to each route's training mean.",
        "",
        "## Evidence tables",
        "",
        "### Correlations",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "### Forward compression tests",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: breathing can be operationalised as a measurable multi-variable response, and the hazard/buffer variables organise overload within the measured three-route subset.",
        "",
        "Not allowed: this is not a universal physiology-like law, not a thermodynamic entropy-production principle and not five-route generality evidence. Loop activation remains the sharper five-route force-loop coordinate in the main mechanism chain.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig65_breathing_vital_signs.*`",
        "- `source_data/nphys_breathing_vital_signs_cycle_metrics.csv`",
        "- `source_data/nphys_breathing_vital_signs_segment_summary.csv`",
        "- `source_data/nphys_breathing_vital_signs_correlations.csv`",
        "- `source_data/nphys_breathing_vital_signs_model_tests.csv`",
        "- `source_data/nphys_breathing_vital_signs_shift_null.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare()
    seg = segment_summary(df)
    corr = correlation_table(df)
    tests = forward_tests(df)
    null = shift_null(df)
    df.to_csv(SRC / "nphys_breathing_vital_signs_cycle_metrics.csv", index=False)
    seg.to_csv(SRC / "nphys_breathing_vital_signs_segment_summary.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_vital_signs_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_breathing_vital_signs_model_tests.csv", index=False)
    null.to_csv(SRC / "nphys_breathing_vital_signs_shift_null.csv", index=False)
    make_figure(df, seg, corr, tests, null)
    write_report(df, corr, tests, null)
    print("Wrote breathing vital-signs audit")
    print(corr[["predictor_label", "spearman_within_route", "p_within_route"]].round(3).to_string(index=False))
    print(tests[["model", "r2_vs_training_mean", "spearman_prediction"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
