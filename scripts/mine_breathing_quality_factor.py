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
INFILE = SRC / "nphys_breathing_response_function_cycle_metrics.csv"

INK = "#252A31"
GRID = "#E6E9EF"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
RED = "#B6423E"
GREEN = "#4F8B67"
VIOLET = "#7E6AAE"
NEUTRAL = "#8D99A6"
COLORS = {"R1": BLUE, "R3": GOLD, "R6": RED}
MARKERS = {"R1": "o", "R3": "s", "R6": "^"}


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


def within_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0
    return (s - s.mean()) / std


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).copy()
    df = df.sort_values(["regime_id", "cycle"]).reset_index(drop=True)
    loop_pos = df["response_loop_activation"].clip(lower=0.0)
    positives = loop_pos[loop_pos > 0]
    eps_loop = float(np.nanpercentile(positives, 10)) if len(positives) else 1e-5
    eps_imprint = float(np.nanpercentile(df["response_imprint_efficiency"].abs().dropna(), 10))
    eps_loop = max(eps_loop, 1e-5)
    eps_imprint = max(eps_imprint, 1e-3)
    df["loop_activation_positive"] = loop_pos
    df["loop_activation_eps"] = loop_pos + eps_loop
    df["buffer_efficiency"] = df["response_imprint_efficiency"].clip(lower=0.0)
    df["buffer_eps"] = df["buffer_efficiency"] + eps_imprint
    df["breathing_hazard_number"] = (
        df["response_amplitude"].clip(lower=0.0) * df["loop_activation_eps"] / df["buffer_eps"]
    )
    df["breathing_quality_factor"] = df["buffer_eps"] / (
        df["response_amplitude"].clip(lower=0.0) * df["loop_activation_eps"] + 1e-12
    )
    df["log_hazard_number"] = np.log10(df["breathing_hazard_number"] + 1e-12)
    df["log_quality_factor"] = np.log10(df["breathing_quality_factor"] + 1e-12)
    df["overload_asinh"] = df["response_overload_asinh"]
    df["strong_overload"] = (df["overload_asinh"] > 1.0).astype(int)
    df["route_centered_hazard"] = within_center(df, "log_hazard_number")
    df["route_centered_overload"] = within_center(df, "overload_asinh")
    return df


def correlation_rows(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("response_amplitude", "overload_asinh"),
        ("loop_activation_positive", "overload_asinh"),
        ("buffer_efficiency", "overload_asinh"),
        ("log_hazard_number", "overload_asinh"),
        ("log_quality_factor", "overload_asinh"),
        ("response_hazard_number", "overload_asinh"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for pred, target in pairs:
        d = df[["regime_id", pred, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        d[pred + "_wc"] = within_center(d, pred)
        d[target + "_wc"] = within_center(d, target)
        raw = spearmanr(d[pred], d[target])
        wc = spearmanr(d[pred + "_wc"], d[target + "_wc"])
        rows.append(
            {
                "predictor": pred,
                "target": target,
                "n": int(len(d)),
                "spearman_raw": float(raw.statistic),
                "p_raw": float(raw.pvalue),
                "spearman_within_route": float(wc.statistic),
                "p_within_route": float(wc.pvalue),
            }
        )
    return pd.DataFrame(rows)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    scaler = StandardScaler().fit(train[features].to_numpy(float))
    x_train = scaler.transform(train[features].to_numpy(float))
    x_test = scaler.transform(test[features].to_numpy(float))
    y_train = train[target].to_numpy(float)
    if len(features) == 1:
        model = LinearRegression().fit(x_train, y_train)
    else:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, y_train)
    return model.predict(x_test)


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


def forward_tests(df: pd.DataFrame) -> pd.DataFrame:
    target = "overload_asinh"
    models = {
        "amplitude": ["response_amplitude"],
        "loop activation": ["loop_activation_positive"],
        "buffer efficiency": ["buffer_efficiency"],
        "hazard number": ["log_hazard_number"],
        "quality factor": ["log_quality_factor"],
        "amplitude + loop": ["response_amplitude", "loop_activation_positive"],
        "hazard + buffer": ["log_hazard_number", "buffer_efficiency"],
        "full breathing triplet": ["response_amplitude", "loop_activation_positive", "buffer_efficiency"],
    }
    rows = []
    for name, features in models.items():
        y_all: list[float] = []
        yhat_all: list[float] = []
        base_all: list[float] = []
        for _, g in df.groupby("regime_id", sort=True):
            train = g[g["cycle"] <= 18].dropna(subset=[target, *features])
            test = g[g["cycle"] > 18].dropna(subset=[target, *features])
            if len(train) < 5 or len(test) < 5:
                continue
            y = test[target].to_numpy(float)
            yhat = fit_predict(train, test, features, target)
            baseline = np.repeat(float(train[target].mean()), len(test))
            y_all.extend(y)
            yhat_all.extend(yhat)
            base_all.extend(baseline)
        y_arr = np.asarray(y_all)
        yhat_arr = np.asarray(yhat_all)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "target": target,
                "validation": "within_route_forward_60_40",
                "n": int(len(y_arr)),
                "r2_vs_training_mean": r2_vs_baseline(y_arr, yhat_arr, np.asarray(base_all)),
                "spearman_y_yhat": float(spearmanr(y_arr, yhat_arr).statistic) if len(y_arr) > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def circular_shift_null(df: pd.DataFrame, n_perm: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    obs = spearmanr(df["route_centered_hazard"], df["route_centered_overload"]).statistic
    for i in range(n_perm):
        shifted = []
        for _, g in df.groupby("regime_id", sort=True):
            vals = g["route_centered_hazard"].to_numpy(float)
            shift = int(rng.integers(0, len(vals)))
            shifted.extend(np.roll(vals, shift))
        rho = spearmanr(shifted, df["route_centered_overload"].to_numpy(float)).statistic
        rows.append({"permutation": i, "null_spearman": float(rho), "observed_spearman": float(obs)})
    null = pd.DataFrame(rows)
    p = (np.sum(np.abs(null["null_spearman"]) >= abs(obs)) + 1.0) / (len(null) + 1.0)
    null["two_sided_p"] = float(p)
    return null


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["regime_id", "segment"], observed=True)
        .agg(
            n=("cycle", "count"),
            response_amplitude=("response_amplitude", "mean"),
            loop_activation_positive=("loop_activation_positive", "mean"),
            buffer_efficiency=("buffer_efficiency", "mean"),
            breathing_hazard_number=("breathing_hazard_number", "mean"),
            breathing_quality_factor=("breathing_quality_factor", "mean"),
            overload_asinh=("overload_asinh", "mean"),
            strong_overload_fraction=("strong_overload", "mean"),
        )
        .reset_index()
    )
    return out


def make_figure(df: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame, null: pd.DataFrame, seg: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["log_hazard_number"],
            g["overload_asinh"],
            s=24,
            marker=MARKERS[rid],
            color=COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.88,
            label=rid,
            zorder=3,
        )
    x = df["log_hazard_number"].to_numpy(float)
    y = df["overload_asinh"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    b = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 120)
    ax.plot(xx, b[0] * xx + b[1], color=INK, lw=0.85, zorder=4)
    row = corr[corr["predictor"] == "log_hazard_number"].iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={row.spearman_raw:.2f}$" + f"\nP={row.p_raw:.1e}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(r"breathing hazard, $\log_{10}{\cal H}_b$")
    ax.set_ylabel(r"hot overload, $\operatorname{asinh}\widehat{\Omega}$")
    ax.set_title("hazard combines inhale, loops and buffering")
    ax.legend(ncol=3, loc="lower right", handlelength=1.0, columnspacing=0.8)
    panel(ax, "a")
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["route_centered_hazard"],
            g["route_centered_overload"],
            s=24,
            marker=MARKERS[rid],
            color=COLORS[rid],
            edgecolor="white",
            lw=0.35,
            alpha=0.88,
            zorder=3,
        )
    x = df["route_centered_hazard"].to_numpy(float)
    y = df["route_centered_overload"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    b = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 120)
    ax.plot(xx, b[0] * xx + b[1], color=INK, lw=0.85, zorder=4)
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.text(
        0.05,
        0.95,
        rf"$\rho_{{within}}={row.spearman_within_route:.2f}$"
        + f"\nP={row.p_within_route:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.set_xlabel(r"route-centred $\log_{10}{\cal H}_b$")
    ax.set_ylabel("route-centred overload")
    ax.set_title("relation survives route centring")
    panel(ax, "b")
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    order = [
        "amplitude",
        "loop activation",
        "buffer efficiency",
        "hazard number",
        "amplitude + loop",
        "full breathing triplet",
    ]
    t = tests.set_index("model").loc[order]
    vals = t["r2_vs_training_mean"].to_numpy(float)
    ypos = np.arange(len(order))
    colors = [NEUTRAL, GOLD, BLUE, RED, VIOLET, INK]
    ax.barh(ypos, vals, height=0.62, color=colors, zorder=3)
    ax.axvline(0, color="#AEB6C0", lw=0.65)
    ax.set_yticks(ypos, ["amp.", "loops", "buffer", r"${\cal H}_b$", "amp.+\nloops", "triplet"])
    ax.set_xlabel(r"within-route forward $R^2$")
    ax.set_title("prediction tests a compact breathing state")
    panel(ax, "c")
    finish(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    plot = seg.copy()
    segment_order = {"early": 0, "middle": 1, "late": 2}
    plot["seg_order"] = plot["segment"].map(segment_order)
    plot = plot.sort_values(["regime_id", "seg_order"])
    cols = ["response_amplitude", "loop_activation_positive", "buffer_efficiency", "breathing_hazard_number", "overload_asinh"]
    z = plot[cols].apply(zscore, axis=0).to_numpy(float)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(np.arange(len(plot)), plot["regime_id"] + "-" + plot["segment"].str[0])
    ax.set_xticks(np.arange(len(cols)), ["inhale\nA", "loop\nL+", "buffer\neta", "hazard\nHb", "overload"])
    ax.set_title("breathing state changes with training stage")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(size=2, width=0.5)
    cbar.set_label("z-score", labelpad=2)
    panel(ax, "d")

    out = FIG / "nphys_fig37_breathing_quality_factor"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(corr: pd.DataFrame, tests: pd.DataFrame, null: pd.DataFrame, seg: pd.DataFrame) -> None:
    hazard = corr[corr["predictor"] == "log_hazard_number"].iloc[0]
    amp = corr[corr["predictor"] == "response_amplitude"].iloc[0]
    loop = corr[corr["predictor"] == "loop_activation_positive"].iloc[0]
    buffer = corr[corr["predictor"] == "buffer_efficiency"].iloc[0]
    t = tests.set_index("model")
    p_null = float(null["two_sided_p"].iloc[0])
    report = ROOT / "nature_physics_breathing_quality_factor.md"
    report.write_text(
        "# Breathing quality-factor audit\n\n"
        "This reserve audit asks whether the breathing metaphor can be converted into a compact, falsifiable state number. "
        "The tested number is not a new constitutive law; it is a dimensionless diagnostic for the three-route lagged subset:\n\n"
        r"\\[{\cal H}_b = A L_+/(\eta+\epsilon),\\]"
        "\n\n"
        "where `A` is the standardized cold-to-hot inhale amplitude, `L_+` is positive force-loop activation and `eta` is next-cold imprint efficiency. "
        "Large values mean that a cycle inhales strongly into the loop sector but exhales with poor buffering.\n\n"
        "## Main findings\n\n"
        f"- The hazard number correlates with hot overload across the lagged three-route subset (`rho={hazard.spearman_raw:.3f}`, `P={hazard.p_raw:.2e}`, n={int(hazard.n)}).\n"
        f"- After route centring, the relation remains positive (`rho={hazard.spearman_within_route:.3f}`, `P={hazard.p_within_route:.2e}`). It is comparable to amplitude alone (`rho={amp.spearman_within_route:.3f}`), captures the opposite sign of buffer efficiency (`rho={buffer.spearman_within_route:.3f}`), but does not exceed positive loop activation (`rho={loop.spearman_within_route:.3f}`).\n"
        f"- A circular-shift null that preserves route-wise cyclic order gives `P={p_null:.3g}` for the route-centred hazard-overload correlation.\n"
        f"- Within-route forward prediction gives R2={t.loc['hazard number','r2_vs_training_mean']:.3f} for the compact hazard number, R2={t.loc['loop activation','r2_vs_training_mean']:.3f} for loop activation alone, R2={t.loc['full breathing triplet','r2_vs_training_mean']:.3f} for the full `(A,L_+,eta)` triplet and R2={t.loc['amplitude + loop','r2_vs_training_mean']:.3f} without the explicit buffering term.\n\n"
        "## Manuscript-safe interpretation\n\n"
        "The audit strengthens the operational meaning of memory-induced breathing: overload is largest when a route combines a strong inhale, positive loop activation and inefficient exhalation. "
        "The single loop-activation coordinate remains the sharper predictor, so the breathing hazard should be used to explain modulation and observability rather than to replace the five-route force-loop mechanism or the dimensionless loop number `Psi`.\n\n"
        "## Boundary\n\n"
        "Use the result as a reserve or Extended Data mechanism panel unless the main text needs a sharper definition of breathing quality. "
        "Do not present `Hb` as a universal material parameter, and do not call its inverse a thermodynamic quality factor.\n",
        encoding="utf-8",
    )


def main() -> None:
    df = prepare()
    corr = correlation_rows(df)
    tests = forward_tests(df)
    null = circular_shift_null(df)
    seg = segment_summary(df)
    df.to_csv(SRC / "nphys_breathing_quality_factor_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_breathing_quality_factor_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_breathing_quality_factor_forward_tests.csv", index=False)
    null.to_csv(SRC / "nphys_breathing_quality_factor_circular_shift_null.csv", index=False)
    seg.to_csv(SRC / "nphys_breathing_quality_factor_route_segments.csv", index=False)
    make_figure(df, corr, tests, null, seg)
    write_report(corr, tests, null, seg)
    print(corr.to_string(index=False))
    print(tests.to_string(index=False))
    print(seg.to_string(index=False))
    print(f"circular_shift_p={null['two_sided_p'].iloc[0]:.4g}")


if __name__ == "__main__":
    main()
