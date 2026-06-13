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
INFILE = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
LOOP = "#B6423E"
PSI = "#D98C3A"
TAIL = "#7E6AAE"
COLD = "#3D6B9C"
ROUTE_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
ROUTE_MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}


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


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def route_centered_rho(d: pd.DataFrame, x: str, y: str) -> float:
    xx = d[x] - d.groupby("regime_id")[x].transform("mean")
    yy = d[y] - d.groupby("regime_id")[y].transform("mean")
    return safe_spearman(xx.to_numpy(float), yy.to_numpy(float))


def load_table() -> pd.DataFrame:
    df = pd.read_csv(INFILE).sort_values(["regime_id", "cycle"]).copy()
    df["overload_asinh"] = np.arcsinh(df["overload_number"] / 2.0)
    for col, out in [
        ("dimensionless_loop_number", "psi_z"),
        ("loop_activation", "loop_z"),
        ("top5_activation", "tail_z"),
        ("force_h1_birth_force_share_cold", "cold_loop_z"),
        ("overload_asinh", "overload_z"),
    ]:
        df[out] = df.groupby("regime_id")[col].transform(zscore)
    threshold = df.groupby("regime_id")["overload_asinh"].transform(lambda s: s.quantile(0.80))
    df["rare_overload"] = (df["overload_asinh"] >= threshold).astype(int)
    return df


def lag_join(df: pd.DataFrame, predictor: str, lag: int) -> pd.DataFrame:
    parts = []
    for rid, g in df.groupby("regime_id", sort=True):
        g = g.sort_values("cycle").copy()
        out = pd.DataFrame(
            {
                "regime_id": rid,
                "cycle": g["cycle"].iloc[lag:].to_numpy(int) if lag else g["cycle"].to_numpy(int),
                "predictor": g[predictor].iloc[: len(g) - lag].to_numpy(float) if lag else g[predictor].to_numpy(float),
                "overload_z": g["overload_z"].iloc[lag:].to_numpy(float) if lag else g["overload_z"].to_numpy(float),
                "rare_overload": g["rare_overload"].iloc[lag:].to_numpy(int) if lag else g["rare_overload"].to_numpy(int),
            }
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True).dropna()


def circular_null(d: pd.DataFrame, n_null: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    obs = route_centered_rho(d, "predictor", "overload_z")
    null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        shifted = []
        for _, g in d.groupby("regime_id", sort=True):
            vals = g["predictor"].to_numpy(float)
            shifted.extend(np.roll(vals, int(rng.integers(0, len(vals)))))
        tmp = d.copy()
        tmp["predictor_shifted"] = shifted
        null[i] = route_centered_rho(tmp.rename(columns={"predictor_shifted": "predictor_null"}), "predictor_null", "overload_z")
    finite = null[np.isfinite(null)]
    p = float((np.sum(np.abs(finite) >= abs(obs)) + 1) / (len(finite) + 1))
    return p, float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def lag_spectrum(df: pd.DataFrame, max_lag: int = 8, n_null: int = 2000) -> pd.DataFrame:
    specs = [
        ("Psi", "psi_z"),
        ("loop activation", "loop_z"),
        ("top-5% tail", "tail_z"),
        ("cold loop memory", "cold_loop_z"),
    ]
    rows = []
    for label, col in specs:
        for lag in range(max_lag + 1):
            d = lag_join(df, col, lag)
            route_rhos = []
            route_aucs = []
            for _, g in d.groupby("regime_id", sort=True):
                route_rhos.append(safe_spearman(g["predictor"].to_numpy(float), g["overload_z"].to_numpy(float)))
                if g["rare_overload"].nunique() == 2:
                    route_aucs.append(float(roc_auc_score(g["rare_overload"], g["predictor"])))
            p, q025, q975 = circular_null(d, n_null=n_null, seed=700 + 41 * lag + len(label))
            pooled_auc = roc_auc_score(d["rare_overload"], d["predictor"]) if d["rare_overload"].nunique() == 2 else np.nan
            rows.append(
                {
                    "predictor": label,
                    "predictor_column": col,
                    "lag_cycles": lag,
                    "n": len(d),
                    "route_centered_spearman": route_centered_rho(d, "predictor", "overload_z"),
                    "route_mean_spearman": float(np.nanmean(route_rhos)),
                    "route_min_spearman": float(np.nanmin(route_rhos)),
                    "route_max_spearman": float(np.nanmax(route_rhos)),
                    "pooled_rare_auc": float(pooled_auc),
                    "route_mean_rare_auc": float(np.nanmean(route_aucs)),
                    "route_min_rare_auc": float(np.nanmin(route_aucs)),
                    "circular_shift_p": p,
                    "null_q025": q025,
                    "null_q975": q975,
                }
            )
    return pd.DataFrame(rows)


def design_matrix(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    cols = ["psi_z", "loop_z", "tail_z", "cold_loop_z"]
    parts = []
    for rid, g in df.groupby("regime_id", sort=True):
        g = g.sort_values("cycle").copy()
        for col in cols:
            for lag in range(max_lag + 1):
                g[f"{col}_lag{lag}"] = g[col].shift(lag)
        keep = ["regime_id", "cycle", "overload_z"] + [f"{col}_lag{lag}" for col in cols for lag in range(max_lag + 1)]
        parts.append(g[keep])
    return pd.concat(parts, ignore_index=True).dropna()


def transfer_tests(df: pd.DataFrame, max_lag: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = design_matrix(df, max_lag=max_lag)
    models = {
        "same-cycle Psi": ["psi_z_lag0"],
        "Psi memory kernel": [f"psi_z_lag{i}" for i in range(max_lag + 1)],
        "loop memory kernel": [f"loop_z_lag{i}" for i in range(max_lag + 1)],
        "force-tail memory kernel": [f"tail_z_lag{i}" for i in range(max_lag + 1)],
        "cold-loop memory kernel": [f"cold_loop_z_lag{i}" for i in range(max_lag + 1)],
        "loop plus tail kernel": [f"loop_z_lag{i}" for i in range(max_lag + 1)] + [f"tail_z_lag{i}" for i in range(max_lag + 1)],
    }
    rows = []
    coefs = []
    for name, features in models.items():
        y_all = []
        yh_all = []
        base_all = []
        for left_out in sorted(d["regime_id"].unique()):
            train = d[d["regime_id"] != left_out].copy()
            test = d[d["regime_id"] == left_out].copy()
            scaler = StandardScaler().fit(train[features].to_numpy(float))
            x_train = scaler.transform(train[features].to_numpy(float))
            x_test = scaler.transform(test[features].to_numpy(float))
            y_train = train["overload_z"].to_numpy(float)
            y_test = test["overload_z"].to_numpy(float)
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x_train, y_train)
            yhat = model.predict(x_test)
            y_all.extend(y_test)
            yh_all.extend(yhat)
            base_all.extend(np.repeat(float(y_train.mean()), len(y_test)))
            for feature, coef in zip(features, model.coef_):
                coefs.append({"model": name, "left_out_route": left_out, "feature": feature, "coefficient": float(coef)})
        y = np.asarray(y_all)
        yh = np.asarray(yh_all)
        base = np.asarray(base_all)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": len(y),
                "r2_vs_training_mean": float(1.0 - np.sum((y - yh) ** 2) / np.sum((y - base) ** 2)),
                "spearman_y_yhat": safe_spearman(y, yh),
            }
        )
    coef = pd.DataFrame(coefs)
    coef["lag_cycles"] = coef["feature"].str.extract(r"lag(\d+)").astype(int)
    coef["base_variable"] = coef["feature"].str.replace(r"_lag\d+", "", regex=True)
    coef_summary = coef.groupby(["model", "base_variable", "lag_cycles"], as_index=False)["coefficient"].mean()
    return pd.DataFrame(rows), coef_summary


def parity_summary(df: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    rows = []
    for series, col in [("Psi", "psi_z"), ("loop activation", "loop_z"), ("top-5% tail", "tail_z"), ("overload", "overload_z")]:
        for lag in range(1, max_lag + 1):
            rhos = []
            for _, g in df.groupby("regime_id", sort=True):
                vals = g.sort_values("cycle")[col].to_numpy(float)
                if len(vals) > lag:
                    rhos.append(safe_spearman(vals[:-lag], vals[lag:]))
            rows.append(
                {
                    "series": series,
                    "lag_cycles": lag,
                    "parity": "even" if lag % 2 == 0 else "odd",
                    "route_mean_spearman": float(np.nanmean(rhos)),
                    "route_min_spearman": float(np.nanmin(rhos)),
                    "route_max_spearman": float(np.nanmax(rhos)),
                }
            )
    return pd.DataFrame(rows)


def route_summary(df: pd.DataFrame, spectrum: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("regime_id", sort=True):
        lag2 = lag_join(g, "loop_z", 2)
        rows.append(
            {
                "regime_id": rid,
                "n_cycles": len(g),
                "mean_overload_asinh": float(g["overload_asinh"].mean()),
                "mean_dimensionless_loop_number": float(g["dimensionless_loop_number"].mean()),
                "lag0_loop_rho": safe_spearman(g["loop_z"].to_numpy(float), g["overload_z"].to_numpy(float)),
                "lag2_loop_rho": safe_spearman(lag2["predictor"].to_numpy(float), lag2["overload_z"].to_numpy(float)),
                "rare_event_rate": float(g["rare_overload"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figure(spectrum: pd.DataFrame, tests: pd.DataFrame, coefs: pd.DataFrame, parity: pd.DataFrame, route: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    color_map = {"Psi": PSI, "loop activation": LOOP, "top-5% tail": TAIL, "cold loop memory": COLD}
    for name in ["Psi", "loop activation", "top-5% tail", "cold loop memory"]:
        g = spectrum[spectrum["predictor"] == name].sort_values("lag_cycles")
        if name == "Psi":
            ax_a.plot(
                g["lag_cycles"],
                g["route_centered_spearman"],
                marker="s",
                ms=3.0,
                lw=0.85,
                ls=(0, (2.2, 1.5)),
                mfc="white",
                mec=color_map[name],
                color=color_map[name],
                alpha=0.78,
                label=r"$\Psi$",
            )
        else:
            ax_a.plot(g["lag_cycles"], g["route_centered_spearman"], marker="o", ms=3.1, lw=1.05, color=color_map[name], label=name)
        if name in {"Psi", "loop activation"}:
            ax_a.fill_between(g["lag_cycles"], g["null_q025"], g["null_q975"], color=color_map[name], alpha=0.08, lw=0)
    sig = spectrum[(spectrum["predictor"].isin(["Psi", "loop activation"])) & (spectrum["circular_shift_p"] < 0.05)]
    ax_a.scatter(sig["lag_cycles"], sig["route_centered_spearman"] + 0.055, marker="*", s=18, color=LOOP, zorder=5)
    ax_a.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_a.set_xlabel("lag before overload cycle")
    ax_a.set_ylabel(r"route-centred $\rho$")
    ax_a.set_title("five-route memory spectrum", loc="left", pad=4)
    ax_a.legend(loc="lower left", fontsize=5.7, handlelength=1.4, ncol=2)
    finish(ax_a)
    panel(ax_a, "a")

    order = ["same-cycle Psi", "Psi memory kernel", "loop memory kernel", "force-tail memory kernel", "cold-loop memory kernel", "loop plus tail kernel"]
    t = tests.set_index("model").loc[order].reset_index()
    y = np.arange(len(t))
    colors = [PSI, PSI, LOOP, TAIL, COLD, LOOP]
    ax_b.barh(y, t["r2_vs_training_mean"], color=colors, alpha=0.88)
    ax_b.axvline(0, color="#AEB6C0", lw=0.7)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(t["model"], fontsize=6.0)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"leave-one-route $R^2$")
    ax_b.set_title("distributed-lag transfer", loc="left", pad=4)
    finish(ax_b, axis="x")
    panel(ax_b, "b")

    for model, color, label in [("Psi memory kernel", PSI, r"$\Psi$ kernel"), ("loop memory kernel", LOOP, "loop kernel"), ("force-tail memory kernel", TAIL, "tail kernel")]:
        g = coefs[(coefs["model"] == model) & (coefs["base_variable"].isin(["psi_z", "loop_z", "tail_z"]))].copy()
        if model == "Psi memory kernel":
            g = g[g["base_variable"] == "psi_z"]
        elif model == "loop memory kernel":
            g = g[g["base_variable"] == "loop_z"]
        else:
            g = g[g["base_variable"] == "tail_z"]
        ax_c.plot(g["lag_cycles"], g["coefficient"], marker="o", ms=3.2, lw=1.1, color=color, label=label)
    ax_c.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_c.set_xlabel("lag in kernel")
    ax_c.set_ylabel("mean ridge coefficient")
    ax_c.set_title("which past breaths write risk?", loc="left", pad=4)
    ax_c.legend(fontsize=5.9, loc="best")
    finish(ax_c)
    panel(ax_c, "c")

    p = parity[parity["series"].isin(["loop activation", "overload"])].copy()
    for series, color in [("loop activation", LOOP), ("overload", INK)]:
        g = p[p["series"] == series].sort_values("lag_cycles")
        ax_d.plot(g["lag_cycles"], g["route_mean_spearman"], marker="o", ms=3.0, lw=1.05, color=color, label=series)
        ax_d.fill_between(g["lag_cycles"], g["route_min_spearman"], g["route_max_spearman"], color=color, alpha=0.05, lw=0)
    ax_d.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax_d.set_xlabel("cycle lag")
    ax_d.set_ylabel("route-mean autocorrelation")
    ax_d.set_title("even-lag route rhythm", loc="left", pad=4)
    ax_d.legend(fontsize=5.8, loc="lower right")
    finish(ax_d)
    panel(ax_d, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig49_five_route_breath_memory_spectrum.{ext}", **kwargs)
    plt.close(fig)


def write_report(spectrum: pd.DataFrame, tests: pd.DataFrame, coefs: pd.DataFrame, parity: pd.DataFrame, route: pd.DataFrame) -> None:
    loop2 = spectrum[(spectrum["predictor"] == "loop activation") & (spectrum["lag_cycles"] == 2)].iloc[0]
    psi2 = spectrum[(spectrum["predictor"] == "Psi") & (spectrum["lag_cycles"] == 2)].iloc[0]
    tail2 = spectrum[(spectrum["predictor"] == "top-5% tail") & (spectrum["lag_cycles"] == 2)].iloc[0]
    same = tests[tests["model"] == "same-cycle Psi"].iloc[0]
    loop_kernel = tests[tests["model"] == "loop memory kernel"].iloc[0]
    tail_kernel = tests[tests["model"] == "force-tail memory kernel"].iloc[0]
    psi_kernel = tests[tests["model"] == "Psi memory kernel"].iloc[0]
    lines = [
        "# Five-route breath-memory spectrum audit",
        "",
        "Purpose: test whether the breathing-memory language survives when restricted to the strongest five-route true-force dataset, rather than the older three-route breathing-state join.",
        "",
        "## Main findings",
        "",
        f"- Lag-2 loop activation predicts route-centred future overload with rho={loop2.route_centered_spearman:.3f} and circular-shift P={loop2.circular_shift_p:.4f}.",
        f"- Lag-2 Psi gives rho={psi2.route_centered_spearman:.3f}, P={psi2.circular_shift_p:.4f}; lag-2 top-5% tail gives rho={tail2.route_centered_spearman:.3f}.",
        f"- Leave-one-route transfer gives R2={same.r2_vs_training_mean:.3f} for same-cycle Psi, R2={psi_kernel.r2_vs_training_mean:.3f} for the Psi memory kernel, R2={loop_kernel.r2_vs_training_mean:.3f} for the loop memory kernel and R2={tail_kernel.r2_vs_training_mean:.3f} for the force-tail memory kernel.",
        "",
        "Interpretation: the same-cycle dimensionless loop number remains the primary overload coordinate, but the five-route true-force data also carry a finite-cycle loop memory at even lags. This supports the operational breathing picture as a dissipative, route-conditioned memory map. It does not establish a universal breathing frequency or a monotonic one-cycle alarm.",
        "",
        "## Lag spectrum",
        "",
        spectrum.round(4).to_markdown(index=False),
        "",
        "## Distributed-lag transfer",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Mean kernel coefficients",
        "",
        coefs.round(4).to_markdown(index=False),
        "",
        "## Even/odd route rhythm",
        "",
        parity.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        route.round(4).to_markdown(index=False),
        "",
        "Allowed wording: the five-route force-loop sector has a short, even-lag memory spectrum; lagged loop activation raises future overload risk but remains weaker than same-cycle loop activation.",
        "",
        "Not allowed: do not call this a universal frequency, critical slowing down, period doubling or a stand-alone forecasting law.",
    ]
    (ROOT / "nature_physics_five_route_breath_memory_spectrum.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_table()
    spectrum = lag_spectrum(df)
    tests, coefs = transfer_tests(df)
    parity = parity_summary(df)
    route = route_summary(df, spectrum)
    df.to_csv(SRC / "nphys_five_route_breath_memory_spectrum_cycle_metrics.csv", index=False)
    spectrum.to_csv(SRC / "nphys_five_route_breath_memory_spectrum_lag_scan.csv", index=False)
    tests.to_csv(SRC / "nphys_five_route_breath_memory_spectrum_transfer_tests.csv", index=False)
    coefs.to_csv(SRC / "nphys_five_route_breath_memory_spectrum_kernel_coefficients.csv", index=False)
    parity.to_csv(SRC / "nphys_five_route_breath_memory_spectrum_parity.csv", index=False)
    route.to_csv(SRC / "nphys_five_route_breath_memory_spectrum_route_summary.csv", index=False)
    make_figure(spectrum, tests, coefs, parity, route)
    write_report(spectrum, tests, coefs, parity, route)
    print("Wrote five-route breath-memory spectrum products")
    print(spectrum[spectrum["lag_cycles"].isin([0, 1, 2, 4])].round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
