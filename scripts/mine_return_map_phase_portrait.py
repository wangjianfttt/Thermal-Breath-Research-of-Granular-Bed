#!/usr/bin/env python3
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INFILE = SRC / "nphys_dimensionless_loop_collapse_cycle_metrics.csv"
GAIN_CURVES = SRC / "nphys_nonreciprocal_transient_map_gain_curves.csv"
GAIN_METRICS = SRC / "nphys_nonreciprocal_transient_map_metrics.csv"
NORMAL_FORM = SRC / "nphys_return_map_normal_form_route_metrics.csv"
NORMAL_FORM_PARITY = SRC / "nphys_return_map_normal_form_parity_summary.csv"

COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#242A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
ACCENT = "#B6423E"
COOL = "#3D6B9C"


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.07) -> None:
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
    base_cols = [
        "force_h1_birth_force_share_cold",
        "dimensionless_loop_number",
        "overload_number",
        "force_p99_cold",
        "force_h1_birth_fraction_cold",
    ]
    for col in base_cols:
        df[f"next_{col}"] = df.groupby("regime_id")[col].shift(-1)
        df[f"{col}_wz"] = within_route_z(df, col)
        route_mean = df.groupby("regime_id")[col].transform("mean")
        route_std = df.groupby("regime_id")[col].transform("std").replace(0, np.nan)
        df[f"next_{col}_wz"] = (df[f"next_{col}"] - route_mean) / route_std

    df["memory_coordinate"] = df["force_h1_birth_force_share_cold_wz"]
    df["hot_excitation_coordinate"] = df["dimensionless_loop_number_wz"]
    df["next_memory_coordinate"] = df["next_force_h1_birth_force_share_cold_wz"]
    df["next_hot_excitation_coordinate"] = df["next_dimensionless_loop_number_wz"]
    return df


def fit_route_maps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = ["memory_coordinate", "hot_excitation_coordinate"]
    targets = ["next_memory_coordinate", "next_hot_excitation_coordinate"]
    rows = []
    pred_rows = []
    d = df.dropna(subset=coords + targets).copy()
    for rid, g in d.groupby("regime_id", sort=True):
        x = g[coords].to_numpy(float)
        y = g[targets].to_numpy(float)
        fit = LinearRegression().fit(x, y)
        yhat = fit.predict(x)
        eig = np.linalg.eigvals(fit.coef_)
        rows.append(
            {
                "regime_id": rid,
                "n": len(g),
                "A11_memory_to_memory": fit.coef_[0, 0],
                "A12_hot_to_memory": fit.coef_[0, 1],
                "A21_memory_to_hot": fit.coef_[1, 0],
                "A22_hot_to_hot": fit.coef_[1, 1],
                "determinant": float(np.linalg.det(fit.coef_)),
                "spectral_radius": float(np.max(np.abs(eig))),
                "eigenvalue_1_real": float(np.real(eig[0])),
                "eigenvalue_1_imag": float(np.imag(eig[0])),
                "eigenvalue_2_real": float(np.real(eig[1])),
                "eigenvalue_2_imag": float(np.imag(eig[1])),
                "r2_next_memory": r2_score(y[:, 0], yhat[:, 0]),
                "r2_next_hot_excitation": r2_score(y[:, 1], yhat[:, 1]),
                "mean_dimensionless_loop_number": g["dimensionless_loop_number"].mean(),
                "mean_overload_number": g["overload_number"].mean(),
            }
        )

    # Conservative forward tests: does the hot coordinate help later-cycle cold-state prediction?
    for target in [
        "next_force_p99_cold_wz",
        "next_force_h1_birth_fraction_cold_wz",
        "next_memory_coordinate",
    ]:
        g = df.dropna(subset=["memory_coordinate", "hot_excitation_coordinate", target]).copy()
        train = g["cycle"] <= 18
        test = g["cycle"] > 18
        x0 = g[["memory_coordinate"]].to_numpy(float)
        x1 = g[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
        y = g[target].to_numpy(float)
        base = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(x0[train], y[train])
        plus = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(x1[train], y[train])
        pred_rows.append(
            {
                "target": target,
                "validation": "within_route_forward_60_40",
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "r2_memory_only": r2_score(y[test], base.predict(x0[test])),
                "r2_memory_plus_hot_excitation": r2_score(y[test], plus.predict(x1[test])),
                "delta_r2": r2_score(y[test], plus.predict(x1[test])) - r2_score(y[test], base.predict(x0[test])),
                "coef_memory": plus.coef_[0],
                "coef_hot_excitation": plus.coef_[1],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def route_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    segment_bins = [0, 5, 15, 30]
    segment_labels = ["early", "middle", "late"]
    rows = []
    for rid, g in df.dropna(subset=["memory_coordinate", "hot_excitation_coordinate"]).groupby("regime_id", sort=True):
        g = g.sort_values("cycle")
        for lo, hi, label in zip(segment_bins[:-1], segment_bins[1:], segment_labels):
            seg = g[(g["cycle"] > lo) & (g["cycle"] <= hi)]
            if seg.empty:
                continue
            rows.append(
                {
                    "regime_id": rid,
                    "segment": label,
                    "cycle_low_exclusive": lo,
                    "cycle_high_inclusive": hi,
                    "n_cycles": int(len(seg)),
                    "median_memory_coordinate": float(np.nanmedian(seg["memory_coordinate"].to_numpy(float))),
                    "median_hot_excitation_coordinate": float(np.nanmedian(seg["hot_excitation_coordinate"].to_numpy(float))),
                    "iqr_memory_coordinate": float(np.nanpercentile(seg["memory_coordinate"], 75) - np.nanpercentile(seg["memory_coordinate"], 25)),
                    "iqr_hot_excitation_coordinate": float(np.nanpercentile(seg["hot_excitation_coordinate"], 75) - np.nanpercentile(seg["hot_excitation_coordinate"], 25)),
                }
            )
    return pd.DataFrame(rows)


def route_phase_displacements(phase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in phase.groupby("regime_id", sort=True):
        by_segment = g.set_index("segment")
        if "early" not in by_segment.index or "late" not in by_segment.index:
            continue
        early = by_segment.loc["early"]
        late = by_segment.loc["late"]
        dx = float(late["median_memory_coordinate"] - early["median_memory_coordinate"])
        dy = float(late["median_hot_excitation_coordinate"] - early["median_hot_excitation_coordinate"])
        row = {
            "regime_id": rid,
            "early_memory_coordinate": float(early["median_memory_coordinate"]),
            "early_hot_excitation_coordinate": float(early["median_hot_excitation_coordinate"]),
            "late_memory_coordinate": float(late["median_memory_coordinate"]),
            "late_hot_excitation_coordinate": float(late["median_hot_excitation_coordinate"]),
            "delta_memory_coordinate": dx,
            "delta_hot_excitation_coordinate": dy,
            "displacement_norm": float(np.hypot(dx, dy)),
            "displacement_angle_deg": float(np.degrees(np.arctan2(dy, dx))),
        }
        if "middle" in by_segment.index:
            middle = by_segment.loc["middle"]
            row["middle_memory_coordinate"] = float(middle["median_memory_coordinate"])
            row["middle_hot_excitation_coordinate"] = float(middle["median_hot_excitation_coordinate"])
            denom = dx * dx + dy * dy
            if denom > 0:
                mx = float(middle["median_memory_coordinate"] - early["median_memory_coordinate"])
                my = float(middle["median_hot_excitation_coordinate"] - early["median_hot_excitation_coordinate"])
                row["middle_projection_fraction"] = float(np.clip((mx * dx + my * dy) / denom, 0, 1))
            else:
                row["middle_projection_fraction"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    obs = abs(rho)
    hits = 0
    total = 0
    for perm in permutations(y):
        trial = abs(float(spearmanr(x, np.asarray(perm, dtype=float)).statistic))
        hits += int(trial >= obs - 1e-12)
        total += 1
    return rho, hits / total


def draw_vector_field(ax: plt.Axes, phase: pd.DataFrame, route_maps: pd.DataFrame) -> None:
    disp = route_phase_displacements(phase)
    order = (
        route_maps.sort_values("mean_overload_number")["regime_id"].tolist()
        if "mean_overload_number" in route_maps
        else sorted(disp["regime_id"].unique())
    )
    disp = disp.set_index("regime_id").loc[order].reset_index()
    overload = (
        route_maps.set_index("regime_id").loc[order, "mean_overload_number"]
        if "mean_overload_number" in route_maps
        else pd.Series(dtype=float)
    )

    y = np.arange(len(disp), dtype=float)
    height = 0.32
    dm = disp["delta_memory_coordinate"].to_numpy(float)
    loop_release = -disp["delta_hot_excitation_coordinate"].to_numpy(float)
    ax.axvline(0, color="#AEB6C0", lw=0.75, ls=(0, (3, 3)), zorder=1)
    for yi in y:
        ax.axhline(yi, color="#F1F3F5", lw=0.5, zorder=0)
    ax.barh(y + height / 2, dm, height=height, color="#3D6B9C", alpha=0.88, label=r"cold memory, $\Delta M$", zorder=3)
    ax.barh(y - height / 2, loop_release, height=height, color="#B6423E", alpha=0.82, label=r"loop release, $-\Delta\Psi$", zorder=3)

    if len(overload) == len(disp):
        x = np.maximum(loop_release, 0.0)
        y = overload.to_numpy(float)
        rho, p = exact_spearman(x, y)
        stat = rf"$[-\Delta\Psi]_+$ orders overload: $\rho={rho:.2f}$, $P={p:.3f}$"
    else:
        stat = r"$[-\Delta\Psi]_+$ orders overload"
    ax.text(0.03, 0.97, stat, transform=ax.transAxes, ha="left", va="top", fontsize=6.15, color=INK)
    ax.set_yticks(np.arange(len(disp)))
    ax.set_yticklabels(disp["regime_id"])
    ax.set_ylim(len(disp) - 0.35, -0.65)
    ax.set_xlim(-2.55, 2.75)
    ax.set_xlabel("early-to-late displacement")
    ax.set_ylabel(r"route, ordered by $\langle\widehat{\Omega}\rangle$")
    ax.set_title("breathing-displacement spectrum", loc="left", pad=4)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.16),
        fontsize=5.8,
        handlelength=1.2,
        borderaxespad=0.0,
    )
    finish(ax, axis="x")


def draw_route_map_heat(ax: plt.Axes, route_maps: pd.DataFrame) -> None:
    mat = route_maps.set_index("regime_id")[
        ["A11_memory_to_memory", "A12_hot_to_memory", "A21_memory_to_hot", "A22_hot_to_hot"]
    ]
    im = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-1.1, vmax=1.1, aspect="auto")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["M->M", r"$\Psi$->M", r"M->$\Psi$", r"$\Psi$->$\Psi$"], rotation=35, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.9, color="white" if abs(v) > 0.55 else INK)
    ax.set_title("linearised route maps", loc="left", pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=5.8, length=2)


def draw_stability(ax: plt.Axes, route_maps: pd.DataFrame) -> None:
    g = route_maps.sort_values("mean_overload_number").copy()
    for _, row in g.iterrows():
        rid = row["regime_id"]
        ax.scatter(
            row["mean_overload_number"],
            row["spectral_radius"],
            s=42 + 260 * max(row["mean_dimensionless_loop_number"], 0),
            color=COLORS.get(rid, MUTED),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(row["mean_overload_number"] + 0.12, row["spectral_radius"], rid, color=COLORS.get(rid, MUTED), fontsize=6.5, va="center")
    ax.axhline(1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel(r"mean overload number $\langle\widehat{\Omega}\rangle$")
    ax.set_ylabel("spectral radius of map")
    ax.set_title("route-conditioned stability", loc="left", pad=4)
    ax.set_ylim(0.45, 1.05)
    finish(ax)


def draw_gain_curves(ax: plt.Axes) -> None:
    curves = pd.read_csv(GAIN_CURVES)
    metrics = pd.read_csv(GAIN_METRICS).set_index("regime_id")
    normal = pd.read_csv(NORMAL_FORM).set_index("regime_id") if NORMAL_FORM.exists() else pd.DataFrame()
    parity = pd.read_csv(NORMAL_FORM_PARITY) if NORMAL_FORM_PARITY.exists() else pd.DataFrame()
    for rid, g in curves.groupby("regime_id", sort=True):
        color = COLORS.get(rid, MUTED)
        lw = 1.35 if rid in {"R3", "R6c"} else 0.95
        alpha = 0.95 if rid in {"R3", "R6c"} else 0.72
        ax.plot(g["step"], g["normalized_transient_gain"], color=color, lw=lw, alpha=alpha)
        last = g.iloc[-1]
        ax.text(
            last["step"] + 0.25,
            last["normalized_transient_gain"],
            rid,
            color=color,
            fontsize=6.1,
            va="center",
        )
    ax.axhline(1.0, color="#AEB6C0", lw=0.75, ls=(0, (3, 3)))
    r3 = metrics.loc["R3"]
    if not normal.empty and "R3" in normal.index:
        r3_flip = normal.loc["R3", "dominant_eigenvalue"]
        r3_text = rf"R3 near flip: $\lambda_1={r3_flip:.2f}$; hidden gain={r3.peak_normalized_gain:.2f}"
    else:
        r3_text = rf"R3: $\rho(A)$={r3.spectral_radius:.2f}, $\|A\|_2$={r3.one_step_gain:.2f}"
    ax.text(
        0.04,
        0.94,
        r3_text,
        transform=ax.transAxes,
        fontsize=6.4,
        color=COLORS["R3"],
        va="top",
    )
    if not parity.empty:
        loop_even = parity[(parity["series"] == "loop activation") & (parity["parity"] == "even")]
        loop_odd = parity[(parity["series"] == "loop activation") & (parity["parity"] == "odd")]
        if not loop_even.empty and not loop_odd.empty:
            ax.text(
                0.04,
                0.84,
                rf"loop memory: even $\rho={loop_even.iloc[0]['mean_pooled_spearman']:.2f}$, "
                rf"odd $\rho={loop_odd.iloc[0]['mean_pooled_spearman']:.2f}$",
                transform=ax.transAxes,
                fontsize=6.15,
                color=MUTED,
                va="top",
            )
    ax.text(
        0.04,
        0.75,
        "all routes decay spectrally;\nselected directions are amplified",
        transform=ax.transAxes,
        fontsize=6.2,
        color=MUTED,
        va="top",
    )
    ax.set_xlim(1, 15.8)
    ax.set_ylim(0.95, 1.62)
    ax.set_xlabel("map iteration")
    ax.set_ylabel(r"$\|A^k\|_2/\rho(A)^k$")
    ax.set_title("transient gain despite stable eigenvalues", loc="left", pad=4)
    finish(ax)


def build_figure(df: pd.DataFrame, route_maps: pd.DataFrame, pred: pd.DataFrame, phase: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])
    draw_vector_field(ax_a, phase, route_maps)
    panel(ax_a, "a", x=-0.10)
    draw_route_map_heat(ax_b, route_maps)
    panel(ax_b, "b")
    draw_stability(ax_c, route_maps)
    panel(ax_c, "c")
    draw_gain_curves(ax_d)
    panel(ax_d, "d", x=-0.07)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig17_return_map_phase_portrait.{ext}", **kwargs)
    plt.close(fig)


def write_report(route_maps: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = ROOT / "nature_physics_return_map_phase_portrait_report.md"
    lines = [
        "# Return-map phase portrait audit",
        "",
        "This audit asks whether the cold memory coordinate and the hot dimensionless loop excitation can be read as two projections of one cycle-to-cycle map.",
        "",
        "## Route linearisation",
        "",
        route_maps.round(4).to_markdown(index=False),
        "",
        "All point-estimate spectral radii are below unity, so the fitted route maps are dissipative rather than explosively unstable. This should be framed as a route-conditioned diagnostic, not as a universal constitutive law.",
        "",
        "## Forward imprint tests",
        "",
        pred.round(4).to_markdown(index=False),
        "",
        "The hot coordinate modestly improves some next-cold imprint predictions but not all. The manuscript-safe claim is therefore that the two coordinates form a useful low-dimensional phase portrait; predictive transfer remains route-conditioned.",
        "",
        "## Transient-gain display",
        "",
        "Main Fig. 5 now uses a breathing-displacement spectrum in panel a and the non-reciprocal transient-map gain curves in panel d. Panel a encodes each route by two early-to-late displacement components, `Delta M` and `-Delta Psi`, after ordering routes by mean overload. This avoids the visually crowded point-line phase portrait while preserving the physical decomposition of each network breath into cold-memory shift and hot-loop release. The positive hot-loop release component `[-Delta Psi]_+` is annotated as the overload-ordering component, but remains a five-route geometric diagnostic rather than a calibrated material law. Panel d annotates the R3 near-flip normal-form result and the even/odd loop-memory contrast while preserving the manuscript-safe claim: route maps can have spectral radius below unity while still amplifying selected directions. The lagged-imprint prediction tests remain in source data and in the report as a boundary audit.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare_table()
    route_maps, pred = fit_route_maps(df)
    phase = route_phase_summary(df)
    df.to_csv(SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv", index=False)
    route_maps.to_csv(SRC / "nphys_return_map_phase_portrait_route_maps.csv", index=False)
    pred.to_csv(SRC / "nphys_return_map_phase_portrait_prediction_tests.csv", index=False)
    phase.to_csv(SRC / "nphys_return_map_phase_portrait_route_phase_summary.csv", index=False)
    route_phase_displacements(phase).to_csv(SRC / "nphys_return_map_phase_portrait_route_displacements.csv", index=False)
    build_figure(df, route_maps, pred, phase)
    write_report(route_maps, pred)
    print("Wrote return-map phase portrait products")
    print(route_maps[["regime_id", "spectral_radius", "r2_next_memory", "r2_next_hot_excitation", "mean_overload_number"]].round(3).to_string(index=False))
    print(pred[["target", "r2_memory_only", "r2_memory_plus_hot_excitation", "delta_r2"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
