#!/usr/bin/env python3
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

CYCLES = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"
ATLAS = SRC / "nphys_route_susceptibility_atlas.csv"

ROUTES = ["R1", "R3", "R5", "R6", "R6c"]
INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
ACCENT = "#8D3138"


COORDINATE_MODES = [
    ("route_z", "route-centred z", "linear"),
    ("route_robust_z", "route robust z", "linear-robust"),
    ("route_centered_raw", "route-centred raw", "linear"),
    ("global_z", "global z", "linear"),
    ("global_robust_z", "global robust z", "linear-robust"),
    ("route_rank", "route rank", "nonlinear rank"),
]


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=INK)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def standard_z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd else s * 0


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not mad:
        mad = s.std(ddof=1)
    return (s - med) / mad if mad else s * 0


def centred_rank(s: pd.Series) -> pd.Series:
    return (s.rank(method="average") - 0.5) / len(s) - 0.5


def exact_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rho = float(spearmanr(x_arr, y_arr).statistic)
    obs = abs(rho)
    hits = 0
    total = 0
    for perm in permutations(y_arr):
        trial = abs(float(spearmanr(x_arr, np.asarray(perm, dtype=float)).statistic))
        hits += int(trial >= obs - 1e-12)
        total += 1
    return rho, hits / total


def signed_area(points: np.ndarray, closed: bool) -> float:
    x = points[:, 0]
    y = points[:, 1]
    area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if closed:
        area += x[-1] * y[0] - x[0] * y[-1]
    return float(0.5 * area)


def transform(cycles: pd.DataFrame, mode: str) -> pd.DataFrame:
    d = cycles.copy()
    m = "force_h1_birth_force_share_cold"
    psi = "dimensionless_loop_number"
    if mode == "route_z":
        d["M"] = d["memory_coordinate"]
        d["Psi"] = d["hot_excitation_coordinate"]
    elif mode == "route_robust_z":
        d["M"] = d.groupby("regime_id")[m].transform(robust_z)
        d["Psi"] = d.groupby("regime_id")[psi].transform(robust_z)
    elif mode == "route_centered_raw":
        d["M"] = d[m] - d.groupby("regime_id")[m].transform("mean")
        d["Psi"] = d[psi] - d.groupby("regime_id")[psi].transform("mean")
    elif mode == "global_z":
        d["M"] = standard_z(d[m])
        d["Psi"] = standard_z(d[psi])
    elif mode == "global_robust_z":
        d["M"] = robust_z(d[m])
        d["Psi"] = robust_z(d[psi])
    elif mode == "route_rank":
        d["M"] = d.groupby("regime_id")[m].transform(centred_rank)
        d["Psi"] = d.groupby("regime_id")[psi].transform(centred_rank)
    else:
        raise ValueError(mode)
    return d


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = pd.read_csv(CYCLES)
    atlas = pd.read_csv(ATLAS).set_index("regime_id")
    rows = []
    for mode, label, family in COORDINATE_MODES:
        d = transform(cycles, mode)
        for rid, g in d.dropna(subset=["M", "Psi"]).groupby("regime_id", sort=True):
            g = g.sort_values("cycle")
            pts = g[["M", "Psi"]].to_numpy(float)
            rows.append(
                {
                    "coordinate_mode": mode,
                    "coordinate_label": label,
                    "coordinate_family": family,
                    "regime_id": rid,
                    "open_signed_area": signed_area(pts, closed=False),
                    "closed_signed_area": signed_area(pts, closed=True),
                    "net_displacement": float(np.linalg.norm(pts[-1] - pts[0])),
                    "mean_overload": float(atlas.loc[rid, "mean_overload"]),
                    "loop_gain": float(atlas.loc[rid, "loop_gain"]),
                    "S": float(atlas.loc[rid, "S"]),
                }
            )
    route = pd.DataFrame(rows)
    route["route_order"] = pd.Categorical(route["regime_id"], ROUTES, ordered=True)
    route = route.sort_values(["coordinate_mode", "route_order"]).drop(columns=["route_order"])

    corr_rows = []
    for (mode, label, family), g in route.groupby(["coordinate_mode", "coordinate_label", "coordinate_family"], sort=False):
        for metric in ["open_signed_area", "closed_signed_area", "net_displacement"]:
            for target in ["mean_overload", "loop_gain", "S"]:
                rho, p = exact_spearman(g[metric], g[target])
                corr_rows.append(
                    {
                        "coordinate_mode": mode,
                        "coordinate_label": label,
                        "coordinate_family": family,
                        "geometry_metric": metric,
                        "target": target,
                        "spearman": rho,
                        "exact_p_two_sided": p,
                    }
                )
    corr = pd.DataFrame(corr_rows)
    summary = (
        corr[corr["target"].eq("mean_overload")]
        .pivot(index="coordinate_label", columns="geometry_metric", values="spearman")
        .reset_index()
    )
    return route, corr, summary


def draw_heatmap(ax: plt.Axes, corr: pd.DataFrame) -> None:
    d = corr[corr["target"].eq("mean_overload")].copy()
    labels = [label for _, label, _ in COORDINATE_MODES]
    metrics = ["open_signed_area", "closed_signed_area", "net_displacement"]
    mat = np.zeros((len(labels), len(metrics)))
    for i, label in enumerate(labels):
        for j, metric in enumerate(metrics):
            mat[i, j] = d[(d["coordinate_label"] == label) & (d["geometry_metric"] == metric)]["spearman"].iloc[0]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["open\narea", "closed\narea", "net\ndispl."], rotation=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6.0, color="white" if abs(mat[i, j]) > 0.65 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.set_label(r"$\rho$ with mean overload", fontsize=6.0)
    ax.set_title("metric-coordinate robustness", loc="left", pad=4)


def draw_closed_area_routes(ax: plt.Axes, route: pd.DataFrame) -> None:
    keep = ["route-centred z", "global z", "global robust z", "route-centred raw"]
    d = route[route["coordinate_label"].isin(keep)].copy()
    x = np.arange(len(ROUTES))
    offsets = np.linspace(-0.24, 0.24, len(keep))
    for off, label in zip(offsets, keep):
        g = d[d["coordinate_label"] == label].set_index("regime_id").loc[ROUTES]
        vals = g["closed_signed_area"].to_numpy(float)
        scale = np.max(np.abs(vals)) or 1
        ax.plot(x + off, vals / scale, marker="o", ms=3.6, lw=0.8, label=label, alpha=0.85)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(ROUTES)
    ax.set_ylabel("normalised closed signed area")
    ax.set_title("same orientation sequence under rescaling", loc="left", pad=4)
    ax.legend(fontsize=5.5, loc="lower left")
    finish(ax, axis="y")


def draw_metric_boundary(ax: plt.Axes, corr: pd.DataFrame) -> None:
    d = corr[(corr["target"] == "mean_overload") & (corr["geometry_metric"] == "closed_signed_area")].copy()
    labels = [label for _, label, _ in COORDINATE_MODES]
    x = np.arange(len(labels))
    colors = ["#8D3138" if "rank" not in label else "#737D89" for label in labels]
    vals = [d[d["coordinate_label"] == label]["spearman"].iloc[0] for label in labels]
    ax.bar(x, vals, color=colors, width=0.66)
    ax.axhline(-1, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-1.1, 0.15)
    ax.set_ylabel(r"$\rho$ with overload")
    ax.set_title("rank coordinates bound the area claim", loc="left", pad=4)
    ax.text(
        0.04,
        0.92,
        "area is metric-coordinate diagnostic,\nnot arbitrary-coordinate invariant",
        transform=ax.transAxes,
        fontsize=5.9,
        color=MUTED,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
    )
    finish(ax, axis="y")


def build_figure(route: pd.DataFrame, corr: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1:])
    draw_heatmap(ax_a, corr)
    panel(ax_a, "a", x=-0.12)
    draw_closed_area_routes(ax_b, route)
    panel(ax_b, "b", x=-0.08)
    draw_metric_boundary(ax_c, corr)
    panel(ax_c, "c", x=-0.08)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig55_phase_circulation_coordinate_robustness.{ext}", **kwargs)
    plt.close(fig)


def write_report(route: pd.DataFrame, corr: pd.DataFrame) -> None:
    closed = corr[(corr["target"] == "mean_overload") & (corr["geometry_metric"] == "closed_signed_area")]
    linear = closed[~closed["coordinate_family"].str.contains("nonlinear")]
    exact_count = int((linear["spearman"].abs() >= 0.999).sum())
    out = ROOT / "nature_physics_phase_circulation_coordinate_robustness.md"
    lines = [
        "# Phase-circulation coordinate-robustness audit",
        "",
        "## Question",
        "",
        "Does the phase-circulation ordering depend on the particular route-centred z coordinates used in Fig. 5, or does it survive reasonable metric rescalings of the same memory/excitation variables?",
        "",
        "## Main result",
        "",
        f"The closed signed circulation area retained perfect overload ordering under {exact_count} of {len(linear)} linear or robust-linear coordinate definitions. The same ordering was not preserved by route-rank coordinates, which intentionally destroy metric spacing. The manuscript-safe statement is therefore metric-coordinate robustness, not invariance under arbitrary nonlinear coordinate transformations.",
        "",
        "## Route-level geometry by coordinate definition",
        "",
        route.round(5).to_markdown(index=False),
        "",
        "## Correlation summary",
        "",
        corr.round(5).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: signed circulation orientation is robust to reasonable centring and scaling choices for the reduced memory/excitation coordinates.",
        "",
        "Not allowed: circulation area is not a topological invariant, not coordinate-free, and not preserved under arbitrary monotone rank transformations.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig55_phase_circulation_coordinate_robustness.*`",
        "- `source_data/nphys_phase_circulation_coordinate_robustness_route.csv`",
        "- `source_data/nphys_phase_circulation_coordinate_robustness_correlations.csv`",
        "- `source_data/nphys_phase_circulation_coordinate_robustness_summary.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    route, corr, summary = build_tables()
    route.to_csv(SRC / "nphys_phase_circulation_coordinate_robustness_route.csv", index=False)
    corr.to_csv(SRC / "nphys_phase_circulation_coordinate_robustness_correlations.csv", index=False)
    summary.to_csv(SRC / "nphys_phase_circulation_coordinate_robustness_summary.csv", index=False)
    build_figure(route, corr)
    write_report(route, corr)
    print("Wrote phase-circulation coordinate-robustness audit")
    print(corr[(corr["target"] == "mean_overload") & (corr["geometry_metric"] == "closed_signed_area")][["coordinate_label", "spearman", "exact_p_two_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
