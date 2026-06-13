#!/usr/bin/env python3
from __future__ import annotations

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
COLORS = {"R1": "#345995", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8D3138"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"


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


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def load_points() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    cycles = pd.read_csv(CYCLES)
    atlas = pd.read_csv(ATLAS).set_index("regime_id")
    points = {}
    rows = []
    for rid, g in cycles.dropna(subset=["memory_coordinate", "hot_excitation_coordinate"]).groupby("regime_id", sort=True):
        g = g.sort_values("cycle")
        pts = g[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
        points[rid] = pts
        rows.append(
            {
                "regime_id": rid,
                "observed_signed_circulation_area": signed_area(pts),
                "mean_overload": float(atlas.loc[rid, "mean_overload"]),
                "loop_gain": float(atlas.loc[rid, "loop_gain"]),
                "S": float(atlas.loc[rid, "S"]),
            }
        )
    obs = pd.DataFrame(rows)
    obs["route_order"] = pd.Categorical(obs["regime_id"], ROUTES, ordered=True)
    return points, obs.sort_values("route_order").drop(columns=["route_order"])


def run_null(points: dict[str, np.ndarray], obs: pd.DataFrame, n_shuffle: int = 50_000, seed: int = 20260613) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    overload = obs.set_index("regime_id").loc[ROUTES, "mean_overload"].to_numpy(float)
    observed_areas = obs.set_index("regime_id").loc[ROUTES, "observed_signed_circulation_area"].to_numpy(float)
    observed_rho = float(spearmanr(observed_areas, overload).statistic)

    rho_rows = []
    route_rows = []
    route_samples = {rid: [] for rid in ROUTES}
    for k in range(n_shuffle):
        areas = []
        for rid in ROUTES:
            pts = points[rid]
            shuffled = pts[rng.permutation(len(pts))]
            area = signed_area(shuffled)
            areas.append(area)
            if k < 5000:
                route_samples[rid].append(area)
        rho_rows.append(
            {
                "shuffle_id": k,
                "spearman_signed_area_overload": float(spearmanr(areas, overload).statistic),
            }
        )
    rho_null = pd.DataFrame(rho_rows)
    for rid in ROUTES:
        samples = np.asarray(route_samples[rid], dtype=float)
        observed = float(obs.loc[obs["regime_id"] == rid, "observed_signed_circulation_area"].iloc[0])
        route_rows.append(
            {
                "regime_id": rid,
                "observed_signed_circulation_area": observed,
                "null_mean": float(samples.mean()),
                "null_sd": float(samples.std(ddof=1)),
                "null_q025": float(np.quantile(samples, 0.025)),
                "null_q975": float(np.quantile(samples, 0.975)),
                "z_vs_null": float((observed - samples.mean()) / samples.std(ddof=1)),
            }
        )
    summary = pd.DataFrame(
        [
            {
                "test": "cycle_order_null_signed_area_overload",
                "n_shuffle": n_shuffle,
                "observed_spearman": observed_rho,
                "one_sided_p_rho_le_observed": float(np.mean(rho_null["spearman_signed_area_overload"] <= observed_rho + 1e-12)),
                "two_sided_p_abs_ge_observed": float(np.mean(np.abs(rho_null["spearman_signed_area_overload"]) >= abs(observed_rho) - 1e-12)),
                "null_median": float(rho_null["spearman_signed_area_overload"].median()),
                "null_q025": float(np.quantile(rho_null["spearman_signed_area_overload"], 0.025)),
                "null_q975": float(np.quantile(rho_null["spearman_signed_area_overload"], 0.975)),
            }
        ]
    )
    return rho_null, pd.DataFrame(route_rows), summary


def draw_route_areas(ax: plt.Axes, obs: pd.DataFrame, route_null: pd.DataFrame) -> None:
    d = obs.merge(route_null, on="regime_id")
    x = np.arange(len(d))
    ax.vlines(x, d["null_q025"], d["null_q975"], color="#B9C2CD", lw=2.0, zorder=1)
    ax.scatter(x, d["null_mean"], s=20, color="#AEB6C0", zorder=2, label="cycle-order null")
    for i, row in d.iterrows():
        rid = row["regime_id"]
        ax.scatter(i, row["observed_signed_circulation_area_x"], s=42, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(d["regime_id"])
    ax.set_ylabel("signed circulation area")
    ax.set_title("observed orientation is route ordered", loc="left", pad=4)
    ax.text(0.04, 0.94, "grey: random cycle order\ncolor: measured cycle order", transform=ax.transAxes, fontsize=6.1, color=MUTED, va="top")
    finish(ax, axis="y")


def draw_null_hist(ax: plt.Axes, rho_null: pd.DataFrame, summary: pd.DataFrame) -> None:
    vals = rho_null["spearman_signed_area_overload"].to_numpy(float)
    obs = float(summary["observed_spearman"].iloc[0])
    p = float(summary["one_sided_p_rho_le_observed"].iloc[0])
    bins = np.arange(-1.05, 1.051, 0.1)
    ax.hist(vals, bins=bins, color="#D7DDE4", edgecolor="white", lw=0.4)
    ax.axvline(obs, color="#8D3138", lw=1.4)
    ax.text(0.04, 0.94, rf"observed $\rho={obs:.2f}$" "\n" rf"cycle-order null $P={p:.4f}$", transform=ax.transAxes, fontsize=6.4, color=INK, va="top")
    ax.set_xlabel("Spearman(signed area, overload)")
    ax.set_ylabel("shuffle count")
    ax.set_title("perfect ordering is cycle-order dependent", loc="left", pad=4)
    finish(ax, axis="y")


def draw_route_order(ax: plt.Axes, obs: pd.DataFrame) -> None:
    d = obs.sort_values("mean_overload").copy()
    x = np.arange(len(d))
    ax.plot(x, d["observed_signed_circulation_area"], color="#AEB6C0", lw=0.8, zorder=1)
    for i, row in d.iterrows():
        rid = row["regime_id"]
        j = list(d.index).index(i)
        ax.scatter(j, row["observed_signed_circulation_area"], s=44, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(j, row["observed_signed_circulation_area"] + 0.45, rid, ha="center", color=COLORS[rid], fontsize=6.3)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(["low", "", "", "", "high"])
    ax.set_xlabel("route overload rank")
    ax.set_ylabel("observed signed area")
    ax.set_title("orientation changes with overload rank", loc="left", pad=4)
    finish(ax, axis="y")


def build_figure(obs: pd.DataFrame, route_null: pd.DataFrame, rho_null: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    draw_route_areas(axes[0], obs, route_null)
    panel(axes[0], "a", x=-0.16)
    draw_null_hist(axes[1], rho_null, summary)
    panel(axes[1], "b", x=-0.16)
    draw_route_order(axes[2], obs)
    panel(axes[2], "c", x=-0.16)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig54_phase_circulation_order_null.{ext}", **kwargs)
    plt.close(fig)


def write_report(obs: pd.DataFrame, route_null: pd.DataFrame, summary: pd.DataFrame) -> None:
    s = summary.iloc[0]
    out = ROOT / "nature_physics_phase_circulation_order_null.md"
    lines = [
        "# Phase-circulation cycle-order null",
        "",
        "## Question",
        "",
        "Is the signed-circulation ordering caused by the measured cycle order, or would the same route-level ordering appear after randomly connecting the same points in each route?",
        "",
        "## Main result",
        "",
        f"Randomly permuting the cycle order within each route while preserving the point cloud gave a one-sided cycle-order null probability P = {s.one_sided_p_rho_le_observed:.4f} for obtaining a Spearman ordering as negative as the observed rho = {s.observed_spearman:.3f}. The two-sided absolute-rank probability was {s.two_sided_p_abs_ge_observed:.4f}.",
        "",
        "Single-route signed areas are not all individually outside their route-wise point-order null intervals. The robust statement is therefore a collective route-ordering result: the measured temporal ordering of the trajectories organises the sign and magnitude of circulation across route severity.",
        "",
        "## Observed route areas",
        "",
        obs.round(4).to_markdown(index=False),
        "",
        "## Route-wise point-order null",
        "",
        route_null.round(4).to_markdown(index=False),
        "",
        "## Null summary",
        "",
        summary.round(5).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "Allowed: the circulation-overload ordering depends on measured cycle order and is not reproduced generically by random route-local point order.",
        "",
        "Not allowed: this is not proof that each route has an individually significant geometric area, nor does it define entropy production or a universal geometric phase.",
        "",
        "## Generated files",
        "",
        "- `figures/nphys_fig54_phase_circulation_order_null.*`",
        "- `source_data/nphys_phase_circulation_order_null_rho.csv`",
        "- `source_data/nphys_phase_circulation_order_null_route.csv`",
        "- `source_data/nphys_phase_circulation_order_null_summary.csv`",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    points, obs = load_points()
    rho_null, route_null, summary = run_null(points, obs)
    rho_null.to_csv(SRC / "nphys_phase_circulation_order_null_rho.csv", index=False)
    route_null.to_csv(SRC / "nphys_phase_circulation_order_null_route.csv", index=False)
    summary.to_csv(SRC / "nphys_phase_circulation_order_null_summary.csv", index=False)
    build_figure(obs, route_null, rho_null, summary)
    write_report(obs, route_null, summary)
    print("Wrote phase-circulation cycle-order null")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
