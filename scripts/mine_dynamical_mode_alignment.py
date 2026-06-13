#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

MAPS = SRC / "nphys_return_map_phase_portrait_route_maps.csv"
CYCLES = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"

COLORS = {"R1": "#3D6B9C", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8C2F2C"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
GRID = "#E7EAEE"
MUTED = "#8B929A"
RED = "#B6423E"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 7.5,
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


def route_matrix(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [row["A11_memory_to_memory"], row["A12_hot_to_memory"]],
            [row["A21_memory_to_hot"], row["A22_hot_to_hot"]],
        ],
        dtype=float,
    )


def build_modal_tables(n_null: int = 5000, seed: int = 77) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(MAPS)
    cycles = pd.read_csv(CYCLES)
    route_rows = []
    cycle_parts = []
    corr_rows = []
    for _, row in maps.iterrows():
        rid = row["regime_id"]
        A = route_matrix(row)
        u, s, vt = np.linalg.svd(A)
        v1 = vt[0]
        u1 = u[:, 0]
        amplified = A @ v1
        route_rows.append(
            {
                "regime_id": rid,
                "spectral_radius": float(row["spectral_radius"]),
                "one_step_gain": float(s[0]),
                "secondary_gain": float(s[1]),
                "right_mode_memory": float(v1[0]),
                "right_mode_hot": float(v1[1]),
                "left_mode_memory": float(u1[0]),
                "left_mode_hot": float(u1[1]),
                "amplified_memory_weight": float(amplified[0]),
                "amplified_hot_weight": float(amplified[1]),
                "mode_angle_deg": float(np.degrees(np.arctan2(v1[1], v1[0]))),
                "mean_overload_number": float(row["mean_overload_number"]),
                "mean_dimensionless_loop_number": float(row["mean_dimensionless_loop_number"]),
            }
        )
        g = cycles[cycles["regime_id"] == rid].copy()
        x = g[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
        projection = x @ v1
        g["amplifying_projection"] = projection
        g["amplified_memory_component"] = projection * amplified[0]
        g["amplified_hot_component"] = projection * amplified[1]
        g["one_step_gain"] = float(s[0])
        g["mode_angle_deg"] = float(np.degrees(np.arctan2(v1[1], v1[0])))
        g["overload_asinh"] = np.arcsinh(g["overload_number"] / 2.0)
        cycle_parts.append(g)
        for feature in ["amplifying_projection", "amplified_memory_component", "amplified_hot_component", "hot_excitation_coordinate"]:
            d = g[[feature, "overload_number"]].replace([np.inf, -np.inf], np.nan).dropna()
            sp = spearmanr(d[feature], d["overload_number"])
            corr_rows.append(
                {
                    "regime_id": rid,
                    "feature": feature,
                    "n": int(len(d)),
                    "spearman": float(sp.statistic),
                    "p_value": float(sp.pvalue),
                }
            )
    route_modes = pd.DataFrame(route_rows)
    cycle_modal = pd.concat(cycle_parts, ignore_index=True)
    correlations = pd.DataFrame(corr_rows)
    tests = build_model_tests(cycle_modal)
    random_null = random_direction_null(cycle_modal, n_null=n_null, seed=seed)
    return route_modes, cycle_modal, correlations, tests, random_null


def within_centered_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for col in cols:
        d[f"{col}_wc"] = d[col] - d.groupby("regime_id")[col].transform("mean")
    return d


def build_model_tests(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "memory coordinate": ["memory_coordinate"],
        "hot excitation Psi": ["hot_excitation_coordinate"],
        "memory + Psi": ["memory_coordinate", "hot_excitation_coordinate"],
        "amplifying projection": ["amplifying_projection"],
        "amplified hot component": ["amplified_hot_component"],
        "amplified memory component": ["amplified_memory_component"],
        "Psi + amplified hot": ["hot_excitation_coordinate", "amplified_hot_component"],
        "memory + Psi + amplified": ["memory_coordinate", "hot_excitation_coordinate", "amplified_hot_component"],
    }
    rows = []
    for name, features in feature_sets.items():
        cols = ["regime_id", "overload_number", *features]
        d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        d = within_centered_frame(d, ["overload_number", *features])
        y = d["overload_number_wc"].to_numpy(float)
        x = d[[f"{feature}_wc" for feature in features]].to_numpy(float)
        groups = d["regime_id"].to_numpy()
        pred = np.full(len(d), np.nan)
        for rid in sorted(set(groups)):
            train = groups != rid
            test = groups == rid
            scaler = StandardScaler().fit(x[train])
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            model.fit(scaler.transform(x[train]), y[train])
            pred[test] = model.predict(scaler.transform(x[test]))
        rows.append(
            {
                "target": "within_route_overload_number",
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out_huber_on_within_route_centered_data",
                "n": int(len(d)),
                "r2_vs_mean": float(r2_score(y, pred)),
                "spearman_y_yhat": float(spearmanr(y, pred).statistic),
            }
        )
    return pd.DataFrame(rows)


def random_direction_null(df: pd.DataFrame, n_null: int = 5000, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = []
    for _, group in df.groupby("regime_id", sort=True):
        d = group[["amplified_hot_component", "overload_number"]].dropna()
        observed.append(abs(float(spearmanr(d["amplified_hot_component"], d["overload_number"]).statistic)))
    observed_mean = float(np.mean(observed))
    rows = [{"sample": -1, "mean_abs_route_spearman": observed_mean, "kind": "observed_modal_direction"}]
    for i in range(n_null):
        vals = []
        for _, group in df.groupby("regime_id", sort=True):
            theta = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])
            x = group[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
            proj = x @ direction
            vals.append(abs(float(spearmanr(proj, group["overload_number"]).statistic)))
        rows.append({"sample": i, "mean_abs_route_spearman": float(np.mean(vals)), "kind": "random_direction"})
    return pd.DataFrame(rows)


def plot_figure(route_modes: pd.DataFrame, cycle_modal: pd.DataFrame, correlations: pd.DataFrame, tests: pd.DataFrame, random_null: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.98])

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    circle = plt.Circle((0, 0), 1, color="#C7CDD4", fill=False, lw=0.65, ls=(0, (2, 2)))
    ax.add_patch(circle)
    for _, row in route_modes.iterrows():
        rid = row["regime_id"]
        v = np.array([row["right_mode_memory"], row["right_mode_hot"]])
        amp = np.array([row["amplified_memory_weight"], row["amplified_hot_weight"]])
        color = COLORS[rid]
        ax.arrow(0, 0, v[0], v[1], color=color, lw=1.0, head_width=0.035, length_includes_head=True, alpha=0.85)
        ax.arrow(v[0], v[1], amp[0] - v[0], amp[1] - v[1], color=color, lw=0.8, head_width=0.03, length_includes_head=True, alpha=0.45, ls="--")
        ax.text(v[0] * 1.08, v[1] * 1.08, rid, color=color, fontsize=6.4, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("cold memory axis")
    ax.set_ylabel(r"hot excitation axis, $\Psi$")
    ax.set_title("max-gain directions of the return map", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    d = cycle_modal.dropna(subset=["amplified_memory_component", "overload_asinh"])
    for rid, group in d.groupby("regime_id", sort=True):
        ax.scatter(
            group["amplified_memory_component"],
            group["overload_asinh"],
            s=18,
            color=COLORS[rid],
            marker=MARKERS[rid],
            edgecolor="white",
            lw=0.3,
            alpha=0.78,
            label=rid,
        )
    route_abs = correlations[correlations["feature"] == "amplified_memory_component"]["spearman"].abs().mean()
    ax.text(0.97, 0.06, rf"mean $|\rho|={route_abs:.2f}$", transform=ax.transAxes, va="bottom", ha="right")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel("amplified memory component")
    ax.set_ylabel(r"overload response, asinh$(\widehat{\Omega}/2)$")
    ax.set_title("modal projection aligns within routes", loc="left", pad=4)
    ax.legend(ncol=3, fontsize=5.8, loc="upper left", handletextpad=0.25, columnspacing=0.55)
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    order = [
        "hot excitation Psi",
        "memory + Psi",
        "amplified memory component",
        "amplified hot component",
        "amplifying projection",
        "memory coordinate",
    ]
    sub = tests.set_index("model").loc[order].reset_index()
    colors = [RED if m in {"hot excitation Psi", "memory + Psi"} else MUTED for m in sub["model"]]
    ax.barh(np.arange(len(sub)), sub["r2_vs_mean"], color=colors, height=0.62)
    ax.axvline(0, color="#AEB6C0", lw=0.65)
    ax.set_yticks(np.arange(len(sub)), ["Psi", "M+Psi", "ampl. M", "ampl. hot", "projection", "M"])
    ax.invert_yaxis()
    ax.set_xlabel(r"leave-one-route $R^2$")
    ax.set_title("modal variables do not replace Psi", loc="left", pad=4)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    null = random_null[random_null["kind"] == "random_direction"]["mean_abs_route_spearman"].to_numpy(float)
    obs = float(random_null[random_null["kind"] == "observed_modal_direction"]["mean_abs_route_spearman"].iloc[0])
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    ax.hist(null, bins=38, color="#D7DDE4", edgecolor="white", lw=0.3)
    ax.axvline(obs, color=RED, lw=1.25)
    ax.text(0.97, 0.92, rf"$P={p:.3f}$" + "\n" + rf"observed={obs:.2f}", transform=ax.transAxes, va="top", ha="right")
    ax.set_xlabel(r"mean route-wise $|\rho|$ for random directions")
    ax.set_ylabel("count")
    ax.set_title("directional alignment is suggestive, not decisive", loc="left", pad=4)
    finish(ax, axis="y")

    out = FIG / "nphys_fig32_dynamical_mode_alignment"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(route_modes: pd.DataFrame, correlations: pd.DataFrame, tests: pd.DataFrame, random_null: pd.DataFrame) -> None:
    null = random_null[random_null["kind"] == "random_direction"]["mean_abs_route_spearman"].to_numpy(float)
    obs = float(random_null[random_null["kind"] == "observed_modal_direction"]["mean_abs_route_spearman"].iloc[0])
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    psi_r2 = tests.query("model == 'hot excitation Psi'")["r2_vs_mean"].iloc[0]
    modal_r2 = tests.query("model == 'amplified memory component'")["r2_vs_mean"].iloc[0]
    mean_abs_rho = correlations[correlations["feature"] == "amplified_memory_component"]["spearman"].abs().mean()
    report = f"""# Dynamical mode-alignment audit

Date: 2026-06-12

## Question

The return-map audit showed that route maps are dissipative but can be non-normal. This audit asks whether the maximum-gain singular direction of each fitted map aligns with the observed hot-overload coordinate. The singular directions are fitted from the cold-memory/hot-excitation return map only; overload is not used to fit them.

## Main result

The maximum-gain modal projection aligns with overload within routes: the mean absolute route-wise Spearman correlation between the amplified memory component and overload is {mean_abs_rho:.3f}. A random-direction null gives P = {p:.3f}, so this alignment is suggestive but not decisive at the conventional 0.05 level.

In leave-one-route transfer, the hot excitation coordinate Psi remains the stronger and simpler overload coordinate (R2 = {psi_r2:.3f}) than the modal component alone (R2 = {modal_r2:.3f}). The modal audit therefore supports non-normal dynamics as a route-local modifier of overload, not as a replacement for the dimensionless loop number.

## Interpretation allowed in the manuscript

Allowed: stable return maps contain gain directions that align with dangerous within-route excursions, supporting the language of transient amplification.

Not allowed: the modal projection is not a universal predictive coordinate and does not outperform Psi in leave-one-route transfer.

## Route modes

{route_modes.round(4).to_markdown(index=False)}

## Model tests

{tests.round(4).to_markdown(index=False)}

## Generated files

- `figures/nphys_fig32_dynamical_mode_alignment.*`
- `source_data/nphys_dynamical_mode_alignment_route_modes.csv`
- `source_data/nphys_dynamical_mode_alignment_cycle_projection.csv`
- `source_data/nphys_dynamical_mode_alignment_correlations.csv`
- `source_data/nphys_dynamical_mode_alignment_model_tests.csv`
- `source_data/nphys_dynamical_mode_alignment_random_direction_null.csv`
"""
    (ROOT / "nature_physics_dynamical_mode_alignment.md").write_text(report, encoding="utf-8")


def main() -> None:
    route_modes, cycle_modal, correlations, tests, random_null = build_modal_tables()
    route_modes.to_csv(SRC / "nphys_dynamical_mode_alignment_route_modes.csv", index=False)
    cycle_modal[
        [
            "regime_id",
            "cycle",
            "memory_coordinate",
            "hot_excitation_coordinate",
            "overload_number",
            "overload_asinh",
            "dimensionless_loop_number",
            "amplifying_projection",
            "amplified_memory_component",
            "amplified_hot_component",
            "one_step_gain",
            "mode_angle_deg",
        ]
    ].to_csv(SRC / "nphys_dynamical_mode_alignment_cycle_projection.csv", index=False)
    correlations.to_csv(SRC / "nphys_dynamical_mode_alignment_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_dynamical_mode_alignment_model_tests.csv", index=False)
    random_null.to_csv(SRC / "nphys_dynamical_mode_alignment_random_direction_null.csv", index=False)
    plot_figure(route_modes, cycle_modal, correlations, tests, random_null)
    write_report(route_modes, correlations, tests, random_null)
    print("Dynamical mode-alignment audit complete.")
    print(tests[["model", "r2_vs_mean", "spearman_y_yhat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
