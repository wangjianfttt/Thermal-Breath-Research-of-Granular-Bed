#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, LinearRegression
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
MUTED = "#7F8790"
GRID = "#E7EAEE"
RED = "#B6423E"
BLUE = "#3D6B9C"


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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color="black")


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


def fit_route_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(MAPS)
    cycles = pd.read_csv(CYCLES)
    rows = []
    cycle_parts = []
    for _, mrow in maps.iterrows():
        rid = mrow["regime_id"]
        A = route_matrix(mrow)
        g = cycles[cycles["regime_id"] == rid].dropna(
            subset=["memory_coordinate", "hot_excitation_coordinate", "overload_number"]
        ).copy()
        x = g[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
        y = np.arcsinh(g["overload_number"].to_numpy(float) / 2.0)
        fit = LinearRegression().fit(x, y)
        c = fit.coef_.astype(float)
        c_norm = float(np.linalg.norm(c))
        if c_norm == 0:
            c_hat = np.array([np.nan, np.nan])
            dangerous = np.array([np.nan, np.nan])
        else:
            c_hat = c / c_norm
            dangerous = A.T @ c
            d_norm = np.linalg.norm(dangerous)
            dangerous = dangerous / d_norm if d_norm else dangerous
        one_step_output_gain = float(np.linalg.norm(A.T @ c) / c_norm) if c_norm else np.nan
        direct_hot_fraction = float(abs(c[1]) / (abs(c[0]) + abs(c[1]))) if np.sum(np.abs(c)) else np.nan
        memory_to_hot_coupling = float(A[1, 0])
        hot_to_memory_coupling = float(A[0, 1])
        nonreciprocal_bias = float(memory_to_hot_coupling - hot_to_memory_coupling)
        yhat = fit.predict(x)
        g["overload_response_asinh"] = y
        g["overload_readout_projection"] = x @ c
        g["post_map_output_projection"] = x @ (A.T @ c)
        g["dangerous_input_projection"] = x @ dangerous
        cycle_parts.append(g)
        rows.append(
            {
                "regime_id": rid,
                "n": int(len(g)),
                "output_memory_weight": float(c[0]),
                "output_hot_weight": float(c[1]),
                "output_angle_deg": float(np.degrees(np.arctan2(c[1], c[0]))),
                "direct_output_sensitivity": c_norm,
                "direct_hot_fraction": direct_hot_fraction,
                "one_step_output_gain": one_step_output_gain,
                "dangerous_input_memory": float(dangerous[0]),
                "dangerous_input_hot": float(dangerous[1]),
                "dangerous_input_angle_deg": float(np.degrees(np.arctan2(dangerous[1], dangerous[0]))),
                "memory_to_hot_coupling": memory_to_hot_coupling,
                "hot_to_memory_coupling": hot_to_memory_coupling,
                "nonreciprocal_bias": nonreciprocal_bias,
                "spectral_radius": float(mrow["spectral_radius"]),
                "mean_overload_number": float(mrow["mean_overload_number"]),
                "mean_dimensionless_loop_number": float(mrow["mean_dimensionless_loop_number"]),
                "within_route_r2_output_readout": float(r2_score(y, yhat)),
            }
        )
    route = pd.DataFrame(rows)
    cycle = pd.concat(cycle_parts, ignore_index=True)
    corr = correlation_table(route, cycle)
    tests = transfer_tests(cycle)
    return route, cycle, corr, tests


def correlation_table(route: pd.DataFrame, cycle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    route_pairs = [
        ("one_step_output_gain", "mean_overload_number"),
        ("direct_output_sensitivity", "mean_overload_number"),
        ("direct_hot_fraction", "mean_overload_number"),
        ("nonreciprocal_bias", "one_step_output_gain"),
        ("mean_dimensionless_loop_number", "one_step_output_gain"),
    ]
    for xcol, ycol in route_pairs:
        d = route[[xcol, ycol]].dropna()
        sp = spearmanr(d[xcol], d[ycol])
        rows.append({"scope": "route", "x": xcol, "y": ycol, "n": len(d), "spearman": sp.statistic, "p_value": sp.pvalue})
    cycle_pairs = [
        ("overload_readout_projection", "overload_response_asinh"),
        ("post_map_output_projection", "overload_response_asinh"),
        ("dangerous_input_projection", "overload_response_asinh"),
        ("hot_excitation_coordinate", "overload_response_asinh"),
    ]
    for xcol, ycol in cycle_pairs:
        vals = []
        for rid, g in cycle.groupby("regime_id"):
            d = g[[xcol, ycol]].dropna()
            sp = spearmanr(d[xcol], d[ycol])
            vals.append(abs(float(sp.statistic)))
            rows.append({"scope": f"within_{rid}", "x": xcol, "y": ycol, "n": len(d), "spearman": sp.statistic, "p_value": sp.pvalue})
        rows.append({"scope": "mean_abs_within_route", "x": xcol, "y": ycol, "n": len(vals), "spearman": float(np.mean(vals)), "p_value": np.nan})
    return pd.DataFrame(rows)


def within_center(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_wc"] = out[col] - out.groupby("regime_id")[col].transform("mean")
    return out


def transfer_tests(cycle: pd.DataFrame) -> pd.DataFrame:
    maps = pd.read_csv(MAPS).set_index("regime_id")
    features = {
        "Psi": ["hot_excitation_coordinate"],
        "M_plus_Psi": ["memory_coordinate", "hot_excitation_coordinate"],
    }
    rows = []
    for name, cols in features.items():
        d = cycle[["regime_id", "overload_response_asinh", *cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        d = within_center(d, ["overload_response_asinh", *cols])
        y = d["overload_response_asinh_wc"].to_numpy(float)
        x = d[[f"{col}_wc" for col in cols]].to_numpy(float)
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
                "target": "within_route_asinh_overload",
                "model": name,
                "features": ";".join(cols),
                "validation": "leave_one_route_out_huber_on_within_route_centered_data",
                "n": int(len(d)),
                "r2_vs_mean": float(r2_score(y, pred)),
                "spearman_y_yhat": float(spearmanr(y, pred).statistic),
            }
        )
    d = cycle[["regime_id", "overload_response_asinh", "memory_coordinate", "hot_excitation_coordinate"]].dropna().copy()
    d = within_center(d, ["overload_response_asinh", "memory_coordinate", "hot_excitation_coordinate"])
    y = d["overload_response_asinh_wc"].to_numpy(float)
    x = d[["memory_coordinate_wc", "hot_excitation_coordinate_wc"]].to_numpy(float)
    groups = d["regime_id"].to_numpy()
    direct_pred = np.full(len(d), np.nan)
    post_pred = np.full(len(d), np.nan)
    dangerous_pred = np.full(len(d), np.nan)
    for rid in sorted(set(groups)):
        train = groups != rid
        test = groups == rid
        scaler = StandardScaler().fit(x[train])
        output_model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
        output_model.fit(scaler.transform(x[train]), y[train])
        direct_pred[test] = output_model.predict(scaler.transform(x[test]))

        c_scaled = output_model.coef_.astype(float)
        scale = scaler.scale_
        c = c_scaled / scale
        train_post = np.zeros(train.sum(), dtype=float)
        train_danger = np.zeros(train.sum(), dtype=float)
        for train_rid in sorted(set(groups[train])):
            mask = train & (groups == train_rid)
            A_train = route_matrix(maps.loc[train_rid])
            post_vec = A_train.T @ c
            norm = np.linalg.norm(post_vec)
            danger_vec = post_vec / norm if norm else post_vec
            train_post[groups[train] == train_rid] = x[mask] @ post_vec
            train_danger[groups[train] == train_rid] = x[mask] @ danger_vec
        A_test = route_matrix(maps.loc[rid])
        post_vec_test = A_test.T @ c
        norm_test = np.linalg.norm(post_vec_test)
        danger_vec_test = post_vec_test / norm_test if norm_test else post_vec_test
        test_post = x[test] @ post_vec_test
        test_danger = x[test] @ danger_vec_test

        for train_feature, test_feature, out in [
            (train_post.reshape(-1, 1), test_post.reshape(-1, 1), post_pred),
            (train_danger.reshape(-1, 1), test_danger.reshape(-1, 1), dangerous_pred),
        ]:
            feature_scaler = StandardScaler().fit(train_feature)
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            model.fit(feature_scaler.transform(train_feature), y[train])
            out[test] = model.predict(feature_scaler.transform(test_feature))

    for name, pred in [
        ("global_output_vector", direct_pred),
        ("global_post_map_output", post_pred),
        ("global_dangerous_input", dangerous_pred),
    ]:
        rows.append(
            {
                "target": "within_route_asinh_overload",
                "model": name,
                "features": "trained_on_other_routes_only",
                "validation": "leave_one_route_out_no_overload_fit_on_heldout_route",
                "n": int(len(d)),
                "r2_vs_mean": float(r2_score(y, pred)),
                "spearman_y_yhat": float(spearmanr(y, pred).statistic),
            }
        )
    return pd.DataFrame(rows)


def plot_figure(route: pd.DataFrame, cycle: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.add_patch(plt.Circle((0, 0), 1, color="#C7CDD4", fill=False, lw=0.65, ls=(0, (2, 2))))
    for _, row in route.iterrows():
        rid = row["regime_id"]
        color = COLORS[rid]
        c = np.array([row["output_memory_weight"], row["output_hot_weight"]], dtype=float)
        c = c / np.linalg.norm(c)
        d = np.array([row["dangerous_input_memory"], row["dangerous_input_hot"]], dtype=float)
        ax.arrow(0, 0, c[0], c[1], color=color, lw=1.2, head_width=0.035, length_includes_head=True)
        ax.arrow(0, 0, d[0] * 0.86, d[1] * 0.86, color=color, lw=0.9, head_width=0.03, length_includes_head=True, alpha=0.35, ls="--")
        label_offsets = {
            "R1": (-0.08, 0.04),
            "R3": (0.08, -0.05),
            "R5": (0.08, -0.03),
            "R6": (-0.07, -0.05),
            "R6c": (-0.05, 0.05),
        }
        dx, dy = label_offsets.get(rid, (0.04, 0.04))
        ax.text(c[0] * 0.96 + dx, c[1] * 0.96 + dy, rid, color=color, fontsize=6.4, ha="center", va="center")
    ax.text(0.04, 0.06, "solid: overload readout\nfaint: easiest input after map", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("cold-memory axis")
    ax.set_ylabel(r"hot-excitation axis, $\Psi$")
    ax.set_title("overload observability directions", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    x = np.arange(len(route))
    route = route.sort_values("mean_overload_number").reset_index(drop=True)
    ax.bar(x - 0.18, route["direct_output_sensitivity"], width=0.34, color="#D7DDE4", label="direct sensitivity")
    ax.bar(x + 0.18, route["one_step_output_gain"], width=0.34, color=RED, alpha=0.82, label="post-map gain")
    ax.set_xticks(x, route["regime_id"])
    ax.set_ylabel("response magnitude")
    ax.set_title("output sensitivity is route conditioned", loc="left", pad=4)
    ax.legend(fontsize=5.8, loc="upper left")
    finish(ax, axis="y")

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    order = ["Psi", "M_plus_Psi", "global_output_vector", "global_post_map_output", "global_dangerous_input"]
    sub = tests.set_index("model").loc[order].reset_index()
    colors = [RED if m in {"Psi", "M_plus_Psi"} else MUTED for m in sub["model"]]
    ax.barh(np.arange(len(sub)), sub["r2_vs_mean"], color=colors, height=0.62)
    ax.axvline(0, color="#AEB6C0", lw=0.65)
    ax.set_yticks(np.arange(len(sub)), ["Psi", "M+Psi", "global out", "post-map out", "danger input"])
    ax.invert_yaxis()
    ax.set_xlabel(r"leave-one-route $R^2$")
    ax.set_title("observability aids interpretation, not transfer", loc="left", pad=4)
    finish(ax, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    d = route.copy()
    sp = spearmanr(d["one_step_output_gain"], d["mean_overload_number"])
    for _, row in d.iterrows():
        rid = row["regime_id"]
        ax.scatter(row["one_step_output_gain"], row["mean_overload_number"], s=48, marker=MARKERS[rid], color=COLORS[rid], edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["one_step_output_gain"] + 0.015, row["mean_overload_number"], rid, color=COLORS[rid], fontsize=6.2, va="center")
    ax.text(0.04, 0.94, rf"route Spearman $\rho={sp.statistic:.2f}$" + f"\nP={sp.pvalue:.2f}", transform=ax.transAxes, va="top", ha="left", color=MUTED, fontsize=6.1)
    ax.set_xlabel("post-map output gain")
    ax.set_ylabel(r"mean overload $\langle\widehat{\Omega}\rangle$")
    ax.set_title("gain is a modifier, not the overload law", loc="left", pad=4)
    finish(ax)

    out = FIG / "nphys_fig38_overload_observability"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(route: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame) -> None:
    gain_row = corr[(corr["scope"] == "route") & (corr["x"] == "one_step_output_gain") & (corr["y"] == "mean_overload_number")].iloc[0]
    psi_r2 = tests.query("model == 'Psi'")["r2_vs_mean"].iloc[0]
    post_r2 = tests.query("model == 'global_post_map_output'")["r2_vs_mean"].iloc[0]
    direct_r2 = tests.query("model == 'global_output_vector'")["r2_vs_mean"].iloc[0]
    text = f"""# Overload observability response audit

Date: 2026-06-12

## Question

The return map is a state equation, but hot overload is an output. This audit asks whether overload is observable from the reduced cold-memory/hot-excitation plane and which state-space direction would most efficiently produce an overload readout after one fitted return-map step.

For each route, overload was treated as an output

\\[
z_n=\\operatorname{{asinh}}(\\widehat{{\\Omega}}_n/2)\\simeq \\mathbf{{c}}_\\pi^\\mathsf{{T}}\\mathbf{{y}}_n+d_\\pi,
\\]

with \\(\\mathbf{{y}}_n=(M_n,\\Psi_n)\\). The direct output vector \\(\\mathbf{{c}}_\\pi\\) measures which reduced coordinate is read as overload. The post-map output vector \\(\\mathbf{{A}}_\\pi^\\mathsf{{T}}\\mathbf{{c}}_\\pi\\) measures the input direction that most efficiently becomes observable as overload after one route-conditioned return step.

## Main result

The fitted overload readout is strongly route conditioned rather than universal. The one-step output gain correlates only descriptively with route mean overload across five routes (Spearman rho = {gain_row.spearman:.3f}, P = {gain_row.p_value:.3f}). It should therefore be treated as an observability modifier, not as the overload law.

Leave-one-route transfer confirms the boundary when the overload-output vector is trained only on the other routes: the dimensionless loop coordinate Psi gives R2 = {psi_r2:.3f}, the global direct output vector gives R2 = {direct_r2:.3f}, and the post-map output projection gives R2 = {post_r2:.3f}. The useful physical interpretation is that return-map observability explains how a stable trained state can be read out as overload in a route-local way; it does not replace the force-loop coordinate.

## Interpretation allowed in the manuscript

Allowed: hot overload is an output projection of the reduced return map, and the route-conditioned map changes which perturbations are most observable as overload.

Not allowed: route-fitted output vectors are not universal constitutive overload laws. The fair transfer audit must train output directions only on the other routes.

## Route observability table

{route.round(4).to_markdown(index=False)}

## Correlation audit

{corr.round(4).to_markdown(index=False)}

## Transfer audit

{tests.round(4).to_markdown(index=False)}

## Generated files

- `figures/nphys_fig38_overload_observability.*`
- `source_data/nphys_overload_observability_route_metrics.csv`
- `source_data/nphys_overload_observability_cycle_projection.csv`
- `source_data/nphys_overload_observability_correlations.csv`
- `source_data/nphys_overload_observability_transfer_tests.csv`
"""
    (ROOT / "nature_physics_overload_observability_response.md").write_text(text, encoding="utf-8")


def main() -> None:
    route, cycle, corr, tests = fit_route_outputs()
    route.to_csv(SRC / "nphys_overload_observability_route_metrics.csv", index=False)
    cycle[
        [
            "regime_id",
            "cycle",
            "memory_coordinate",
            "hot_excitation_coordinate",
            "overload_number",
            "overload_response_asinh",
            "overload_readout_projection",
            "post_map_output_projection",
            "dangerous_input_projection",
        ]
    ].to_csv(SRC / "nphys_overload_observability_cycle_projection.csv", index=False)
    corr.to_csv(SRC / "nphys_overload_observability_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_overload_observability_transfer_tests.csv", index=False)
    plot_figure(route, cycle, corr, tests)
    write_report(route, corr, tests)
    print("Overload observability response audit complete.")
    print(route[["regime_id", "direct_hot_fraction", "one_step_output_gain", "mean_overload_number"]].round(3).to_string(index=False))
    print(tests[["model", "r2_vs_mean", "spearman_y_yhat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
