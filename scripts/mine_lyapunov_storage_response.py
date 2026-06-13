#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

MAPS = SRC / "nphys_return_map_phase_portrait_route_maps.csv"
CYCLES = SRC / "nphys_return_map_phase_portrait_cycle_metrics.csv"
OBS = SRC / "nphys_overload_observability_route_metrics.csv"

COLORS = {"R1": "#3D6B9C", "R3": "#D98C3A", "R5": "#7E6AAE", "R6": "#C95F3F", "R6c": "#8C2F2C"}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v"}
INK = "#252A31"
MUTED = "#7F8790"
GRID = "#E7EAEE"
RED = "#B6423E"


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


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(MAPS)
    cycles = pd.read_csv(CYCLES)
    obs = pd.read_csv(OBS).set_index("regime_id") if OBS.exists() else pd.DataFrame()
    route_rows = []
    cycle_parts = []
    for _, row in maps.iterrows():
        rid = row["regime_id"]
        A = route_matrix(row)
        P = solve_discrete_lyapunov(A.T, np.eye(2))
        P = 0.5 * (P + P.T)
        eig = np.linalg.eigvalsh(P)
        residual = A.T @ P @ A - P + np.eye(2)
        anisotropy = float(eig.max() / eig.min())
        dissipativity_margin = float(1.0 / eig.max())
        storage_axes = np.linalg.eigh(P)[1]
        if rid in obs.index:
            c = obs.loc[rid, ["output_memory_weight", "output_hot_weight"]].to_numpy(float)
            c_norm = np.linalg.norm(c)
            c_hat = c / c_norm if c_norm else np.array([np.nan, np.nan])
            output_storage = float(c_hat @ P @ c_hat)
            hot_axis_storage = float(np.array([0.0, 1.0]) @ P @ np.array([0.0, 1.0]))
        else:
            output_storage = np.nan
            hot_axis_storage = np.nan
        route_rows.append(
            {
                "regime_id": rid,
                "P11": float(P[0, 0]),
                "P12": float(P[0, 1]),
                "P22": float(P[1, 1]),
                "storage_eigen_min": float(eig.min()),
                "storage_eigen_max": float(eig.max()),
                "storage_anisotropy": anisotropy,
                "dissipativity_margin": dissipativity_margin,
                "lyapunov_residual_fro": float(np.linalg.norm(residual)),
                "principal_storage_axis_memory": float(storage_axes[0, -1]),
                "principal_storage_axis_hot": float(storage_axes[1, -1]),
                "output_direction_storage": output_storage,
                "hot_axis_storage": hot_axis_storage,
                "spectral_radius": float(row["spectral_radius"]),
                "mean_overload_number": float(row["mean_overload_number"]),
                "mean_dimensionless_loop_number": float(row["mean_dimensionless_loop_number"]),
            }
        )
        g = cycles[cycles["regime_id"] == rid].copy()
        y = g[["memory_coordinate", "hot_excitation_coordinate"]].to_numpy(float)
        ynext = g[["next_memory_coordinate", "next_hot_excitation_coordinate"]].to_numpy(float)
        v = np.einsum("ij,jk,ik->i", y, P, y)
        v_map_next = np.einsum("ij,jk,ik->i", y @ A.T, P, y @ A.T)
        v_obs_next = np.einsum("ij,jk,ik->i", ynext, P, ynext)
        euclid = np.sum(y * y, axis=1)
        g["lyapunov_storage"] = v
        g["map_predicted_storage_next"] = v_map_next
        g["observed_next_storage"] = v_obs_next
        g["storage_drop_map"] = v - v_map_next
        g["storage_drop_observed"] = v - v_obs_next
        g["euclidean_state_energy"] = euclid
        g["storage_anisotropy"] = anisotropy
        g["dissipativity_margin"] = dissipativity_margin
        g["overload_response_asinh"] = np.arcsinh(g["overload_number"].to_numpy(float) / 2.0)
        cycle_parts.append(g)
    route = pd.DataFrame(route_rows)
    cycle = pd.concat(cycle_parts, ignore_index=True)
    corr = correlation_table(route, cycle)
    tests = transfer_tests(cycle)
    return route, cycle, corr, tests


def correlation_table(route: pd.DataFrame, cycle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    route_pairs = [
        ("storage_anisotropy", "mean_overload_number"),
        ("dissipativity_margin", "mean_overload_number"),
        ("output_direction_storage", "mean_overload_number"),
        ("hot_axis_storage", "mean_overload_number"),
        ("storage_anisotropy", "spectral_radius"),
    ]
    for xcol, ycol in route_pairs:
        d = route[[xcol, ycol]].dropna()
        sp = spearmanr(d[xcol], d[ycol])
        rows.append({"scope": "route", "x": xcol, "y": ycol, "n": len(d), "spearman": sp.statistic, "p_value": sp.pvalue})
    cycle_pairs = [
        ("lyapunov_storage", "overload_response_asinh"),
        ("storage_drop_map", "overload_response_asinh"),
        ("storage_drop_observed", "overload_response_asinh"),
        ("euclidean_state_energy", "overload_response_asinh"),
        ("hot_excitation_coordinate", "overload_response_asinh"),
    ]
    for xcol, ycol in cycle_pairs:
        vals = []
        for rid, g in cycle.groupby("regime_id", sort=True):
            d = g[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
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
    features = {
        "Psi": ["hot_excitation_coordinate"],
        "lyapunov_storage": ["lyapunov_storage"],
        "storage_drop_map": ["storage_drop_map"],
        "storage_drop_observed": ["storage_drop_observed"],
        "storage_plus_Psi": ["lyapunov_storage", "hot_excitation_coordinate"],
        "storage_drop_plus_Psi": ["storage_drop_map", "hot_excitation_coordinate"],
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
    return pd.DataFrame(rows)


def plot_figure(route: pd.DataFrame, cycle: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.96])

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    for _, row in route.iterrows():
        rid = row["regime_id"]
        P = np.array([[row["P11"], row["P12"]], [row["P12"], row["P22"]]], dtype=float)
        vals, vecs = np.linalg.eigh(P)
        ell = vecs @ np.diag(1 / np.sqrt(vals)) @ circle
        ax.plot(ell[0], ell[1], color=COLORS[rid], lw=1.0, alpha=0.9)
        v = np.array([row["principal_storage_axis_memory"], row["principal_storage_axis_hot"]])
        ax.plot([0, v[0] * 0.8], [0, v[1] * 0.8], color=COLORS[rid], lw=0.8, alpha=0.55)
        ax.text(ell[0, 18], ell[1, 18], rid, color=COLORS[rid], fontsize=6.2)
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_aspect("equal")
    ax.set_xlabel("cold-memory axis")
    ax.set_ylabel(r"hot-excitation axis, $\Psi$")
    ax.set_title(r"route-local storage ellipses, $y^TPy=1$", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for _, row in route.sort_values("mean_overload_number").iterrows():
        rid = row["regime_id"]
        ax.scatter(row["storage_anisotropy"], row["mean_overload_number"], s=48, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.5, zorder=3)
        ax.text(row["storage_anisotropy"] * 1.02, row["mean_overload_number"], rid, color=COLORS[rid], fontsize=6.2, va="center")
    sp = spearmanr(route["storage_anisotropy"], route["mean_overload_number"])
    ax.text(0.04, 0.94, rf"route $\rho={sp.statistic:.2f}$" + f"\nP={sp.pvalue:.2f}", transform=ax.transAxes, va="top", ha="left", color=MUTED, fontsize=6.1)
    ax.set_xscale("log")
    ax.set_xlabel("storage anisotropy")
    ax.set_ylabel(r"mean overload $\langle\widehat{\Omega}\rangle$")
    ax.set_title("storage anisotropy is not the overload law", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c")
    d = cycle.dropna(subset=["lyapunov_storage", "overload_response_asinh"])
    for rid, g in d.groupby("regime_id", sort=True):
        ax.scatter(g["lyapunov_storage"], g["overload_response_asinh"], s=13, color=COLORS[rid], marker=MARKERS[rid], edgecolor="white", lw=0.25, alpha=0.72)
    mean_abs = corr[(corr["scope"] == "mean_abs_within_route") & (corr["x"] == "lyapunov_storage")]["spearman"].iloc[0]
    ax.text(0.97, 0.06, rf"mean route $|\rho|={mean_abs:.2f}$", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=6.1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(r"storage $V=y^TPy$")
    ax.set_ylabel(r"overload response, asinh$(\widehat{\Omega}/2)$")
    ax.set_title("high storage is only a weak output proxy", loc="left", pad=4)
    finish(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    order = ["Psi", "lyapunov_storage", "storage_drop_map", "storage_plus_Psi", "storage_drop_plus_Psi"]
    sub = tests.set_index("model").loc[order].reset_index()
    colors = [RED if m == "Psi" else MUTED for m in sub["model"]]
    ax.barh(np.arange(len(sub)), sub["r2_vs_mean"], color=colors, height=0.62)
    ax.axvline(0, color="#AEB6C0", lw=0.65)
    ax.set_yticks(np.arange(len(sub)), ["Psi", "storage", "map drop", "storage+Psi", "drop+Psi"])
    ax.invert_yaxis()
    ax.set_xlabel(r"leave-one-route $R^2$")
    ax.set_title("storage supports dissipation, not prediction", loc="left", pad=4)
    finish(ax, axis="x")

    out = FIG / "nphys_fig39_lyapunov_storage"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(route: pd.DataFrame, corr: pd.DataFrame, tests: pd.DataFrame) -> None:
    anis = corr[(corr["scope"] == "route") & (corr["x"] == "storage_anisotropy") & (corr["y"] == "mean_overload_number")].iloc[0]
    storage_rho = corr[(corr["scope"] == "mean_abs_within_route") & (corr["x"] == "lyapunov_storage")]["spearman"].iloc[0]
    psi_r2 = tests.query("model == 'Psi'")["r2_vs_mean"].iloc[0]
    storage_r2 = tests.query("model == 'lyapunov_storage'")["r2_vs_mean"].iloc[0]
    drop_r2 = tests.query("model == 'storage_drop_map'")["r2_vs_mean"].iloc[0]
    text = f"""# Lyapunov storage-response audit

Date: 2026-06-12

## Question

The reduced return maps have spectral radii below one. This audit asks whether that stability can be written as a route-local storage function and whether the storage function explains overload.

For each route map \\(\\mathbf{{y}}_{{n+1}}=\\mathbf{{A}}_\\pi\\mathbf{{y}}_n\\), we solved the discrete Lyapunov equation

\\[
\\mathbf{{A}}_\\pi^\\mathsf{{T}}\\mathbf{{P}}_\\pi\\mathbf{{A}}_\\pi-\\mathbf{{P}}_\\pi=-\\mathbf{{I}},
\\]

and used \\(V_n=\\mathbf{{y}}_n^\\mathsf{{T}}\\mathbf{{P}}_\\pi\\mathbf{{y}}_n\\) as a route-conditioned storage coordinate.

## Main result

Every fitted route map admits a positive route-local storage matrix because all point-estimate spectral radii are below one. This supports the dissipative-map language: there is a quadratic metric in which the linearized cycle map is contractive.

The storage coordinate is not the overload law. Route-level storage anisotropy correlates only weakly/descriptively with mean overload across five routes (Spearman rho = {anis.spearman:.3f}, P = {anis.p_value:.3f}). Within routes, storage has mean absolute rank association {storage_rho:.3f} with overload response, but leave-one-route transfer is weaker for storage alone (R2 = {storage_r2:.3f}) and map-predicted storage drop (R2 = {drop_r2:.3f}) than for the dimensionless loop coordinate Psi (R2 = {psi_r2:.3f}).

## Interpretation allowed in the manuscript

Allowed: the trained bed can be described as dissipative in a route-local storage metric while still showing output overload in selected directions. This is the mathematical form of "stable but excitable" breathing.

Not allowed: Lyapunov storage, storage anisotropy or storage drop should not replace force-loop activation or Psi as the overload mechanism.

## Route storage table

{route.round(4).to_markdown(index=False)}

## Correlation audit

{corr.round(4).to_markdown(index=False)}

## Transfer audit

{tests.round(4).to_markdown(index=False)}

## Generated files

- `figures/nphys_fig39_lyapunov_storage.*`
- `source_data/nphys_lyapunov_storage_route_metrics.csv`
- `source_data/nphys_lyapunov_storage_cycle_metrics.csv`
- `source_data/nphys_lyapunov_storage_correlations.csv`
- `source_data/nphys_lyapunov_storage_transfer_tests.csv`
"""
    (ROOT / "nature_physics_lyapunov_storage_response.md").write_text(text, encoding="utf-8")


def main() -> None:
    route, cycle, corr, tests = build_tables()
    route.to_csv(SRC / "nphys_lyapunov_storage_route_metrics.csv", index=False)
    cycle[
        [
            "regime_id",
            "cycle",
            "memory_coordinate",
            "hot_excitation_coordinate",
            "overload_number",
            "overload_response_asinh",
            "lyapunov_storage",
            "map_predicted_storage_next",
            "observed_next_storage",
            "storage_drop_map",
            "storage_drop_observed",
            "euclidean_state_energy",
            "storage_anisotropy",
            "dissipativity_margin",
        ]
    ].to_csv(SRC / "nphys_lyapunov_storage_cycle_metrics.csv", index=False)
    corr.to_csv(SRC / "nphys_lyapunov_storage_correlations.csv", index=False)
    tests.to_csv(SRC / "nphys_lyapunov_storage_transfer_tests.csv", index=False)
    plot_figure(route, cycle, corr, tests)
    write_report(route, corr, tests)
    print("Lyapunov storage-response audit complete.")
    print(route[["regime_id", "storage_anisotropy", "dissipativity_margin", "mean_overload_number"]].round(3).to_string(index=False))
    print(tests[["model", "r2_vs_mean", "spearman_y_yhat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
