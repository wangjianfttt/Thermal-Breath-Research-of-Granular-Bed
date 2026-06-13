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
from sklearn.metrics import r2_score

from mine_force_loop_spatial_fingerprint import (
    FIG,
    GRID,
    INK,
    MARKERS,
    MUTED,
    ROUTE_COLORS,
    SRC,
    StateKey,
    discover_keys,
    read_state,
    route_center,
)
from mine_true_force_percolation import UnionFind


ROOT = Path(__file__).resolve().parent
THRESHOLDS = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00], dtype=float)
OVERLOAD_SCALE = 0.003
RED = "#B6423E"
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
GREEN = "#4F8B67"
VIOLET = "#7E6AAE"
GREY = "#7A838E"


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


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float, int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 4 or np.nanstd(xx[ok]) == 0 or np.nanstd(yy[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    out = spearmanr(xx[ok], yy[ok])
    return float(out.statistic), float(out.pvalue), int(ok.sum())


def state_filtration(key: StateKey) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    state = read_state(key)
    contacts = state.contacts.copy()
    atoms = state.atoms.copy()
    valid = contacts["id1"].isin(atoms.index) & contacts["id2"].isin(atoms.index) & (contacts["force"] > 0)
    d = contacts.loc[valid].copy()
    base: dict[str, float | str | int] = {
        "run": key.run,
        "tag": key.tag,
        "regime_id": key.regime_id,
        "cycle": key.cycle,
        "phase": key.phase,
        "n_contacts": int(len(d)),
    }
    curve_rows: list[dict[str, float | str | int]] = []
    if d.empty:
        return base, curve_rows

    ids = np.array(sorted(set(d["id1"]).union(set(d["id2"]))))
    id_to_idx = {int(v): i for i, v in enumerate(ids)}
    n_nodes = len(ids)
    bottom = np.zeros(n_nodes, dtype=bool)
    top = np.zeros(n_nodes, dtype=bool)
    side = np.zeros(n_nodes, dtype=bool)
    uf = UnionFind(n_nodes, bottom, top, side)

    forces = d["force"].to_numpy(float)
    total_force = float(forces.sum())
    order = np.argsort(forces)[::-1]
    n_edges = len(order)
    threshold_ranks = np.maximum(1, np.ceil(THRESHOLDS * n_edges).astype(int))
    threshold_lookup = {int(rank): float(th) for rank, th in zip(threshold_ranks, THRESHOLDS)}

    cycle_births = 0
    cycle_force = 0.0
    added_force = 0.0
    first_cycle_edge_fraction = np.nan
    first_cycle_force_fraction = np.nan
    curve_beta: list[float] = []
    curve_loop_force: list[float] = []
    curve_giant: list[float] = []
    curve_added_force: list[float] = []

    for rank, row_idx in enumerate(order, start=1):
        row = d.iloc[int(row_idx)]
        f = float(row["force"])
        added_force += f
        a = id_to_idx[int(row["id1"])]
        b = id_to_idx[int(row["id2"])]
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            cycle_births += 1
            cycle_force += f
            if np.isnan(first_cycle_edge_fraction):
                first_cycle_edge_fraction = rank / n_edges
                first_cycle_force_fraction = added_force / total_force if total_force > 0 else np.nan
        else:
            uf.union(a, b)

        if rank in threshold_lookup:
            th = threshold_lookup[rank]
            beta_density = cycle_births / max(rank, 1)
            loop_force_share_partial = cycle_force / total_force if total_force > 0 else np.nan
            curve_beta.append(beta_density)
            curve_loop_force.append(loop_force_share_partial)
            curve_giant.append(uf.max_size / n_nodes)
            curve_added_force.append(added_force / total_force if total_force > 0 else np.nan)
            curve_rows.append(
                {
                    **base,
                    "edge_fraction": th,
                    "rank": int(rank),
                    "beta1_birth_density": float(beta_density),
                    "loop_force_share_partial": float(loop_force_share_partial),
                    "giant_component_fraction": float(uf.max_size / n_nodes),
                    "added_force_fraction": float(added_force / total_force) if total_force > 0 else np.nan,
                }
            )

    x = THRESHOLDS[: len(curve_beta)]
    beta_area = float(np.trapz(np.asarray(curve_beta, dtype=float), x)) if len(curve_beta) > 1 else np.nan
    loop_force_area = float(np.trapz(np.asarray(curve_loop_force, dtype=float), x)) if len(curve_loop_force) > 1 else np.nan
    giant_area = float(np.trapz(np.asarray(curve_giant, dtype=float), x)) if len(curve_giant) > 1 else np.nan
    base.update(
        {
            "first_cycle_edge_fraction": float(first_cycle_edge_fraction),
            "first_cycle_force_fraction": float(first_cycle_force_fraction),
            "beta1_birth_density_5pct": float(curve_beta[2]) if len(curve_beta) > 2 else np.nan,
            "beta1_birth_density_10pct": float(curve_beta[3]) if len(curve_beta) > 3 else np.nan,
            "loop_force_share_5pct": float(curve_loop_force[2]) if len(curve_loop_force) > 2 else np.nan,
            "loop_force_share_10pct": float(curve_loop_force[3]) if len(curve_loop_force) > 3 else np.nan,
            "beta1_birth_area": beta_area,
            "loop_force_persistence_area": loop_force_area,
            "giant_component_area": giant_area,
            "total_beta1_birth_density": float(cycle_births / n_edges),
            "total_loop_force_share": float(cycle_force / total_force) if total_force > 0 else np.nan,
            "top5_force_share": float(forces[order[: max(1, int(np.ceil(0.05 * n_edges)))]].sum() / total_force),
        }
    )
    return base, curve_rows


def build_delta(states: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "first_cycle_edge_fraction",
        "first_cycle_force_fraction",
        "beta1_birth_density_5pct",
        "beta1_birth_density_10pct",
        "loop_force_share_5pct",
        "loop_force_share_10pct",
        "beta1_birth_area",
        "loop_force_persistence_area",
        "giant_component_area",
        "total_beta1_birth_density",
        "total_loop_force_share",
        "top5_force_share",
    ]
    cold = states[states["phase"] == "cold"].set_index(["run", "tag", "regime_id", "cycle"])
    hot = states[states["phase"] == "hot"].set_index(["run", "tag", "regime_id", "cycle"])
    rows: list[dict[str, float | str | int]] = []
    for idx in cold.index.intersection(hot.index):
        run, tag, regime_id, cycle = idx
        row: dict[str, float | str | int] = {"run": run, "tag": tag, "regime_id": regime_id, "cycle": int(cycle)}
        for col in cols:
            row[f"{col}_cold"] = float(cold.loc[idx, col])
            row[f"{col}_hot"] = float(hot.loc[idx, col])
            row[f"{col}_delta"] = float(hot.loc[idx, col] - cold.loc[idx, col])
        rows.append(row)
    delta = pd.DataFrame(rows).sort_values(["regime_id", "cycle"])
    target = pd.read_csv(SRC / "nphys_force_loop_spatial_fingerprint_cycle_delta.csv")[
        [
            "run",
            "tag",
            "regime_id",
            "cycle",
            "force_p99_hot_minus_cold",
            "overload_asinh",
            "loop_force_share_delta",
            "loop_conduit_force_share_delta",
        ]
    ]
    return delta.merge(target, on=["run", "tag", "regime_id", "cycle"], how="left")


def add_centred(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col + "_rc"] = route_center(out, col)
    out["overload_asinh_rc"] = route_center(out, "overload_asinh")
    return out


def correlations(delta: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        ("Betti-1 birth area", "beta1_birth_area_delta"),
        ("loop-force persistence area", "loop_force_persistence_area_delta"),
        ("early Betti-1 density", "beta1_birth_density_5pct_delta"),
        ("early loop-force share", "loop_force_share_5pct_delta"),
        ("total loop force", "total_loop_force_share_delta"),
        ("conduit-loop force", "loop_conduit_force_share_delta"),
        ("top-5 force tail", "top5_force_share_delta"),
    ]
    delta = add_centred(delta, [c for _, c in predictors])
    rows = []
    for name, col in predictors:
        raw_r, raw_p, n = safe_spearman(delta[col], delta["overload_asinh"])
        rc_r, rc_p, _ = safe_spearman(delta[col + "_rc"], delta["overload_asinh_rc"])
        rows.append(
            {
                "predictor": name,
                "column": col,
                "n": n,
                "spearman_raw": raw_r,
                "p_raw": raw_p,
                "spearman_route_centered": rc_r,
                "p_route_centered": rc_p,
            }
        )
    return pd.DataFrame(rows)


def leave_one_route(delta: pd.DataFrame) -> pd.DataFrame:
    model_defs = {
        "Betti area": ["beta1_birth_area_delta"],
        "loop persistence": ["loop_force_persistence_area_delta"],
        "early Betti": ["beta1_birth_density_5pct_delta"],
        "total loop": ["total_loop_force_share_delta"],
        "conduit loop": ["loop_conduit_force_share_delta"],
        "top-5 tail": ["top5_force_share_delta"],
        "topology pair": ["beta1_birth_area_delta", "loop_force_persistence_area_delta"],
    }
    y = delta["overload_asinh"].to_numpy(float)
    route = delta["regime_id"].to_numpy(str)
    rows = []
    for model, cols in model_defs.items():
        pred = np.full(len(delta), np.nan)
        for rid in np.unique(route):
            train = route != rid
            test = route == rid
            fit = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0]).fit(delta.loc[train, cols].fillna(0.0), y[train])
            pred[test] = fit.predict(delta.loc[test, cols].fillna(0.0))
        ok = np.isfinite(pred) & np.isfinite(y)
        rho, p, n = safe_spearman(pred[ok], y[ok])
        rows.append(
            {
                "model": model,
                "predictors": ";".join(cols),
                "validation": "leave_one_route_out",
                "n": int(n),
                "r2_vs_training_mean": float(r2_score(y[ok], pred[ok])),
                "spearman_y_yhat": rho,
                "spearman_p": p,
            }
        )
    return pd.DataFrame(rows)


def route_summary(delta: pd.DataFrame) -> pd.DataFrame:
    return (
        delta.groupby("regime_id", as_index=False)
        .agg(
            n=("cycle", "count"),
            mean_overload_asinh=("overload_asinh", "mean"),
            mean_beta1_birth_area_delta=("beta1_birth_area_delta", "mean"),
            mean_loop_force_persistence_area_delta=("loop_force_persistence_area_delta", "mean"),
            mean_total_loop_force_share_delta=("total_loop_force_share_delta", "mean"),
            mean_top5_force_share_delta=("top5_force_share_delta", "mean"),
        )
        .sort_values("mean_overload_asinh")
    )


def make_figure(curves: pd.DataFrame, delta: pd.DataFrame, corr: pd.DataFrame, models: pd.DataFrame, routes: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 0.95], height_ratios=[1.0, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    examples = [("R1", 10), ("R6c", 10)]
    for rid, cyc in examples:
        for phase, color, ls in [("cold", BLUE, "-"), ("hot", RED, "-")]:
            g = curves[(curves["regime_id"] == rid) & (curves["cycle"] == cyc) & (curves["phase"] == phase)]
            if g.empty:
                continue
            ax.plot(
                g["edge_fraction"],
                g["beta1_birth_density"],
                color=ROUTE_COLORS.get(rid, color),
                ls=ls if phase == "hot" else (0, (3, 2)),
                lw=1.25,
                label=f"{rid} {phase}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("force-ranked edge fraction")
    ax.set_ylabel(r"$\beta_1$ birth density")
    ax.set_title("force-filtration loop birth curves", loc="left", pad=4)
    ax.legend(fontsize=5.8, ncol=2, loc="upper left")
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    order = [
        "Betti-1 birth area",
        "loop-force persistence area",
        "early Betti-1 density",
        "early loop-force share",
        "total loop force",
        "conduit-loop force",
        "top-5 force tail",
    ]
    cc = corr.set_index("predictor").loc[order].reset_index()
    y = np.arange(len(cc))
    colors = [BLUE, RED, BLUE, RED, GOLD, VIOLET, GREY]
    ax.barh(y, cc["spearman_route_centered"], color=colors, alpha=0.88)
    ax.axvline(0, color="#AEB6C0", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(cc["predictor"], fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlabel("route-centred Spearman")
    ax.set_title("threshold-free topology retains signal", loc="left", pad=4)
    finish(ax, "x")
    panel(ax, "b", x=-0.18)

    ax = fig.add_subplot(gs[1, 0])
    delta = add_centred(delta, ["beta1_birth_area_delta"])
    for rid, g in delta.groupby("regime_id", sort=True):
        ax.scatter(
            g["beta1_birth_area_delta_rc"],
            g["overload_asinh_rc"],
            s=18,
            color=ROUTE_COLORS.get(rid, INK),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            lw=0.3,
            alpha=0.85,
            label=rid,
        )
    r = corr.loc[corr["predictor"] == "Betti-1 birth area"].iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={r.spearman_route_centered:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.3)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel(r"route-centred $\beta_1$ birth area")
    ax.set_ylabel("route-centred overload")
    ax.set_title("whole-filtration topology predicts overload", loc="left", pad=4)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    order = ["Betti area", "loop persistence", "early Betti", "total loop", "conduit loop", "top-5 tail", "topology pair"]
    mm = models.set_index("model").loc[order].reset_index()
    x = np.arange(len(mm))
    ax.bar(x, mm["r2_vs_training_mean"], color=[BLUE, RED, BLUE, GOLD, VIOLET, GREY, GREEN], alpha=0.88)
    ax.axhline(0, color="#AEB6C0", lw=0.7)
    ax.set_xticks(x, ["Betti\narea", "loop\narea", "early\nBetti", "total\nloop", "conduit", "top-5%", "topology\npair"], fontsize=5.6)
    ax.set_ylabel(r"leave-route-out $R^2$")
    ax.set_title("filtration metrics are support, not replacement", loc="left", pad=4)
    finish(ax, "y")
    panel(ax, "d", x=-0.18)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig68_force_filtration_topology.{ext}", **kwargs)
    plt.close(fig)


def write_report(corr: pd.DataFrame, models: pd.DataFrame, routes: pd.DataFrame) -> None:
    def cv(name: str) -> float:
        return float(corr.loc[corr["predictor"] == name, "spearman_route_centered"].iloc[0])

    def mv(name: str) -> float:
        return float(models.loc[models["model"] == name, "r2_vs_training_mean"].iloc[0])

    lines = [
        "# Eight-route force-filtration topology audit",
        "",
        "This reserve audit asks whether the force-loop mechanism is an artefact of a single graph threshold. Each true-force contact network is filtered from strongest to weakest contacts, and the full Betti-1 birth curve is summarised by threshold-independent area metrics before taking hot-minus-cold differences.",
        "",
        "## Correlations",
        "",
        corr.round(4).to_markdown(index=False),
        "",
        "## Leave-one-route transfer",
        "",
        models.round(4).to_markdown(index=False),
        "",
        "## Route summary",
        "",
        routes.round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"The whole-filtration topology signal supports the same mechanism as the force-loop coordinate. Route-centred overload correlates with loop-force persistence area (rho={cv('loop-force persistence area'):.3f}) and Betti-1 birth area (rho={cv('Betti-1 birth area'):.3f}), whereas the top-5% force-tail control is oppositely signed (rho={cv('top-5 force tail'):.3f}).",
        "",
        f"Transfer remains bounded: leave-one-route R2 is {mv('loop persistence'):.3f} for loop-force persistence area, {mv('Betti area'):.3f} for Betti area and {mv('total loop'):.3f} for the simpler total loop-force coordinate. The filtration metrics therefore defend threshold robustness, but they do not replace the cycle-closing loop-force coordinate used in the main mechanism.",
        "",
        "Allowed wording: hot overload is encoded across the force-network filtration, not only at one arbitrary force threshold. Not allowed: this is not a universal topological order parameter, not persistent-homology proof of a phase transition and not independent experimental validation.",
    ]
    (ROOT / "nature_physics_force_filtration_topology.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    state_path = SRC / "nphys_force_filtration_topology_state_metrics.csv"
    curve_path = SRC / "nphys_force_filtration_topology_curves.csv"
    if state_path.exists() and curve_path.exists():
        states = pd.read_csv(state_path)
        curves = pd.read_csv(curve_path)
    else:
        state_rows: list[dict[str, float | str | int]] = []
        curve_rows: list[dict[str, float | str | int]] = []
        for key in discover_keys():
            state, curves_for_state = state_filtration(key)
            state_rows.append(state)
            curve_rows.extend(curves_for_state)
        states = pd.DataFrame(state_rows).sort_values(["regime_id", "cycle", "phase"])
        curves = pd.DataFrame(curve_rows).sort_values(["regime_id", "cycle", "phase", "edge_fraction"])
    delta = build_delta(states)
    corr = correlations(delta)
    models = leave_one_route(delta)
    routes = route_summary(delta)
    states.to_csv(SRC / "nphys_force_filtration_topology_state_metrics.csv", index=False)
    curves.to_csv(SRC / "nphys_force_filtration_topology_curves.csv", index=False)
    delta.to_csv(SRC / "nphys_force_filtration_topology_cycle_delta.csv", index=False)
    corr.to_csv(SRC / "nphys_force_filtration_topology_correlations.csv", index=False)
    models.to_csv(SRC / "nphys_force_filtration_topology_transfer_tests.csv", index=False)
    routes.to_csv(SRC / "nphys_force_filtration_topology_route_summary.csv", index=False)
    make_figure(curves, delta, corr, models, routes)
    write_report(corr, models, routes)
    print(f"states={len(states)} pairs={len(delta)} curves={len(curves)}")
    print(corr[["predictor", "spearman_route_centered", "p_route_centered"]].round(3).to_string(index=False))
    print(models[["model", "r2_vs_training_mean", "spearman_y_yhat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
