#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mine_eight_route_force_filtration_topology import (
    FIG,
    GRID,
    INK,
    OVERLOAD_SCALE,
    RED,
    SRC,
    THRESHOLDS,
    finish,
    panel,
    safe_spearman,
    setup_style,
)
from mine_force_loop_spatial_fingerprint import StateKey, discover_keys, read_state, route_center
from mine_true_force_percolation import UnionFind


ROOT = Path(__file__).resolve().parent
N_PERM = 25
SEED = 20260613
BLUE = "#3D6B9C"
GOLD = "#D98C3A"
GREY = "#87919C"


@dataclass(frozen=True)
class GraphState:
    key: StateKey
    edges: np.ndarray
    n_nodes: int


def graph_state(key: StateKey) -> GraphState | None:
    state = read_state(key)
    contacts = state.contacts.copy()
    atoms = state.atoms.copy()
    valid = contacts["id1"].isin(atoms.index) & contacts["id2"].isin(atoms.index) & (contacts["force"] > 0)
    d = contacts.loc[valid].copy()
    if d.empty:
        return None
    ids = np.array(sorted(set(d["id1"]).union(set(d["id2"]))))
    id_to_idx = {int(v): i for i, v in enumerate(ids)}
    edges = np.column_stack(
        [
            [id_to_idx[int(v)] for v in d["id1"]],
            [id_to_idx[int(v)] for v in d["id2"]],
        ]
    ).astype(np.int64)
    return GraphState(key=key, edges=edges, n_nodes=len(ids))


def beta_area_for_order(edges: np.ndarray, n_nodes: int, order: np.ndarray) -> float:
    n_edges = len(order)
    if n_edges == 0 or n_nodes == 0:
        return np.nan
    dummy = np.zeros(n_nodes, dtype=bool)
    uf = UnionFind(n_nodes, dummy, dummy, dummy)
    threshold_ranks = np.maximum(1, np.ceil(THRESHOLDS * n_edges).astype(int))
    lookup = {int(rank): i for i, rank in enumerate(threshold_ranks)}
    curve = np.full(len(THRESHOLDS), np.nan, dtype=float)
    births = 0
    for rank, edge_idx in enumerate(order, start=1):
        a = int(edges[int(edge_idx), 0])
        b = int(edges[int(edge_idx), 1])
        if uf.find(a) == uf.find(b):
            births += 1
        else:
            uf.union(a, b)
        if rank in lookup:
            curve[lookup[rank]] = births / rank
    if np.any(~np.isfinite(curve)):
        curve = pd.Series(curve).ffill().bfill().to_numpy(float)
    return float(np.trapz(curve, THRESHOLDS))


def build_null_state_metrics(n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []
    for key in discover_keys():
        gs = graph_state(key)
        if gs is None:
            continue
        n_edges = len(gs.edges)
        for perm in range(n_perm):
            order = rng.permutation(n_edges)
            rows.append(
                {
                    "perm": perm,
                    "run": key.run,
                    "tag": key.tag,
                    "regime_id": key.regime_id,
                    "cycle": key.cycle,
                    "phase": key.phase,
                    "null_beta1_birth_area": beta_area_for_order(gs.edges, gs.n_nodes, order),
                    "n_edges": n_edges,
                    "n_nodes": gs.n_nodes,
                }
            )
    return pd.DataFrame(rows)


def build_null_delta(null_states: pd.DataFrame) -> pd.DataFrame:
    cold = null_states[null_states["phase"] == "cold"].set_index(["perm", "run", "tag", "regime_id", "cycle"])
    hot = null_states[null_states["phase"] == "hot"].set_index(["perm", "run", "tag", "regime_id", "cycle"])
    rows: list[dict[str, float | str | int]] = []
    for idx in cold.index.intersection(hot.index):
        perm, run, tag, rid, cycle = idx
        rows.append(
            {
                "perm": int(perm),
                "run": run,
                "tag": tag,
                "regime_id": rid,
                "cycle": int(cycle),
                "null_beta1_birth_area_delta": float(hot.loc[idx, "null_beta1_birth_area"] - cold.loc[idx, "null_beta1_birth_area"]),
            }
        )
    delta = pd.DataFrame(rows)
    target = pd.read_csv(SRC / "nphys_force_filtration_topology_cycle_delta.csv")[
        ["run", "tag", "regime_id", "cycle", "overload_asinh", "beta1_birth_area_delta"]
    ]
    return delta.merge(target, on=["run", "tag", "regime_id", "cycle"], how="left")


def summarise(null_delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = null_delta.drop_duplicates(["run", "tag", "regime_id", "cycle"]).copy()
    observed["beta1_birth_area_delta_rc"] = route_center(observed, "beta1_birth_area_delta")
    observed["overload_asinh_rc"] = route_center(observed, "overload_asinh")
    obs_rho, obs_p, obs_n = safe_spearman(observed["beta1_birth_area_delta_rc"], observed["overload_asinh_rc"])

    rows: list[dict[str, float | int | str]] = []
    for perm, g in null_delta.groupby("perm", sort=True):
        g = g.copy()
        g["null_beta1_birth_area_delta_rc"] = route_center(g, "null_beta1_birth_area_delta")
        g["overload_asinh_rc"] = route_center(g, "overload_asinh")
        rho, p, n = safe_spearman(g["null_beta1_birth_area_delta_rc"], g["overload_asinh_rc"])
        rows.append({"perm": int(perm), "null_rho": rho, "null_p": p, "n": n})
    null = pd.DataFrame(rows)
    p_emp = float((np.sum(null["null_rho"].abs() >= abs(obs_rho)) + 1) / (len(null) + 1))
    summary = pd.DataFrame(
        [
            {
                "observed_rho": obs_rho,
                "observed_p": obs_p,
                "observed_n": obs_n,
                "null_permutations": int(len(null)),
                "null_rho_median": float(null["null_rho"].median()),
                "null_rho_q025": float(null["null_rho"].quantile(0.025)),
                "null_rho_q975": float(null["null_rho"].quantile(0.975)),
                "empirical_p_two_sided": p_emp,
            }
        ]
    )
    return null, summary


def make_figure(null: pd.DataFrame, summary: pd.DataFrame, null_delta: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05])

    obs = summary.iloc[0]
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(null["null_rho"], bins=12, color=GREY, edgecolor="white", lw=0.4)
    ax.axvline(obs["observed_rho"], color=RED, lw=1.4, label="observed")
    ax.axvline(obs["null_rho_median"], color=INK, lw=1.0, ls=(0, (3, 2)), label="null median")
    ax.set_xlabel(r"route-centred $\rho$")
    ax.set_ylabel("force-order shuffles")
    ax.set_title("Betti area survives force-order shuffle", loc="left", pad=4)
    ax.text(0.05, 0.94, rf"$P_{{null}}={obs.empirical_p_two_sided:.3f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.4)
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 0.82))
    finish(ax, "y")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    observed = null_delta.drop_duplicates(["run", "tag", "regime_id", "cycle"]).copy()
    observed["beta1_birth_area_delta_rc"] = route_center(observed, "beta1_birth_area_delta")
    observed["overload_asinh_rc"] = route_center(observed, "overload_asinh")
    for rid, g in observed.groupby("regime_id", sort=True):
        ax.scatter(g["beta1_birth_area_delta_rc"], g["overload_asinh_rc"], s=15, alpha=0.72, edgecolor="white", lw=0.25)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel(r"observed $\Delta A_{\beta_1}$")
    ax.set_ylabel("route-centred overload")
    ax.set_title("observed filtration topology", loc="left", pad=4)
    ax.text(0.05, 0.94, rf"$\rho={obs.observed_rho:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.4)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    first_perm = int(null["perm"].iloc[0])
    g = null_delta[null_delta["perm"] == first_perm].copy()
    g["null_beta1_birth_area_delta_rc"] = route_center(g, "null_beta1_birth_area_delta")
    g["overload_asinh_rc"] = route_center(g, "overload_asinh")
    rho = float(null.loc[null["perm"] == first_perm, "null_rho"].iloc[0])
    ax.scatter(g["null_beta1_birth_area_delta_rc"], g["overload_asinh_rc"], s=15, color=GREY, alpha=0.72, edgecolor="white", lw=0.25)
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel(r"shuffled $\Delta A_{\beta_1}$")
    ax.set_ylabel("route-centred overload")
    ax.set_title("same graphs, shuffled force order", loc="left", pad=4)
    ax.text(0.05, 0.94, rf"$\rho={rho:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.4)
    finish(ax)
    panel(ax, "c")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig69_force_filtration_shuffle_null.{ext}", **kwargs)
    plt.close(fig)


def write_report(null: pd.DataFrame, summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    lines = [
        "# Force-filtration force-order shuffle null",
        "",
        "This reserve audit tests whether the Betti-1 force-filtration signal depends on the measured ordering of contact forces. The contact graph of each state is kept fixed, but edge order in the filtration is randomly shuffled before recomputing the Betti-1 birth-area hot-minus-cold difference.",
        "",
        "## Summary",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Null distribution",
        "",
        null.round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"The observed route-centred association between Betti-1 birth-area change and overload is rho={row.observed_rho:.3f}. Force-order shuffles give a median rho={row.null_rho_median:.3f} with a 95% interval [{row.null_rho_q025:.3f}, {row.null_rho_q975:.3f}], giving empirical P={row.empirical_p_two_sided:.3f} for exceeding the observed magnitude. The null therefore is not rejected.",
        "",
        "Allowed wording: Betti-1 birth area records a route-severity topology background that remains visible even when force order is shuffled. This bounds the topology audit and explains why the manuscript should keep the force-weighted cycle-closing loop coordinate as the primary overload variable. Not allowed: do not claim that Betti-1 area proves force-order-specific overload physics, a persistent-homology phase transition or independent experimental validation.",
    ]
    (ROOT / "nature_physics_force_filtration_shuffle_null.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    state_path = SRC / "nphys_force_filtration_shuffle_null_state_metrics.csv"
    delta_path = SRC / "nphys_force_filtration_shuffle_null_cycle_delta.csv"
    if state_path.exists() and delta_path.exists():
        null_states = pd.read_csv(state_path)
        null_delta = pd.read_csv(delta_path)
    else:
        null_states = build_null_state_metrics()
        null_delta = build_null_delta(null_states)
        null_states.to_csv(state_path, index=False)
        null_delta.to_csv(delta_path, index=False)
    null, summary = summarise(null_delta)
    null.to_csv(SRC / "nphys_force_filtration_shuffle_null_distribution.csv", index=False)
    summary.to_csv(SRC / "nphys_force_filtration_shuffle_null_summary.csv", index=False)
    make_figure(null, summary, null_delta)
    write_report(null, summary)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
