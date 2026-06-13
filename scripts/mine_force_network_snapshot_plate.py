#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mine_long_cycle_true_force_memory import LONG_REGIMES, RUN_BASE
from mine_true_force_percolation import ForceState, UnionFind, read_dump, read_local


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
METRICS = SRC / "nphys_force_loop_conduit_gating_cycle_metrics.csv"

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
HOT = "#B6423E"
GOLD = "#D98C3A"
BLUE = "#3D6B9C"


@dataclass(frozen=True)
class SnapshotCase:
    label: str
    tag: str
    regime_id: str
    cycle: int
    phase: str


CASES = [
    SnapshotCase("early dangerous inhale", "a150_mu060_g000", "R6c", 1, "hot"),
    SnapshotCase("later buffered inhale", "a150_mu060_g000", "R6c", 12, "hot"),
]


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


def panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.42, zorder=0)
    ax.tick_params(width=0.65)


def state_for(case: SnapshotCase) -> ForceState:
    folder = RUN_BASE / LONG_REGIMES[case.tag][0]
    local = folder / f"contacts_cycle_{case.cycle}_{case.phase}.local"
    dump = folder / f"cycle_{case.cycle}_{case.phase}.dump"
    if not local.exists() or not dump.exists():
        raise FileNotFoundError(f"Missing snapshot files for {case}")
    return ForceState(case.tag, case.phase, read_local(local), read_dump(dump))


def classify_edges(state: ForceState, case: SnapshotCase) -> pd.DataFrame:
    contacts = state.contacts.copy()
    atoms = state.atoms.copy()
    valid = contacts["id1"].isin(atoms.index) & contacts["id2"].isin(atoms.index) & (contacts["force"] > 0)
    d = contacts.loc[valid].copy()
    ids = np.array(sorted(set(d["id1"]).union(set(d["id2"]))))
    id_to_idx = {int(v): i for i, v in enumerate(ids)}
    x = atoms.loc[ids, "x"].to_numpy(float)
    y = atoms.loc[ids, "y"].to_numpy(float)
    z = atoms.loc[ids, "z"].to_numpy(float)
    r = np.sqrt(x**2 + y**2)
    bottom = z <= np.quantile(z, 0.03)
    top = z >= np.quantile(z, 0.97)
    side = r >= np.quantile(r, 0.97)
    uf = UnionFind(len(ids), bottom, top, side)

    order = np.argsort(d["force"].to_numpy(float))[::-1]
    ranks = np.empty(len(d), dtype=int)
    cycle_closing = np.zeros(len(d), dtype=bool)
    bottom_side_component = np.zeros(len(d), dtype=bool)
    for rank, row_idx in enumerate(order, start=1):
        row = d.iloc[int(row_idx)]
        a = id_to_idx[int(row["id1"])]
        b = id_to_idx[int(row["id2"])]
        ranks[int(row_idx)] = rank
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            cycle_closing[int(row_idx)] = True
            root = ra
        else:
            root, _merged = uf.union(a, b)
        bottom_side_component[int(row_idx)] = bool(uf.bottom[root] and uf.side[root])

    idx1 = np.array([id_to_idx[int(v)] for v in d["id1"]], dtype=int)
    idx2 = np.array([id_to_idx[int(v)] for v in d["id2"]], dtype=int)
    out = pd.DataFrame(
        {
            "case_label": case.label,
            "tag": case.tag,
            "regime_id": case.regime_id,
            "cycle": case.cycle,
            "phase": case.phase,
            "id1": d["id1"].to_numpy(int),
            "id2": d["id2"].to_numpy(int),
            "force": d["force"].to_numpy(float),
            "rank": ranks,
            "edge_fraction": ranks / len(d),
            "cycle_closing": cycle_closing,
            "bottom_side_component_at_insertion": bottom_side_component,
            "r1": r[idx1],
            "z1": z[idx1],
            "r2": r[idx2],
            "z2": z[idx2],
        }
    )
    out["force_rank_percentile"] = 1 - (out["rank"] - 1) / max(len(out) - 1, 1)
    out["display_edge"] = (out["edge_fraction"] <= 0.018) | (out["cycle_closing"] & (out["edge_fraction"] <= 0.09))
    return out.sort_values("rank")


def snapshot_summary(edges: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tag, cycle, phase), g in edges.groupby(["tag", "cycle", "phase"], sort=False):
        m = metrics[(metrics["tag"] == tag) & (metrics["cycle"] == cycle)]
        mrow = m.iloc[0] if len(m) else pd.Series(dtype=float)
        displayed = g[g["display_edge"]]
        rows.append(
            {
                "case_label": g["case_label"].iloc[0],
                "tag": tag,
                "regime_id": g["regime_id"].iloc[0],
                "cycle": int(cycle),
                "phase": phase,
                "n_contacts": int(len(g)),
                "n_display_edges": int(len(displayed)),
                "display_cycle_closing_edges": int(displayed["cycle_closing"].sum()),
                "display_cycle_closing_force_share": float(displayed.loc[displayed["cycle_closing"], "force"].sum() / g["force"].sum()),
                "top_1p_force_share": float(g.loc[g["edge_fraction"] <= 0.01, "force"].sum() / g["force"].sum()),
                "top_5p_force_share": float(g.loc[g["edge_fraction"] <= 0.05, "force"].sum() / g["force"].sum()),
                "cycle_closing_force_share_total": float(g.loc[g["cycle_closing"], "force"].sum() / g["force"].sum()),
                "bottom_side_edge_fraction_first": float(g.loc[g["bottom_side_component_at_insertion"], "edge_fraction"].min())
                if g["bottom_side_component_at_insertion"].any()
                else np.nan,
                "overload_asinh": float(mrow.get("overload_asinh", np.nan)),
                "loop_activation": float(mrow.get("loop_activation", np.nan)),
                "wall_conduit_activation": float(mrow.get("wall_conduit_activation", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def plot_edges(ax: plt.Axes, edges: pd.DataFrame, summary: pd.Series) -> None:
    show = edges[edges["display_edge"]].copy()
    nonloop = show[~show["cycle_closing"]]
    loops = show[show["cycle_closing"]]
    fmax = show["force"].quantile(0.99)
    fmin = show["force"].quantile(0.08)

    def widths(force: pd.Series, scale: float) -> np.ndarray:
        vals = np.clip((force.to_numpy(float) - fmin) / max(fmax - fmin, 1e-12), 0, 1)
        return scale * (0.25 + 1.75 * vals)

    for _, row in nonloop.iterrows():
        ax.plot(
            [row["r1"], row["r2"]],
            [row["z1"], row["z2"]],
            color="#4E5865",
            lw=widths(pd.Series([row["force"]]), 0.42)[0],
            alpha=0.15,
            zorder=2,
        )
    for _, row in loops.iterrows():
        conduit = bool(row["bottom_side_component_at_insertion"])
        ax.plot(
            [row["r1"], row["r2"]],
            [row["z1"], row["z2"]],
            color=HOT if conduit else GOLD,
            lw=widths(pd.Series([row["force"]]), 0.82)[0],
            alpha=0.70 if conduit else 0.45,
            zorder=4 if conduit else 3,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("radial position")
    ax.set_ylabel("height")
    ax.set_title(summary["case_label"], loc="left", pad=4)
    ax.text(
        0.03,
        0.97,
        rf"cycle {int(summary['cycle'])}; $\Omega_s={summary['overload_asinh']:.2f}$"
        + "\n"
        + rf"loop share={summary['cycle_closing_force_share_total']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=INK,
    )
    finish(ax)


def make_figure(edges: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 0.82], height_ratios=[1.0, 0.72])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])

    for ax, case, label in [(ax_a, CASES[0], "a"), (ax_b, CASES[1], "b")]:
        subset = edges[(edges["tag"] == case.tag) & (edges["cycle"] == case.cycle) & (edges["phase"] == case.phase)]
        srow = summary[(summary["tag"] == case.tag) & (summary["cycle"] == case.cycle)].iloc[0]
        plot_edges(ax, subset, srow)
        panel(ax, label, x=-0.06)

    order = summary["case_label"].tolist()
    x = np.arange(len(summary))
    ax_c.bar(
        x - 0.18,
        summary["cycle_closing_force_share_total"],
        width=0.34,
        color=HOT,
        label="cycle-closing",
    )
    ax_c.bar(
        x + 0.18,
        summary["top_5p_force_share"],
        width=0.34,
        color="#687380",
        label="top-5%",
    )
    ax_c.set_xticks(x, ["early", "later"])
    ax_c.set_ylabel("force share")
    ax_c.set_title("force carried by motifs", loc="left", pad=4)
    ax_c.text(
        0.03,
        0.94,
        "tail alone is not decisive",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=MUTED,
    )
    ax_c.legend(fontsize=5.8, loc="upper right")
    finish(ax_c, "y")
    panel(ax_c, "c", x=-0.20)

    ax_d.bar(x, summary["overload_asinh"], color=[HOT, BLUE], width=0.58)
    ax_d.set_xticks(x, ["early", "later"])
    ax_d.set_ylabel("asinh overload")
    ax_d.set_title("same route, weaker breath", loc="left", pad=4)
    finish(ax_d, "y")
    panel(ax_d, "d", x=-0.20)

    handles = [
        plt.Line2D([0], [0], color="#4E5865", lw=1.2, alpha=0.35, label="strong force edge"),
        plt.Line2D([0], [0], color=GOLD, lw=1.6, alpha=0.65, label="cycle-closing edge"),
        plt.Line2D([0], [0], color=HOT, lw=1.8, alpha=0.80, label="cycle-closing conduit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.2, bbox_to_anchor=(0.48, -0.015))
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig47_force_network_snapshot_plate.{ext}", **kwargs)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    early = summary.iloc[0]
    late = summary.iloc[1]
    lines = [
        "# Force-network snapshot plate",
        "",
        "This reserve visual audit turns the force-loop mechanism into a spatial object using true contact-local force outputs. It compares two hot states from the same R6c route: an early dangerous inhale and a later buffered inhale.",
        "",
        "## Snapshot summary",
        "",
        summary.round(5).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"The early R6c hot state has asinh overload {early['overload_asinh']:.2f}, compared with {late['overload_asinh']:.2f} for the later same-route state. The total force share carried by cycle-closing edges decreases from {early['cycle_closing_force_share_total']:.2f} to {late['cycle_closing_force_share_total']:.2f}.",
        "",
        "The figure is a visual audit rather than an independent statistical test. It should be used to make the abstract force-loop mechanism inspectable: the dangerous branch is not shown as a single force chain, but as force-carrying cycle-closing contacts embedded in a wall-coupled network projection.",
        "",
        "Allowed wording: representative true-force snapshots show how early high-overload breathing is accompanied by a denser set of force-carrying cycle-closing edges in radial-height projection. Not allowed: this two-snapshot plate does not prove universal spatial morphology or replace the five-route quantitative audits.",
    ]
    (ROOT / "nature_physics_force_network_snapshot_plate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    metrics = pd.read_csv(METRICS)
    tables = []
    for case in CASES:
        tables.append(classify_edges(state_for(case), case))
    edges = pd.concat(tables, ignore_index=True)
    summary = snapshot_summary(edges, metrics)
    edges.to_csv(SRC / "nphys_force_network_snapshot_plate_edges.csv", index=False)
    summary.to_csv(SRC / "nphys_force_network_snapshot_plate_summary.csv", index=False)
    make_figure(edges, summary)
    write_report(summary)
    print("Wrote force-network snapshot plate")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
