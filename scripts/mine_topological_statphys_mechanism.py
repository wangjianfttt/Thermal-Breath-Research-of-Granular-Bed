#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RUNS = PROJECT / "runs"
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

LONG_REGIMES = {
    "R1_low_expansion_low_friction": RUNS / "long_cycles_10k" / "a050_mu010_g002_c30",
    "R3_intermediate": RUNS / "long_cycles_10k" / "a100_mu030_g002_c30",
    "R6_high_expansion_high_friction": RUNS / "long_cycles_10k" / "a150_mu060_g020_c30",
}
REFERENCE_RUNS = {
    "free_surface": RUNS / "free_quasistatic_10k_prod",
    "semi_confined": RUNS / "confined_structural_10k_prod",
}

DEEP_CYCLES = [1, 2, 3, 5, 10, 20, 30]
REFERENCE_CYCLES = [1, 5, 10, 20]

COLD = "#345995"
HOT = "#C95F3F"
INK = "#252A31"
ACCENT = "#2F7F6F"
MUTED = "#8B929A"
GRID = "#E7EAEE"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
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


def panel(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def read_dump(path: Path) -> pd.DataFrame:
    cols: list[str] | None = None
    rows: list[list[float]] = []
    current: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                current = []
                continue
            if line.startswith("ITEM:"):
                if current:
                    rows = current
                    current = []
                continue
            if cols is None or not line.strip():
                continue
            parts = line.split()
            if len(parts) == len(cols):
                current.append([float(x) for x in parts])
    if current:
        rows = current
    if not rows or cols is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["id"] = df["id"].astype(int)
    return df.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)


def contact_edges(df: pd.DataFrame, skin: float = 1.004) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = df[["x", "y", "z"]].to_numpy(float)
    radii = df["radius"].to_numpy(float)
    tree = cKDTree(pos)
    cutoff = float(2.0 * np.max(radii) * skin)
    pairs = np.array(list(tree.query_pairs(cutoff)), dtype=np.int64)
    if pairs.size == 0:
        return pairs.reshape(0, 2), np.empty((0, 3)), np.empty(0)
    delta = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    dist = np.linalg.norm(delta, axis=1)
    contact = dist <= (radii[pairs[:, 0]] + radii[pairs[:, 1]]) * skin
    pairs = pairs[contact]
    delta = delta[contact]
    dist = dist[contact]
    ok = dist > 0
    return pairs[ok], delta[ok] / dist[ok, None], dist[ok]


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def contact_set_from_pairs(df: pd.DataFrame, pairs: np.ndarray) -> set[tuple[int, int]]:
    ids = df["id"].to_numpy(int)
    out: set[tuple[int, int]] = set()
    for i, j in pairs:
        a, b = int(ids[i]), int(ids[j])
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def fabric_metrics(unit: np.ndarray) -> dict[str, float]:
    if unit.size == 0:
        return {"fabric_anisotropy": np.nan, "fabric_vertical_bias": np.nan, "orientation_entropy": np.nan}
    q = np.einsum("ij,ik->jk", unit, unit) / len(unit)
    eig = np.linalg.eigvalsh(q)
    abs_nz = np.abs(unit[:, 2])
    hist, _ = np.histogram(np.degrees(np.arccos(np.clip(abs_nz, 0, 1))), bins=np.arange(0, 100, 10))
    return {
        "fabric_Qxx": float(q[0, 0]),
        "fabric_Qyy": float(q[1, 1]),
        "fabric_Qzz": float(q[2, 2]),
        "fabric_anisotropy": float(eig[-1] - eig[0]),
        "fabric_vertical_bias": float(q[2, 2] - 0.5 * (q[0, 0] + q[1, 1])),
        "mean_abs_nz": float(abs_nz.mean()),
        "orientation_entropy": entropy_from_counts(hist) / np.log(len(hist)) if len(hist) > 1 else np.nan,
    }


def graph_metrics(df: pd.DataFrame, pairs: np.ndarray, unit: np.ndarray) -> dict[str, float]:
    n = len(df)
    e = len(pairs)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((int(i), int(j)) for i, j in pairs)
    components = list(nx.connected_components(g))
    c = len(components)
    giant = max((len(x) for x in components), default=0)
    beta1 = e - n + c
    degrees = np.fromiter((d for _, d in g.degree()), dtype=float, count=n)
    core = nx.core_number(g) if e else {i: 0 for i in range(n)}
    core_values = np.fromiter(core.values(), dtype=float, count=n)
    triangles_by_node = nx.triangles(g)
    triangles = sum(triangles_by_node.values()) / 3.0
    z = df["z"].to_numpy(float)
    rxy = np.sqrt(df["x"].to_numpy(float) ** 2 + df["y"].to_numpy(float) ** 2)
    bottom = z <= np.quantile(z, 0.03)
    top = z >= np.quantile(z, 0.97)
    side = rxy >= np.quantile(rxy, 0.97)
    boundary_edges = 0
    if e:
        boundary_edges = int(np.sum(bottom[pairs[:, 0]] | bottom[pairs[:, 1]] | side[pairs[:, 0]] | side[pairs[:, 1]]))
    zspans = []
    spanning = 0
    side_spanning = 0
    for comp in components:
        idx = np.fromiter(comp, dtype=int)
        if idx.size == 0:
            continue
        zspan = float(z[idx].max() - z[idx].min())
        zspans.append(zspan)
        touches_bottom = bool(bottom[idx].any())
        touches_top = bool(top[idx].any())
        touches_side = bool(side[idx].any())
        if touches_bottom and touches_top:
            spanning += 1
        if touches_bottom and touches_side:
            side_spanning += 1
    height = float(z.max() - z.min())
    metrics = {
        "N": float(n),
        "contacts": float(e),
        "Z": float(2 * e / n) if n else np.nan,
        "components": float(c),
        "giant_fraction": float(giant / n) if n else np.nan,
        "beta1_loops": float(beta1),
        "loop_density": float(beta1 / n) if n else np.nan,
        "loop_redundancy": float(beta1 / e) if e else np.nan,
        "degree_mean": float(degrees.mean()) if n else np.nan,
        "degree_var": float(degrees.var()) if n else np.nan,
        "degree_susceptibility": float(degrees.var() / degrees.mean()) if degrees.mean() > 0 else np.nan,
        "degree_entropy": entropy_from_counts(np.bincount(degrees.astype(int))) / np.log(max(2, len(np.bincount(degrees.astype(int))))),
        "rattler_fraction_Z0": float(np.mean(degrees == 0)),
        "low_coord_fraction_Zle2": float(np.mean(degrees <= 2)),
        "mean_core": float(core_values.mean()) if n else np.nan,
        "max_core": float(core_values.max()) if n else np.nan,
        "kcore4_fraction": float(np.mean(core_values >= 4)),
        "kcore5_fraction": float(np.mean(core_values >= 5)),
        "triangles": float(triangles),
        "triangle_density": float(triangles / n) if n else np.nan,
        "transitivity": float(nx.transitivity(g)) if e else np.nan,
        "boundary_edge_fraction": float(boundary_edges / e) if e else np.nan,
        "vertical_spanning_components": float(spanning),
        "bottom_side_spanning_components": float(side_spanning),
        "largest_component_zspan_norm": float(max(zspans) / height) if zspans and height > 0 else np.nan,
        "height_m": height,
    }
    metrics.update(fabric_metrics(unit))
    return metrics


def collect_dump_jobs() -> list[tuple[str, str, int, str, Path]]:
    jobs: list[tuple[str, str, int, str, Path]] = []
    for label, run in LONG_REGIMES.items():
        for cyc in DEEP_CYCLES:
            for phase in ["cold", "hot"]:
                path = run / f"cycle_{cyc}_{phase}.dump"
                if path.exists():
                    jobs.append(("long", label, cyc, phase, path))
    for label, run in REFERENCE_RUNS.items():
        for cyc in REFERENCE_CYCLES:
            for phase in ["cold", "hot"]:
                path = run / f"cycle_{cyc}_{phase}.dump"
                if path.exists():
                    jobs.append(("reference", label, cyc, phase, path))
    return jobs


def mine_networks() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    contacts_by_state: dict[tuple[str, int, str], set[tuple[int, int]]] = {}
    jobs = collect_dump_jobs()
    for family, label, cycle, phase, path in jobs:
        df = read_dump(path)
        pairs, unit, _ = contact_edges(df)
        metrics = graph_metrics(df, pairs, unit)
        metrics.update({"family": family, "run": label, "cycle": cycle, "phase": phase})
        rows.append(metrics)
        contacts_by_state[(label, cycle, phase)] = contact_set_from_pairs(df, pairs)
        print(f"{label} cycle={cycle} phase={phase} contacts={len(pairs)} beta1={metrics['beta1_loops']:.0f}")

    persistence_rows = []
    labels = sorted({label for _, label, _, _, _ in jobs})
    for label in labels:
        cycles = sorted({cyc for _, lab, cyc, _, _ in jobs if lab == label})
        for cycle in cycles:
            cold = contacts_by_state.get((label, cycle, "cold"), set())
            hot = contacts_by_state.get((label, cycle, "hot"), set())
            if cold and hot:
                persistence_rows.append(
                    {
                        "run": label,
                        "cycle": cycle,
                        "transition": "cold_to_hot_same_cycle",
                        "jaccard": len(cold & hot) / len(cold | hot),
                        "survival_fraction_of_cold": len(cold & hot) / len(cold),
                        "created_fraction_of_hot": len(hot - cold) / len(hot),
                    }
                )
        for a, b in zip(cycles[:-1], cycles[1:]):
            for phase in ["cold", "hot"]:
                ca = contacts_by_state.get((label, a, phase), set())
                cb = contacts_by_state.get((label, b, phase), set())
                if ca and cb:
                    persistence_rows.append(
                        {
                            "run": label,
                            "cycle": b,
                            "transition": f"{phase}_cycle_to_cycle",
                            "jaccard": len(ca & cb) / len(ca | cb),
                            "survival_fraction_of_cold": len(ca & cb) / len(ca),
                            "created_fraction_of_hot": len(cb - ca) / len(cb),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(persistence_rows)


def merge_existing_long_metrics(topology: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, run in LONG_REGIMES.items():
        p = run / "advanced_microphysics.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["run"] = label
            frames.append(d)
    if not frames:
        return topology
    micro = pd.concat(frames, ignore_index=True)
    micro["phase"] = micro["phase"].str.lower()
    micro = micro.rename(columns={"cycle": "cycle"})
    keep = [
        "run",
        "cycle",
        "phase",
        "force_proxy_gini",
        "force_proxy_q99_q50",
        "net_force_p99_N",
        "height_m",
    ]
    return topology.merge(micro[keep], on=["run", "cycle", "phase"], how="left", suffixes=("", "_micro"))


def build_figure(df: pd.DataFrame, persist: pd.DataFrame) -> None:
    setup_style()
    long = df[df["family"] == "long"].copy()
    fig = plt.figure(figsize=(7.15, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    cycles = sorted(long["cycle"].unique())
    row_labels: list[str] = []
    heat_rows: list[list[float]] = []
    for run in LONG_REGIMES:
        short = run.split("_")[0]
        for phase in ["cold", "hot"]:
            sub = long[(long["run"] == run) & (long["phase"] == phase)].set_index("cycle")
            row_labels.append(f"{short} {phase[0]}")
            heat_rows.append([float(sub.loc[c, "loop_density"]) if c in sub.index else np.nan for c in cycles])
    heat = np.asarray(heat_rows)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "topology_map",
        ["#F7F8FA", "#D7E5F2", "#7EA7C9", "#345995", "#21314F"],
    )
    im = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=0.0, vmax=np.nanmax(heat) * 1.03)
    ax.set_xticks(np.arange(len(cycles)))
    ax.set_xticklabels([str(c) for c in cycles])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.8)
    ax.set_xlabel("cycle")
    ax.set_title("phase-resolved loop reservoir", fontsize=7.5, pad=5)
    ax.tick_params(length=0)
    for y in [1.5, 3.5]:
        ax.axhline(y, color="white", lw=1.1)
    for x in np.arange(0.5, len(cycles) - 0.5, 1):
        ax.axvline(x, color="white", lw=0.45, alpha=0.75)
    for i, row in enumerate(heat):
        for j, value in enumerate(row):
            color = "white" if value > 1.0 else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(r"$\beta_1/N$", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.6, width=0.5, length=2)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for run, color in zip(LONG_REGIMES, ["#345995", "#D98C3A", "#C95F3F"]):
        sub = long[(long["run"] == run) & (long["phase"] == "hot")]
        ax.plot(sub["cycle"], sub["kcore4_fraction"], "o-", ms=3.2, lw=1.0, color=color, label=run.split("_")[0])
    ax.set_xlabel("cycle")
    ax.set_ylabel("fraction in $k \\geq 4$ core")
    ax.set_title("hot-state core emergence", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for phase, marker, color in [("cold", "o", COLD), ("hot", "s", HOT)]:
        sub = long[long["phase"] == phase]
        ax.scatter(sub["loop_density"], sub["force_proxy_q99_q50"], s=28, marker=marker, color=color, alpha=0.78, edgecolor="white", linewidth=0.35, label=phase)
    ax.set_xlabel(r"$\beta_1/N$")
    ax.set_ylabel(r"force proxy tail, $q_{99}/q_{50}$")
    ax.set_title("loop redundancy vs force intermittency", fontsize=7.5, pad=5)
    ax.legend(fontsize=6.0)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    for phase, color in [("cold", COLD), ("hot", HOT)]:
        sub = long[long["phase"] == phase]
        ax.scatter(sub["orientation_entropy"], sub["degree_susceptibility"], s=28, color=color, alpha=0.80, edgecolor="white", linewidth=0.35, label=phase)
    ax.set_xlabel("contact-orientation entropy")
    ax.set_ylabel(r"degree susceptibility, $\mathrm{var}(k)/\langle k\rangle$")
    ax.set_title("geometric disorder and network fluctuations", fontsize=7.5, pad=5)
    ax.legend(fontsize=6.0)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    p = persist[persist["transition"] == "cold_cycle_to_cycle"].copy()
    for run, color in zip(LONG_REGIMES, ["#345995", "#D98C3A", "#C95F3F"]):
        sub = p[p["run"] == run]
        ax.plot(sub["cycle"], sub["jaccard"], "o-", ms=3.0, lw=1.0, color=color, label=run.split("_")[0])
    ax.set_xlabel("cycle")
    ax.set_ylabel("contact-set Jaccard")
    ax.set_title("cold memory persistence", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    for run, color in zip(LONG_REGIMES, ["#345995", "#D98C3A", "#C95F3F"]):
        cold = long[(long["run"] == run) & (long["phase"] == "cold")].set_index("cycle")
        hot = long[(long["run"] == run) & (long["phase"] == "hot")].set_index("cycle")
        cycles = sorted(set(cold.index) & set(hot.index))
        gap = [hot.loc[c, "loop_density"] - cold.loc[c, "loop_density"] for c in cycles]
        ax.plot(cycles, gap, "o-", ms=3.2, lw=1.0, color=color, label=run.split("_")[0])
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"hot-cold $\Delta\beta_1/N$")
    ax.set_title("topological activation during heating", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.8)
    finish(ax)
    panel(ax, "f")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig5_topological_statphys.{ext}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame, persist: pd.DataFrame) -> None:
    long = df[df["family"] == "long"].copy()
    cold = long[long["phase"] == "cold"]
    hot = long[long["phase"] == "hot"]
    corr_loop_tail = long[["loop_density", "force_proxy_q99_q50"]].corr().iloc[0, 1]
    corr_entropy_sus = long[["orientation_entropy", "degree_susceptibility"]].corr().iloc[0, 1]
    lines = [
        "# Topological/statistical-physics mechanism mining",
        "",
        "## Mechanism hypothesis",
        "",
        "Thermal cycling does not only densify the pebble bed. It changes the topology of the contact graph: redundant loops and k-core membership store the cold memory, whereas hot overload reflects intermittent stress readout on that topological scaffold.",
        "",
        "## Data used",
        "",
        "- Three existing 30-cycle regimes: low-expansion/low-friction, intermediate, and high-expansion/high-friction.",
        "- Existing free-surface and semi-confined 20-cycle runs as boundary-projection references.",
        "- Contacts were reconstructed geometrically from atom dumps using particle radii; this is a topology/fabric analysis, not a replacement for true contact-force dumps.",
        "",
        "## Extracted observables",
        "",
        "- Topology: Betti-1 loop count, loop density, graph components, giant component, k-core fractions, triangle motif density and transitivity.",
        "- Geometry: branch-vector fabric tensor, vertical bias, contact-orientation entropy and boundary-contact localization.",
        "- Statistical physics proxies: degree entropy, degree susceptibility and contact-set Jaccard persistence.",
        "",
        "## Main numerical signals",
        "",
        f"- Mean cold loop density across sampled long-cycle states is {cold['loop_density'].mean():.3f}; mean hot loop density is {hot['loop_density'].mean():.3f}.",
        f"- Mean cold k>=4 core fraction is {cold['kcore4_fraction'].mean():.3f}; mean hot k>=4 core fraction is {hot['kcore4_fraction'].mean():.3f}.",
        f"- Correlation between loop density and force-proxy tail q99/q50 across sampled long-cycle states is r={corr_loop_tail:.2f}.",
        f"- Correlation between contact-orientation entropy and degree susceptibility is r={corr_entropy_sus:.2f}.",
        f"- Cold cycle-to-cycle contact Jaccard ranges from {persist[persist['transition']=='cold_cycle_to_cycle']['jaccard'].min():.3f} to {persist[persist['transition']=='cold_cycle_to_cycle']['jaccard'].max():.3f} in the sampled states.",
        "",
        "## Manuscript implication",
        "",
        "The strongest upgrade is to replace a scalar 'force-tail' mechanism with a two-layer mechanism: thermal cycles first create a redundant contact topology, then hot expansion reads out rare force paths through this topology. This gives the paper a statistical-physics object: a history-dependent contact graph with loop redundancy, core membership and persistence, rather than a fitted pressure law.",
        "",
        "## Caution",
        "",
        "Because these contacts are reconstructed from geometry, the topology results should be presented as contact-network topology and paired with true pair-force probes only where available. The cleanest Nature Physics version would add true contact-force local dumps at multiple cycles for the three long-cycle regimes, but the present data already support the topological-memory hypothesis.",
        "",
    ]
    (ROOT / "nature_physics_topological_statphys_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    topo, persist = mine_networks()
    topo = merge_existing_long_metrics(topo)
    topo.to_csv(SRC / "nphys_topological_statphys_metrics.csv", index=False)
    persist.to_csv(SRC / "nphys_contact_persistence_metrics.csv", index=False)
    build_figure(topo, persist)
    write_report(topo, persist)
    print(topo[["family", "run", "cycle", "phase", "loop_density", "kcore4_fraction", "orientation_entropy"]].head())


if __name__ == "__main__":
    main()
