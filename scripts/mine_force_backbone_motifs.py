#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RUN_BASE = PROJECT / "runs" / "contact_force_probe"
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

REGIME_ORDER = [
    "a050_mu010_g002",
    "a050_mu060_g002",
    "a100_mu030_g002",
    "a100_mu060_g000",
    "a150_mu010_g002",
    "a150_mu060_g020",
]
REGIME_ID = {tag: f"R{i + 1}" for i, tag in enumerate(REGIME_ORDER)}
REGIME_LABEL = {
    "a050_mu010_g002": r"$0.5,0.1$",
    "a050_mu060_g002": r"$0.5,0.6$",
    "a100_mu030_g002": r"$1.0,0.3$",
    "a100_mu060_g000": r"$1.0,0.6$",
    "a150_mu010_g002": r"$1.5,0.1$",
    "a150_mu060_g020": r"$1.5,0.6$",
}

COLD = "#345995"
HOT = "#C95F3F"
INK = "#252A31"
MUTED = "#828A93"
GRID = "#E7EAEE"
ACCENT = "#2F7F6F"


@dataclass
class ContactState:
    tag: str
    phase: str
    contacts: pd.DataFrame
    atoms: pd.DataFrame


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: plt.Axes, label: str, x: float = -0.12) -> None:
    ax.text(x, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def read_local(path: Path) -> pd.DataFrame:
    cols: list[str] | None = None
    rows: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("ITEM: ENTRIES"):
                cols = line.split()[2:]
                continue
            if line.startswith("ITEM:") or not line.strip() or cols is None:
                continue
            parts = line.split()
            if len(parts) == len(cols):
                rows.append([float(x) for x in parts])
    if not rows or cols is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    return df.rename(
        columns={
            "c_cp[1]": "id1",
            "c_cp[2]": "id2",
            "c_cp[3]": "periodic_flag",
            "c_cp[4]": "fx",
            "c_cp[5]": "fy",
            "c_cp[6]": "fz",
            "c_cp[7]": "fnx",
            "c_cp[8]": "fny",
            "c_cp[9]": "fnz",
            "c_cp[10]": "ftx",
            "c_cp[11]": "fty",
            "c_cp[12]": "ftz",
            "c_cp[13]": "cx",
            "c_cp[14]": "cy",
            "c_cp[15]": "cz",
            "c_cp[16]": "delta",
        }
    )


def read_dump(path: Path) -> pd.DataFrame:
    cols: list[str] | None = None
    rows: list[list[float]] = []
    current_rows: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                current_rows = []
                continue
            if line.startswith("ITEM:"):
                if current_rows:
                    rows = current_rows
                    current_rows = []
                continue
            if not line.strip() or cols is None:
                continue
            parts = line.split()
            if len(parts) == len(cols):
                current_rows.append([float(x) for x in parts])
    if current_rows:
        rows = current_rows
    if not rows or cols is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates("id", keep="last")
    return df.set_index("id", drop=False)


def gini(x: np.ndarray) -> float:
    y = np.sort(np.asarray(x, float))
    y = y[np.isfinite(y) & (y >= 0)]
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    n = len(y)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(y) / (n * y.sum()))


def top_share(force: np.ndarray, fraction: float) -> float:
    if len(force) == 0 or force.sum() <= 0:
        return float("nan")
    n = max(1, int(np.ceil(len(force) * fraction)))
    return float(np.sort(force)[-n:].sum() / force.sum())


class UnionFind:
    def __init__(self, nodes: np.ndarray) -> None:
        self.parent = {int(n): int(n) for n in nodes}
        self.size = {int(n): 1 for n in nodes}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def enrich_contacts(state: ContactState) -> pd.DataFrame:
    d = state.contacts.copy()
    atoms = state.atoms
    ids1 = d["id1"].astype(int).to_numpy()
    ids2 = d["id2"].astype(int).to_numpy()
    valid = np.isin(ids1, atoms.index.to_numpy()) & np.isin(ids2, atoms.index.to_numpy())
    d = d.loc[valid].copy()
    ids1 = d["id1"].astype(int).to_numpy()
    ids2 = d["id2"].astype(int).to_numpy()
    p1 = atoms.loc[ids1, ["x", "y", "z"]].to_numpy(float)
    p2 = atoms.loc[ids2, ["x", "y", "z"]].to_numpy(float)
    branch = p2 - p1
    length = np.linalg.norm(branch, axis=1)
    ok = length > 0
    d = d.loc[ok].copy()
    branch = branch[ok]
    length = length[ok]
    n = branch / length[:, None]
    d["force"] = np.linalg.norm(d[["fx", "fy", "fz"]].to_numpy(float), axis=1)
    d["branch_length"] = length
    d["nx"] = n[:, 0]
    d["ny"] = n[:, 1]
    d["nz"] = n[:, 2]
    d["abs_nz"] = np.abs(n[:, 2])
    d["theta_from_vertical_deg"] = np.degrees(np.arccos(np.clip(d["abs_nz"], 0, 1)))
    return d


def component_metrics(d: pd.DataFrame, atoms: pd.DataFrame, quantile: float) -> dict[str, float]:
    if d.empty:
        return {}
    threshold = float(np.quantile(d["force"], quantile))
    high = d[d["force"] >= threshold].copy()
    if high.empty:
        return {}
    ids1 = high["id1"].astype(int).to_numpy()
    ids2 = high["id2"].astype(int).to_numpy()
    nodes = np.unique(np.concatenate([ids1, ids2]))
    uf = UnionFind(nodes)
    for a, b in zip(ids1, ids2):
        uf.union(int(a), int(b))
    roots = np.array([uf.find(int(n)) for n in nodes])
    root_counts = pd.Series(roots).value_counts()
    largest_root = int(root_counts.index[0])
    largest_nodes = nodes[roots == largest_root]
    high_roots1 = np.array([uf.find(int(a)) for a in ids1])
    high_roots2 = np.array([uf.find(int(b)) for b in ids2])
    largest_edges = high[(high_roots1 == largest_root) & (high_roots2 == largest_root)]
    pos = atoms.loc[largest_nodes, ["x", "y", "z"]].to_numpy(float)
    all_pos = atoms[["x", "y", "z"]].to_numpy(float)
    z_span = float(pos[:, 2].max() - pos[:, 2].min()) if len(pos) else float("nan")
    bed_height = float(all_pos[:, 2].max() - all_pos[:, 2].min())
    radius = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    all_radius = np.sqrt(all_pos[:, 0] ** 2 + all_pos[:, 1] ** 2)
    side_cut = float(np.quantile(all_radius, 0.98))
    bottom_cut = float(np.quantile(all_pos[:, 2], 0.02))
    top_cut = float(np.quantile(all_pos[:, 2], 0.98))
    return {
        f"p{int(quantile * 100):02d}_threshold": threshold,
        f"p{int(quantile * 100):02d}_edge_fraction": float(len(high) / len(d)),
        f"p{int(quantile * 100):02d}_largest_component_edges": float(len(largest_edges)),
        f"p{int(quantile * 100):02d}_largest_component_nodes": float(len(largest_nodes)),
        f"p{int(quantile * 100):02d}_largest_component_edge_fraction": float(len(largest_edges) / len(high)),
        f"p{int(quantile * 100):02d}_largest_component_node_fraction": float(len(largest_nodes) / len(nodes)),
        f"p{int(quantile * 100):02d}_largest_component_z_span_m": z_span,
        f"p{int(quantile * 100):02d}_largest_component_z_span_norm": float(z_span / bed_height) if bed_height > 0 else float("nan"),
        f"p{int(quantile * 100):02d}_largest_component_touches_side": float(np.any(radius >= side_cut)),
        f"p{int(quantile * 100):02d}_largest_component_touches_bottom": float(np.any(pos[:, 2] <= bottom_cut)),
        f"p{int(quantile * 100):02d}_largest_component_touches_top": float(np.any(pos[:, 2] >= top_cut)),
    }


def fabric_metrics(d: pd.DataFrame, weight_col: str = "force") -> dict[str, float]:
    n = d[["nx", "ny", "nz"]].to_numpy(float)
    w = d[weight_col].to_numpy(float)
    if len(n) == 0 or w.sum() <= 0:
        return {}
    q = np.einsum("i,ij,ik->jk", w, n, n) / w.sum()
    eig = np.linalg.eigvalsh(q)
    return {
        "fabric_Qxx": float(q[0, 0]),
        "fabric_Qyy": float(q[1, 1]),
        "fabric_Qzz": float(q[2, 2]),
        "fabric_anisotropy": float(eig[-1] - eig[0]),
        "fabric_vertical_bias": float(q[2, 2] - 0.5 * (q[0, 0] + q[1, 1])),
        "force_weighted_abs_nz": float(np.average(d["abs_nz"], weights=w)),
    }


def load_states() -> list[ContactState]:
    states: list[ContactState] = []
    for tag in REGIME_ORDER:
        run = RUN_BASE / tag
        for phase in ["cold", "hot"]:
            local = run / f"contacts_cycle_10_{phase}.local"
            dump = run / f"cycle_10_{phase}.dump"
            if not local.exists() or not dump.exists():
                continue
            states.append(ContactState(tag, phase, read_local(local), read_dump(dump)))
    return states


def compute_tables(states: list[ContactState]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_rows: list[dict[str, float | str]] = []
    hist_rows: list[dict[str, float | str]] = []
    enriched: dict[str, pd.DataFrame] = {}
    bins = np.arange(0, 100, 10)
    for state in states:
        d = enrich_contacts(state)
        key = f"{state.tag}_{state.phase}"
        enriched[key] = d
        force = d["force"].to_numpy(float)
        row: dict[str, float | str] = {
            "tag": state.tag,
            "regime_id": REGIME_ID[state.tag],
            "phase": state.phase,
            "contacts": float(len(d)),
            "force_mean": float(force.mean()),
            "force_p95": float(np.percentile(force, 95)),
            "force_p99": float(np.percentile(force, 99)),
            "force_max": float(force.max()),
            "force_gini": gini(force),
            "top1_force_share": top_share(force, 0.01),
            "top5_force_share": top_share(force, 0.05),
            "mean_overlap_m": float(d["delta"].mean()),
            "force_weighted_theta_deg": float(np.average(d["theta_from_vertical_deg"], weights=force)),
        }
        row.update(fabric_metrics(d))
        row.update(component_metrics(d, state.atoms, 0.95))
        row.update(component_metrics(d, state.atoms, 0.99))
        metric_rows.append(row)

        binned = pd.cut(d["theta_from_vertical_deg"], bins=bins, include_lowest=True, right=False)
        h_count = d.groupby(binned, observed=False)["force"].count()
        h_force = d.groupby(binned, observed=False)["force"].sum()
        for interval in h_count.index:
            count = float(h_count.loc[interval])
            fsum = float(h_force.loc[interval])
            hist_rows.append(
                {
                    "tag": state.tag,
                    "regime_id": REGIME_ID[state.tag],
                    "phase": state.phase,
                    "theta_bin_left_deg": float(interval.left),
                    "theta_bin_right_deg": float(interval.right),
                    "count_fraction": count / len(d) if len(d) else np.nan,
                    "force_fraction": fsum / force.sum() if force.sum() > 0 else np.nan,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    hist = pd.DataFrame(hist_rows)
    return metrics, hist, enriched


def paired_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    cold = metrics[metrics["phase"] == "cold"].set_index("tag")
    hot = metrics[metrics["phase"] == "hot"].set_index("tag")
    cols = [
        "top1_force_share",
        "top5_force_share",
        "force_gini",
        "fabric_anisotropy",
        "fabric_vertical_bias",
        "force_weighted_abs_nz",
        "p95_largest_component_z_span_norm",
        "p99_largest_component_z_span_norm",
    ]
    rows = []
    for tag in REGIME_ORDER:
        if tag not in cold.index or tag not in hot.index:
            continue
        row = {"tag": tag, "regime_id": REGIME_ID[tag]}
        for col in cols:
            row[f"{col}_cold"] = cold.loc[tag, col]
            row[f"{col}_hot"] = hot.loc[tag, col]
            row[f"{col}_hot_minus_cold"] = hot.loc[tag, col] - cold.loc[tag, col]
        rows.append(row)
    return pd.DataFrame(rows)


def draw_backbone(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    delta: pd.DataFrame,
    enriched: dict[str, pd.DataFrame],
    states: list[ContactState],
) -> str:
    tag = str(delta.sort_values("p95_largest_component_z_span_norm_hot_minus_cold", ascending=False).iloc[0]["tag"])
    state = next(s for s in states if s.tag == tag and s.phase == "hot")
    d = enriched[f"{tag}_hot"].copy()
    threshold = np.quantile(d["force"], 0.95)
    high = d[d["force"] >= threshold].copy()
    atoms = state.atoms
    segs = []
    vals = []
    for _, row in high.iterrows():
        p1 = atoms.loc[int(row["id1"]), ["x", "z"]].to_numpy(float) * 1000
        p2 = atoms.loc[int(row["id2"]), ["x", "z"]].to_numpy(float) * 1000
        segs.append([p1, p2])
        vals.append(row["force"])
    force = np.asarray(vals)
    widths = 0.25 + 1.15 * (force - force.min()) / (force.max() - force.min() + 1e-30)
    lc = LineCollection(segs, cmap="magma_r", linewidths=widths, alpha=0.58)
    lc.set_array(np.log10(np.asarray(vals)))
    ax.add_collection(lc)
    sample = atoms.sample(min(2600, len(atoms)), random_state=9)
    ax.scatter(sample["x"] * 1000, sample["z"] * 1000, s=0.65, color="#D5D9DE", alpha=0.18, linewidths=0)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"{REGIME_ID[tag]} hot, top 5% force backbone", fontsize=7.5, pad=5)
    panel(ax, "a", x=-0.10)
    cbar = plt.colorbar(lc, ax=ax, fraction=0.040, pad=0.015)
    cbar.set_label(r"$\log_{10}|f|$", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8, width=0.55)
    return tag


def draw_paired(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    col: str,
    ylabel: str,
    label: str,
    ylim: tuple[float, float] | None = None,
    show_legend: bool = False,
) -> None:
    cold = metrics[metrics["phase"] == "cold"].set_index("tag")
    hot = metrics[metrics["phase"] == "hot"].set_index("tag")
    x = np.arange(len(REGIME_ORDER))
    for i, tag in enumerate(REGIME_ORDER):
        if tag not in cold.index or tag not in hot.index:
            continue
        ax.plot([i - 0.12, i + 0.12], [cold.loc[tag, col], hot.loc[tag, col]], color="#B8BEC6", lw=0.9, zorder=1)
        ax.scatter(i - 0.12, cold.loc[tag, col], s=25, color=COLD, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(i + 0.12, hot.loc[tag, col], s=27, color=HOT, edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks(x, [REGIME_ID[t] for t in REGIME_ORDER])
    ax.set_xlabel("regime")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(label, fontsize=7.5, pad=5)
    if show_legend:
        ax.scatter([], [], s=25, color=COLD, edgecolor="white", linewidth=0.5, label="cold")
        ax.scatter([], [], s=27, color=HOT, edgecolor="white", linewidth=0.5, label="hot")
        ax.legend(loc="upper left", fontsize=6.1, handletextpad=0.35, borderaxespad=0.2)
    finish(ax, "y")


def draw_orientation_hist(ax: plt.Axes, hist: pd.DataFrame, tag: str) -> None:
    sub = hist[hist["tag"] == tag].copy()
    for phase, color in [("cold", COLD), ("hot", HOT)]:
        d = sub[sub["phase"] == phase]
        x = 0.5 * (d["theta_bin_left_deg"].to_numpy(float) + d["theta_bin_right_deg"].to_numpy(float))
        y = d["force_fraction"].to_numpy(float)
        ax.plot(x, y, color=color, lw=1.3, label=phase)
        ax.fill_between(x, 0, y, color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel(r"contact angle from vertical, $\theta$ (deg)")
    ax.set_ylabel("force-weighted fraction")
    ax.set_xlim(0, 90)
    ax.set_title(f"{REGIME_ID[tag]} orientation spectrum", fontsize=7.5, pad=5)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2)
    finish(ax, "y")


def build_figure(
    metrics: pd.DataFrame,
    hist: pd.DataFrame,
    delta: pd.DataFrame,
    enriched: dict[str, pd.DataFrame],
    states: list[ContactState],
) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.15, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 1, 1], height_ratios=[1, 1])
    ax_backbone = fig.add_subplot(gs[:, 0])
    tag = draw_backbone(ax_backbone, metrics, delta, enriched, states)

    ax_top = fig.add_subplot(gs[0, 1])
    draw_paired(ax_top, metrics, "top1_force_share", "share of total force", "load carried by top 1%", show_legend=True)
    panel(ax_top, "b")
    ax_aniso = fig.add_subplot(gs[0, 2])
    draw_paired(ax_aniso, metrics, "fabric_anisotropy", r"$\lambda_{\max}-\lambda_{\min}$", "force-weighted fabric anisotropy")
    panel(ax_aniso, "c")
    ax_span = fig.add_subplot(gs[1, 1])
    draw_paired(ax_span, metrics, "p95_largest_component_z_span_norm", "normalized z-span", "largest top-5% component")
    panel(ax_span, "d")
    ax_hist = fig.add_subplot(gs[1, 2])
    draw_orientation_hist(ax_hist, hist, tag)
    panel(ax_hist, "e")

    for ext in ["svg", "pdf", "png", "tiff"]:
        out = FIG / f"nphys_fig4_force_backbone.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out, **kwargs)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, delta: pd.DataFrame) -> None:
    hot = metrics[metrics["phase"] == "hot"].copy()
    cold = metrics[metrics["phase"] == "cold"].copy()
    strongest = hot.sort_values("top1_force_share", ascending=False).iloc[0]
    aniso_delta = delta.sort_values("fabric_anisotropy_hot_minus_cold", ascending=False).iloc[0]
    span_delta = delta.sort_values("p95_largest_component_z_span_norm_hot_minus_cold", ascending=False).iloc[0]
    lines = [
        "# Force-backbone motif mining report",
        "",
        "This analysis follows the motif-oriented logic suggested by the arXiv granular quasi-crystal paper, but maps it to the present DEM observables: true pair contact forces, branch-vector orientations and high-force contact-network components.",
        "",
        "## What was mined",
        "",
        "- Source files: cycle-10 hot/cold `compute pair/gran/local id force force_normal force_tangential contactPoint delta` outputs and matched atom dumps.",
        "- Derived observables: top-1% and top-5% force shares, Gini coefficient, force-weighted contact fabric tensor, vertical orientation bias, and largest high-force connected component span.",
        "- Outputs: `source_data/nphys_force_backbone_metrics.csv`, `source_data/nphys_force_orientation_histogram.csv`, `source_data/nphys_force_backbone_hot_cold_delta.csv`, and `figures/nphys_fig4_force_backbone.*`.",
        "",
        "## Main signals",
        "",
        f"- The strongest hot state by top-1% force concentration is {strongest['regime_id']} ({strongest['tag']}), where the top 1% contacts carry {100 * strongest['top1_force_share']:.1f}% of total contact force.",
        f"- The largest hot-minus-cold increase in force-weighted fabric anisotropy occurs in {aniso_delta['regime_id']} ({aniso_delta['tag']}), with Δanisotropy = {aniso_delta['fabric_anisotropy_hot_minus_cold']:.3f}.",
        f"- The largest hot-minus-cold increase in largest top-5% backbone vertical span occurs in {span_delta['regime_id']} ({span_delta['tag']}), with Δnormalized span = {span_delta['p95_largest_component_z_span_norm_hot_minus_cold']:.3f}.",
        f"- Across the six regimes, mean hot top-1% force share is {100 * hot['top1_force_share'].mean():.1f}% versus {100 * cold['top1_force_share'].mean():.1f}% in cold states.",
        f"- Mean force-weighted fabric anisotropy is {hot['fabric_anisotropy'].mean():.3f} in hot states versus {cold['fabric_anisotropy'].mean():.3f} in cold states.",
        "",
        "## Interpretation for the manuscript",
        "",
        "This gives a stronger mesoscopic bridge between the macroscopic two-channel regime map and the microscopic contact-force dump: thermal cycling can now be described as selecting a sparse force-bearing backbone with a measurable fabric tensor, not only as changing a scalar stress-tail metric. The figure is promising as an Extended Data figure immediately, and as a main-text Fig. 4 if the next remote run confirms the same trends across seeds and earlier/later cycles.",
        "",
        "## Review risk",
        "",
        "The present mining uses one contact-force snapshot per regime and phase, so it should not yet be overclaimed as a universal distribution. For Nature Physics level, the clean upgrade is to repeat the same backbone metrics over all three seeds and several cycles, then show that the hot/cold backbone bifurcation survives preparation noise.",
        "",
    ]
    (ROOT / "nature_physics_force_backbone_mining_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    states = load_states()
    if not states:
        raise SystemExit(f"No contact-force states found under {RUN_BASE}")
    metrics, hist, enriched = compute_tables(states)
    delta = paired_delta(metrics)
    metrics.to_csv(SRC / "nphys_force_backbone_metrics.csv", index=False)
    hist.to_csv(SRC / "nphys_force_orientation_histogram.csv", index=False)
    delta.to_csv(SRC / "nphys_force_backbone_hot_cold_delta.csv", index=False)
    build_figure(metrics, hist, delta, enriched, states)
    write_report(metrics, delta)
    print(f"states={len(states)}")
    print(metrics[["regime_id", "tag", "phase", "top1_force_share", "fabric_anisotropy", "p95_largest_component_z_span_norm"]])


if __name__ == "__main__":
    main()
