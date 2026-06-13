#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


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

COLD = "#345995"
HOT = "#C95F3F"
R1 = "#345995"
R3 = "#D98C3A"
R6 = "#C95F3F"
GRID = "#E7EAEE"
INK = "#252A31"


@dataclass
class ContactState:
    run: str
    cycle: int
    phase: str
    ids: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    pairs: np.ndarray
    overlap: np.ndarray
    unit: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    side: np.ndarray


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.bottom = np.zeros(n, dtype=bool)
        self.top = np.zeros(n, dtype=bool)
        self.side = np.zeros(n, dtype=bool)

    def set_flags(self, bottom: np.ndarray, top: np.ndarray, side: np.ndarray) -> None:
        self.bottom[:] = bottom
        self.top[:] = top
        self.side[:] = side

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.bottom[ra] = self.bottom[ra] or self.bottom[rb]
        self.top[ra] = self.top[ra] or self.top[rb]
        self.side[ra] = self.side[ra] or self.side[rb]
        return ra


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


def build_contact_state(run: str, cycle: int, phase: str, path: Path, skin: float = 1.004) -> ContactState:
    df = read_dump(path)
    xyz = df[["x", "y", "z"]].to_numpy(float)
    radius = df["radius"].to_numpy(float)
    ids = df["id"].to_numpy(int)
    tree = cKDTree(xyz)
    candidate = np.array(list(tree.query_pairs(float(2 * radius.max() * skin))), dtype=np.int64)
    if candidate.size == 0:
        candidate = candidate.reshape(0, 2)
        unit = np.empty((0, 3))
        overlap = np.empty(0)
    else:
        delta = xyz[candidate[:, 1]] - xyz[candidate[:, 0]]
        dist = np.linalg.norm(delta, axis=1)
        raw_overlap = radius[candidate[:, 0]] + radius[candidate[:, 1]] - dist
        keep = raw_overlap >= -float((skin - 1.0) * 2 * radius.max())
        candidate = candidate[keep]
        delta = delta[keep]
        dist = dist[keep]
        raw_overlap = raw_overlap[keep]
        ok = dist > 0
        candidate = candidate[ok]
        unit = delta[ok] / dist[ok, None]
        overlap = raw_overlap[ok]
    z = xyz[:, 2]
    rxy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    return ContactState(
        run=run,
        cycle=cycle,
        phase=phase,
        ids=ids,
        xyz=xyz,
        radius=radius,
        pairs=candidate,
        overlap=overlap,
        unit=unit,
        bottom=z <= np.quantile(z, 0.03),
        top=z >= np.quantile(z, 0.97),
        side=rxy >= np.quantile(rxy, 0.97),
    )


def edge_set(state: ContactState) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i, j in state.pairs:
        a, b = int(state.ids[i]), int(state.ids[j])
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def graph_h1_births(state: ContactState) -> dict[str, float]:
    n = len(state.ids)
    if len(state.pairs) == 0:
        return {"cycle_birth_count": 0.0}
    order = np.argsort(state.overlap)[::-1]
    uf = UnionFind(n)
    births = []
    for idx in order:
        i, j = map(int, state.pairs[idx])
        oi = float(max(0.0, state.overlap[idx]))
        if uf.find(i) == uf.find(j):
            births.append(oi)
        else:
            uf.union(i, j)
    arr = np.asarray(births, float)
    positive = arr[arr > 0]
    return {
        "cycle_birth_count": float(len(arr)),
        "cycle_birth_positive_count": float(len(positive)),
        "cycle_birth_overlap_median": float(np.median(arr)) if len(arr) else np.nan,
        "cycle_birth_overlap_p90": float(np.percentile(arr, 90)) if len(arr) else np.nan,
        "cycle_birth_positive_fraction": float(len(positive) / len(arr)) if len(arr) else np.nan,
    }


def percolation_metrics(state: ContactState) -> dict[str, float]:
    n_edges = len(state.pairs)
    if n_edges == 0:
        return {}
    order = np.argsort(state.overlap)[::-1]
    uf = UnionFind(len(state.ids))
    uf.set_flags(state.bottom.copy(), state.top.copy(), state.side.copy())
    bottom_top_at = None
    bottom_side_at = None
    giant_at_10 = None
    giant_at_05 = None
    cycle_births = 0
    for rank, edge_idx in enumerate(order, start=1):
        i, j = map(int, state.pairs[edge_idx])
        if uf.find(i) == uf.find(j):
            cycle_births += 1
            root = uf.find(i)
        else:
            root = uf.union(i, j)
        roots = [uf.find(int(x)) for x in (i, j)]
        for root in roots:
            frac = rank / n_edges
            if bottom_top_at is None and uf.bottom[root] and uf.top[root]:
                bottom_top_at = frac
            if bottom_side_at is None and uf.bottom[root] and uf.side[root]:
                bottom_side_at = frac
        if giant_at_05 is None and rank >= int(0.05 * n_edges):
            giant_at_05 = float(uf.size[[uf.find(i) for i in range(len(state.ids))]].max() / len(state.ids))
        if giant_at_10 is None and rank >= int(0.10 * n_edges):
            giant_at_10 = float(uf.size[[uf.find(i) for i in range(len(state.ids))]].max() / len(state.ids))
    return {
        "bottom_top_percolation_edge_fraction": float(bottom_top_at) if bottom_top_at is not None else np.nan,
        "bottom_side_percolation_edge_fraction": float(bottom_side_at) if bottom_side_at is not None else np.nan,
        "giant_fraction_after_top5_edges": giant_at_05,
        "giant_fraction_after_top10_edges": giant_at_10,
        "cycle_births_during_overlap_filtration": float(cycle_births),
    }


def orientation_entropy(state: ContactState) -> float:
    if len(state.unit) == 0:
        return np.nan
    theta = np.degrees(np.arccos(np.clip(np.abs(state.unit[:, 2]), 0, 1)))
    hist, _ = np.histogram(theta, bins=np.arange(0, 100, 10))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum() / np.log(9))


def state_row(state: ContactState) -> dict[str, float | str | int]:
    overlap_pos = np.maximum(0.0, state.overlap)
    row: dict[str, float | str | int] = {
        "run": state.run,
        "cycle": state.cycle,
        "phase": state.phase,
        "contacts": float(len(state.pairs)),
        "Z_geom": float(2 * len(state.pairs) / len(state.ids)),
        "mean_positive_overlap_m": float(overlap_pos[overlap_pos > 0].mean()) if np.any(overlap_pos > 0) else 0.0,
        "overlap_p90_m": float(np.percentile(overlap_pos, 90)) if len(overlap_pos) else np.nan,
        "overlap_p99_m": float(np.percentile(overlap_pos, 99)) if len(overlap_pos) else np.nan,
        "orientation_entropy": orientation_entropy(state),
    }
    row.update(percolation_metrics(state))
    row.update(graph_h1_births(state))
    return row


def load_all_states() -> list[ContactState]:
    states = []
    for run, folder in LONG_REGIMES.items():
        for cycle in range(1, 31):
            for phase in ["cold", "hot"]:
                path = folder / f"cycle_{cycle}_{phase}.dump"
                if path.exists():
                    states.append(build_contact_state(run, cycle, phase, path))
    return states


def persistence_decomposition(states: list[ContactState]) -> pd.DataFrame:
    by_key = {(s.run, s.cycle, s.phase): edge_set(s) for s in states}
    rows = []
    for run in LONG_REGIMES:
        for phase in ["cold", "hot"]:
            for cycle in range(2, 31):
                prev = by_key.get((run, cycle - 1, phase), set())
                cur = by_key.get((run, cycle, phase), set())
                if not prev or not cur:
                    continue
                persistent = prev & cur
                created = cur - prev
                broken = prev - cur
                rows.append(
                    {
                        "run": run,
                        "cycle": cycle,
                        "phase": phase,
                        "persistent_edges": len(persistent),
                        "created_edges": len(created),
                        "broken_edges": len(broken),
                        "jaccard": len(persistent) / len(prev | cur),
                        "persistent_fraction_of_current": len(persistent) / len(cur),
                        "created_fraction_of_current": len(created) / len(cur),
                        "broken_fraction_of_previous": len(broken) / len(prev),
                    }
                )
        for cycle in range(1, 31):
            cold = by_key.get((run, cycle, "cold"), set())
            hot = by_key.get((run, cycle, "hot"), set())
            if not cold or not hot:
                continue
            rows.append(
                {
                    "run": run,
                    "cycle": cycle,
                    "phase": "cold_to_hot",
                    "persistent_edges": len(cold & hot),
                    "created_edges": len(hot - cold),
                    "broken_edges": len(cold - hot),
                    "jaccard": len(cold & hot) / len(cold | hot),
                    "persistent_fraction_of_current": len(cold & hot) / len(hot),
                    "created_fraction_of_current": len(hot - cold) / len(hot),
                    "broken_fraction_of_previous": len(cold - hot) / len(cold),
                }
            )
    return pd.DataFrame(rows)


def merge_force_proxy(metrics: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for run, folder in LONG_REGIMES.items():
        path = folder / "advanced_microphysics.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["run"] = run
        d["phase"] = d["phase"].str.lower()
        frames.append(d[["run", "cycle", "phase", "force_proxy_q99_q50", "force_proxy_gini", "net_force_p99_N"]])
    if not frames:
        return metrics
    return metrics.merge(pd.concat(frames, ignore_index=True), on=["run", "cycle", "phase"], how="left")


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 7) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 6:
        return np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return rho, p


def prediction_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in LONG_REGIMES:
        cold = metrics[(metrics["run"] == run) & (metrics["phase"] == "cold")].set_index("cycle")
        hot = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")].set_index("cycle")
        for lag in [0, 1]:
            pairs = []
            for c in cold.index:
                target_cycle = c + lag
                if target_cycle in hot.index:
                    pairs.append((c, target_cycle))
            if len(pairs) < 6:
                continue
            for xcol in ["bottom_side_percolation_edge_fraction", "cycle_birth_positive_fraction", "orientation_entropy", "Z_geom"]:
                x = np.array([cold.loc[c, xcol] for c, _ in pairs], float)
                y = np.array([hot.loc[h, "force_proxy_q99_q50"] for _, h in pairs], float)
                rho, p = permutation_spearman(x, y)
                rows.append(
                    {
                        "run": run,
                        "lag_cycles": lag,
                        "predictor": f"cold_{xcol}",
                        "target": "hot_force_proxy_q99_q50",
                        "n_pairs": len(pairs),
                        "spearman_rho": rho,
                        "permutation_p_two_sided": p,
                    }
                )
    pooled = []
    for run in LONG_REGIMES:
        cold = metrics[(metrics["run"] == run) & (metrics["phase"] == "cold")].set_index("cycle")
        hot = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")].set_index("cycle")
        for c in cold.index:
            if c in hot.index:
                rec = {"run": run, "cycle": c}
                for xcol in ["bottom_side_percolation_edge_fraction", "cycle_birth_positive_fraction", "orientation_entropy", "Z_geom"]:
                    rec[xcol] = cold.loc[c, xcol]
                rec["target"] = hot.loc[c, "force_proxy_q99_q50"]
                pooled.append(rec)
    pooled_df = pd.DataFrame(pooled)
    for xcol in ["bottom_side_percolation_edge_fraction", "cycle_birth_positive_fraction", "orientation_entropy", "Z_geom"]:
        x = pooled_df[xcol].to_numpy(float)
        y = pooled_df["target"].to_numpy(float)
        rho, p = permutation_spearman(x, y)
        rows.append(
            {
                "run": "pooled_three_regimes",
                "lag_cycles": 0,
                "predictor": f"cold_{xcol}",
                "target": "hot_force_proxy_q99_q50",
                "n_pairs": len(pooled_df),
                "spearman_rho": rho,
                "permutation_p_two_sided": p,
            }
        )
    return pd.DataFrame(rows)


def build_figure(metrics: pd.DataFrame, persistence: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.15, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    colors = {
        "R1_low_expansion_low_friction": R1,
        "R3_intermediate": R3,
        "R6_high_expansion_high_friction": R6,
    }
    labels = {
        "R1_low_expansion_low_friction": "R1",
        "R3_intermediate": "R3",
        "R6_high_expansion_high_friction": "R6",
    }

    ax = fig.add_subplot(gs[0, 0])
    for run, color in colors.items():
        sub = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")]
        ax.plot(sub["cycle"], sub["bottom_side_percolation_edge_fraction"], "o-", ms=2.8, lw=1.0, color=color, label=labels[run])
    ax.set_xlabel("cycle")
    ax.set_ylabel("edge fraction at bottom-side percolation")
    ax.set_title("overlap-threshold percolation", fontsize=7.5, pad=5)
    ax.set_yscale("log")
    ax.legend(fontsize=5.8)
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for run, color in colors.items():
        cold = metrics[(metrics["run"] == run) & (metrics["phase"] == "cold")].set_index("cycle")
        hot = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")].set_index("cycle")
        cycles = sorted(set(cold.index) & set(hot.index))
        gap = [hot.loc[c, "cycle_birth_positive_fraction"] - cold.loc[c, "cycle_birth_positive_fraction"] for c in cycles]
        ax.plot(cycles, gap, "o-", ms=2.8, lw=1.0, color=color, label=labels[run])
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel("hot-cold positive H1 birth fraction")
    ax.set_title("graph-filtration cycle activation", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for run, color in colors.items():
        sub = persistence[(persistence["run"] == run) & (persistence["phase"] == "cold")]
        ax.plot(sub["cycle"], sub["created_fraction_of_current"], "o-", ms=2.8, lw=1.0, color=color, label=labels[run])
    ax.set_xlabel("cycle")
    ax.set_ylabel("new cold-edge fraction")
    ax.set_title("cold contact rewriting", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    for run, color in colors.items():
        cold = metrics[(metrics["run"] == run) & (metrics["phase"] == "cold")].set_index("cycle")
        hot = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")].set_index("cycle")
        common = sorted(set(cold.index) & set(hot.index))
        x = np.array([cold.loc[c, "bottom_side_percolation_edge_fraction"] for c in common])
        y = np.array([hot.loc[c, "force_proxy_q99_q50"] for c in common])
        ax.scatter(x, y, s=24, color=color, edgecolor="white", linewidth=0.35, alpha=0.82, label=labels[run])
    ax.set_xlabel("cold percolation edge fraction")
    ax.set_ylabel(r"hot force proxy tail, $q_{99}/q_{50}$")
    ax.set_title("cold topology vs hot tail", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.8)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    pooled = tests[tests["run"] == "pooled_three_regimes"].copy()
    names = {
        "cold_bottom_side_percolation_edge_fraction": "percolation",
        "cold_cycle_birth_positive_fraction": "H1 births",
        "cold_orientation_entropy": "orient.",
        "cold_Z_geom": "Z",
    }
    x = np.arange(len(pooled))
    ax.bar(x, pooled["spearman_rho"], color=[R1, R3, R6, "#7F5AA2"], width=0.65)
    for i, row in enumerate(pooled.itertuples()):
        ax.text(i, row.spearman_rho + (0.04 if row.spearman_rho >= 0 else -0.08), f"p={row.permutation_p_two_sided:.2f}", ha="center", va="bottom" if row.spearman_rho >= 0 else "top", fontsize=5.7, color=INK)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xticks(x, [names[n] for n in pooled["predictor"]], rotation=0, ha="center")
    ax.set_ylabel("pooled Spearman rho")
    ax.set_title("exploratory prediction tests", fontsize=7.5, pad=8)
    finish(ax, "y")
    panel(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    summary = []
    for run in LONG_REGIMES:
        psub = persistence[(persistence["run"] == run) & (persistence["phase"] == "cold")]
        hot = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")].set_index("cycle")
        summary.append(
            {
                "run": labels[run],
                "cold_rewrite": psub["created_fraction_of_current"].mean(),
                "hot_tail_end": hot.loc[30, "force_proxy_q99_q50"],
            }
        )
    s = pd.DataFrame(summary)
    ax.scatter(s["cold_rewrite"], s["hot_tail_end"], s=55, color=[R1, R3, R6], edgecolor="white", linewidth=0.6)
    for row in s.itertuples():
        ax.text(row.cold_rewrite + 0.008, row.hot_tail_end, row.run, fontsize=7.0, va="center")
    ax.set_xlabel("mean new cold-edge fraction")
    ax.set_ylabel(r"cycle-30 hot $q_{99}/q_{50}$")
    ax.set_title("rewriting route separates regimes", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "f")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig6_deep_topology_percolation.{ext}", **kwargs)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, persistence: pd.DataFrame, tests: pd.DataFrame) -> None:
    end_rows = []
    for run in LONG_REGIMES:
        cold = metrics[(metrics["run"] == run) & (metrics["phase"] == "cold")].set_index("cycle")
        hot = metrics[(metrics["run"] == run) & (metrics["phase"] == "hot")].set_index("cycle")
        psub = persistence[(persistence["run"] == run) & (persistence["phase"] == "cold")]
        end_rows.append(
            {
                "run": run,
                "cold_perc_end": cold.loc[30, "bottom_side_percolation_edge_fraction"],
                "hot_perc_end": hot.loc[30, "bottom_side_percolation_edge_fraction"],
                "hot_force_tail_end": hot.loc[30, "force_proxy_q99_q50"],
                "mean_cold_rewrite": psub["created_fraction_of_current"].mean(),
                "min_cold_jaccard": psub["jaccard"].min(),
                "hot_h1_birth_positive_end": hot.loc[30, "cycle_birth_positive_fraction"],
            }
        )
    end = pd.DataFrame(end_rows)
    pooled = tests[tests["run"] == "pooled_three_regimes"].copy()
    lines = [
        "# Deep topology/percolation mining report",
        "",
        "## Strict scope",
        "",
        "This report does not claim causality. It uses existing atom dumps to reconstruct geometric contacts and uses geometric overlap as a filtration/percolation weight. These quantities are valid topology and geometry proxies, but they are not true pair-force measurements.",
        "",
        "## Analyses actually performed",
        "",
        "- Overlap-threshold percolation: contacts are added from largest to smallest overlap; the reported threshold is the edge fraction at which a bottom-side path first appears.",
        "- Graph-filtration H1 births: when an overlap-ordered edge closes a graph cycle, its overlap value is recorded as a cycle birth. Because no 2-simplices are filled, this is a graph-cycle birth statistic, not full persistent homology.",
        "- Edge persistence decomposition: cold-to-cold and hot-to-hot edges are decomposed into persistent, created and broken fractions.",
        "- Exploratory prediction tests: cold-state topology proxies are compared with same-cycle or next-cycle hot force-proxy tails using Spearman rho and permutation p values.",
        "",
        "## Regime-level results",
        "",
    ]
    for row in end.itertuples():
        lines.append(
            f"- {row.run}: cycle-30 hot force-tail q99/q50 = {row.hot_force_tail_end:.2f}; "
            f"mean new cold-edge fraction = {row.mean_cold_rewrite:.2f}; "
            f"minimum cold Jaccard = {row.min_cold_jaccard:.2f}; "
            f"cycle-30 hot positive graph-cycle birth fraction = {row.hot_h1_birth_positive_end:.2f}."
        )
    lines.extend(
        [
            "",
            "## Exploratory predictive tests",
            "",
        ]
    )
    for row in pooled.itertuples():
        lines.append(
            f"- {row.predictor} vs {row.target}: Spearman rho = {row.spearman_rho:.2f}, permutation p = {row.permutation_p_two_sided:.3f}, n = {row.n_pairs}."
        )
    lines.extend(
        [
            "",
            "## What is supported",
            "",
            "The data support a topology-aware mechanism hypothesis: regimes differ not only in coordination or force-tail magnitude, but in how cold contacts are rewritten and how overlap-ordered contact graphs percolate during heating. The intermediate regime combines substantial cold-edge rewriting with the largest late-cycle hot force tail, which is consistent with a transient activation route rather than a stable-memory route.",
            "",
            "## What is not supported yet",
            "",
            "The present data do not prove information transfer or causality from topology to hot overload. The prediction tests are small-sample diagnostics. Full persistent homology and true force-weighted percolation require pair-force local dumps across multiple cycles.",
            "",
            "## Next concrete validation",
            "",
            "Run true pair-force local probes for cycles 1, 5, 10, 20 and 30 in the three long-cycle regimes, then repeat the percolation analysis using force rather than overlap as edge weight. That would test whether geometric topology is merely correlated with, or actually organizes, the stress-bearing subnetwork.",
            "",
        ]
    )
    (ROOT / "nature_physics_deep_topology_percolation_report.md").write_text("\n".join(lines), encoding="utf-8")
    end.to_csv(SRC / "nphys_deep_topology_regime_summary.csv", index=False)


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    states = load_all_states()
    rows = []
    for state in states:
        row = state_row(state)
        rows.append(row)
        print(f"{state.run} cycle={state.cycle:02d} {state.phase} contacts={len(state.pairs)}")
    metrics = pd.DataFrame(rows)
    metrics = merge_force_proxy(metrics)
    persistence = persistence_decomposition(states)
    tests = prediction_tests(metrics)
    metrics.to_csv(SRC / "nphys_deep_topology_percolation_metrics.csv", index=False)
    persistence.to_csv(SRC / "nphys_deep_edge_persistence_decomposition.csv", index=False)
    tests.to_csv(SRC / "nphys_deep_prediction_tests.csv", index=False)
    build_figure(metrics, persistence, tests)
    write_report(metrics, persistence, tests)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
