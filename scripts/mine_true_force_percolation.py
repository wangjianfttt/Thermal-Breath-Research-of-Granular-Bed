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
GRID = "#E7EAEE"
ACCENT = "#2F7F6F"


@dataclass
class ForceState:
    tag: str
    phase: str
    contacts: pd.DataFrame
    atoms: pd.DataFrame


class UnionFind:
    def __init__(self, n: int, bottom: np.ndarray, top: np.ndarray, side: np.ndarray) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.bottom = bottom.copy()
        self.top = top.copy()
        self.side = side.copy()
        self.max_size = 1

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> tuple[int, bool]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.max_size = max(self.max_size, int(self.size[ra]))
        self.bottom[ra] = self.bottom[ra] or self.bottom[rb]
        self.top[ra] = self.top[ra] or self.top[rb]
        self.side[ra] = self.side[ra] or self.side[rb]
        return ra, True


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


def read_local(path: Path) -> pd.DataFrame:
    cols: list[str] | None = None
    rows: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("ITEM: ENTRIES"):
                cols = line.split()[2:]
                continue
            if line.startswith("ITEM:") or cols is None or not line.strip():
                continue
            parts = line.split()
            if len(parts) == len(cols):
                rows.append([float(x) for x in parts])
    if not rows or cols is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df = df.rename(
        columns={
            "c_cp[1]": "id1",
            "c_cp[2]": "id2",
            "c_cp[3]": "periodic_flag",
            "c_cp[4]": "fx",
            "c_cp[5]": "fy",
            "c_cp[6]": "fz",
            "c_cp[13]": "cx",
            "c_cp[14]": "cy",
            "c_cp[15]": "cz",
            "c_cp[16]": "delta",
        }
    )
    df["id1"] = df["id1"].astype(int)
    df["id2"] = df["id2"].astype(int)
    df["force"] = np.linalg.norm(df[["fx", "fy", "fz"]].to_numpy(float), axis=1)
    return df


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
    return df.drop_duplicates("id", keep="last").set_index("id", drop=False)


def load_states() -> list[ForceState]:
    states: list[ForceState] = []
    for tag in REGIME_ORDER:
        folder = RUN_BASE / tag
        for phase in ["cold", "hot"]:
            local = folder / f"contacts_cycle_10_{phase}.local"
            dump = folder / f"cycle_10_{phase}.dump"
            if local.exists() and dump.exists():
                states.append(ForceState(tag, phase, read_local(local), read_dump(dump)))
    return states


def enrich_contacts(state: ForceState) -> tuple[pd.DataFrame, dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    atoms = state.atoms
    d = state.contacts.copy()
    valid = d["id1"].isin(atoms.index) & d["id2"].isin(atoms.index) & (d["force"] > 0)
    d = d.loc[valid].copy()
    ids = np.array(sorted(set(d["id1"]).union(set(d["id2"]))))
    id_to_idx = {int(v): i for i, v in enumerate(ids)}
    z = atoms.loc[ids, "z"].to_numpy(float)
    rxy = np.sqrt(atoms.loc[ids, "x"].to_numpy(float) ** 2 + atoms.loc[ids, "y"].to_numpy(float) ** 2)
    bottom = z <= np.quantile(z, 0.03)
    top = z >= np.quantile(z, 0.97)
    side = rxy >= np.quantile(rxy, 0.97)
    return d, id_to_idx, bottom, top, side


def force_percolation(state: ForceState) -> dict[str, float | str]:
    d, id_to_idx, bottom, top, side = enrich_contacts(state)
    if d.empty:
        return {"tag": state.tag, "regime_id": REGIME_ID[state.tag], "phase": state.phase}
    n_nodes = len(id_to_idx)
    order = np.argsort(d["force"].to_numpy(float))[::-1]
    forces = d["force"].to_numpy(float)
    total_force = float(forces.sum())
    uf = UnionFind(n_nodes, bottom, top, side)
    bottom_top_at = np.nan
    bottom_side_at = np.nan
    h1_births = 0
    h1_birth_force_sum = 0.0
    giant_after_top1 = np.nan
    giant_after_top5 = np.nan
    giant_after_top10 = np.nan
    force_share_top1 = np.nan
    force_share_top5 = np.nan
    force_share_top10 = np.nan
    current_force = 0.0
    for rank, row_idx in enumerate(order, start=1):
        row = d.iloc[int(row_idx)]
        a = id_to_idx[int(row["id1"])]
        b = id_to_idx[int(row["id2"])]
        f = float(row["force"])
        current_force += f
        root_a = uf.find(a)
        root_b = uf.find(b)
        if root_a == root_b:
            h1_births += 1
            h1_birth_force_sum += f
            root = root_a
        else:
            root, _ = uf.union(a, b)
        edge_frac = rank / len(order)
        force_frac = current_force / total_force if total_force > 0 else np.nan
        if np.isnan(bottom_top_at) and uf.bottom[root] and uf.top[root]:
            bottom_top_at = edge_frac
        if np.isnan(bottom_side_at) and uf.bottom[root] and uf.side[root]:
            bottom_side_at = edge_frac
        if np.isnan(giant_after_top1) and edge_frac >= 0.01:
            giant_after_top1 = uf.max_size / n_nodes
            force_share_top1 = force_frac
        if np.isnan(giant_after_top5) and edge_frac >= 0.05:
            giant_after_top5 = uf.max_size / n_nodes
            force_share_top5 = force_frac
        if np.isnan(giant_after_top10) and edge_frac >= 0.10:
            giant_after_top10 = uf.max_size / n_nodes
            force_share_top10 = force_frac
    return {
        "tag": state.tag,
        "regime_id": REGIME_ID[state.tag],
        "phase": state.phase,
        "contacts": float(len(d)),
        "force_sum": total_force,
        "force_mean": float(forces.mean()),
        "force_p99": float(np.percentile(forces, 99)),
        "bottom_top_percolation_edge_fraction": float(bottom_top_at),
        "bottom_side_percolation_edge_fraction": float(bottom_side_at),
        "giant_fraction_after_top1_edges": float(giant_after_top1),
        "giant_fraction_after_top5_edges": float(giant_after_top5),
        "giant_fraction_after_top10_edges": float(giant_after_top10),
        "force_share_top1_edges": float(force_share_top1),
        "force_share_top5_edges": float(force_share_top5),
        "force_share_top10_edges": float(force_share_top10),
        "force_h1_births": float(h1_births),
        "force_h1_birth_fraction": float(h1_births / len(d)),
        "force_h1_birth_force_share": float(h1_birth_force_sum / total_force) if total_force > 0 else np.nan,
    }


def hot_cold_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    cold = metrics[metrics["phase"] == "cold"].set_index("tag")
    hot = metrics[metrics["phase"] == "hot"].set_index("tag")
    cols = [
        "bottom_side_percolation_edge_fraction",
        "giant_fraction_after_top5_edges",
        "force_share_top5_edges",
        "force_h1_birth_fraction",
        "force_h1_birth_force_share",
        "force_p99",
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


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 11) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 5:
        return np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return rho, p


def build_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    hot = metrics[metrics["phase"] == "hot"].copy()
    rows = []
    for predictor in [
        "bottom_side_percolation_edge_fraction",
        "giant_fraction_after_top5_edges",
        "force_share_top5_edges",
        "force_h1_birth_fraction",
        "force_h1_birth_force_share",
    ]:
        rho, p = permutation_spearman(hot[predictor].to_numpy(float), hot["force_p99"].to_numpy(float))
        rows.append(
            {
                "phase": "hot",
                "predictor": predictor,
                "target": "force_p99",
                "n": len(hot),
                "spearman_rho": rho,
                "permutation_p_two_sided": p,
            }
        )
    return pd.DataFrame(rows)


def build_figure(metrics: pd.DataFrame, delta: pd.DataFrame, tests: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.15, 4.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    x = np.arange(len(REGIME_ORDER))
    labels = [REGIME_ID[t] for t in REGIME_ORDER]
    cold = metrics[metrics["phase"] == "cold"].set_index("tag")
    hot = metrics[metrics["phase"] == "hot"].set_index("tag")

    ax = fig.add_subplot(gs[0, 0])
    for i, tag in enumerate(REGIME_ORDER):
        ax.plot([i - 0.12, i + 0.12], [cold.loc[tag, "force_share_top5_edges"], hot.loc[tag, "force_share_top5_edges"]], color="#B8BEC6", lw=0.9)
        ax.scatter(i - 0.12, cold.loc[tag, "force_share_top5_edges"], color=COLD, s=25, edgecolor="white", linewidth=0.5)
        ax.scatter(i + 0.12, hot.loc[tag, "force_share_top5_edges"], color=HOT, s=28, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("force share in top 5% edges")
    ax.set_title("force concentration", fontsize=7.5, pad=5)
    finish(ax, "y")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for i, tag in enumerate(REGIME_ORDER):
        ax.plot([i - 0.12, i + 0.12], [cold.loc[tag, "giant_fraction_after_top5_edges"], hot.loc[tag, "giant_fraction_after_top5_edges"]], color="#B8BEC6", lw=0.9)
        ax.scatter(i - 0.12, cold.loc[tag, "giant_fraction_after_top5_edges"], color=COLD, s=25, edgecolor="white", linewidth=0.5)
        ax.scatter(i + 0.12, hot.loc[tag, "giant_fraction_after_top5_edges"], color=HOT, s=28, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("giant component after top 5%")
    ax.set_title("force-subgraph growth", fontsize=7.5, pad=5)
    finish(ax, "y")
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for i, tag in enumerate(REGIME_ORDER):
        ax.plot([i - 0.12, i + 0.12], [cold.loc[tag, "force_h1_birth_force_share"], hot.loc[tag, "force_h1_birth_force_share"]], color="#B8BEC6", lw=0.9)
        ax.scatter(i - 0.12, cold.loc[tag, "force_h1_birth_force_share"], color=COLD, s=25, edgecolor="white", linewidth=0.5, label="cold" if i == 0 else None)
        ax.scatter(i + 0.12, hot.loc[tag, "force_h1_birth_force_share"], color=HOT, s=28, edgecolor="white", linewidth=0.5, label="hot" if i == 0 else None)
    ax.set_xticks(x, labels)
    ax.set_ylabel("force share in cycle-closing edges")
    ax.set_title("force-weighted graph cycles", fontsize=7.5, pad=5)
    ax.legend(loc="upper left", fontsize=5.8)
    finish(ax, "y")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(hot["force_share_top5_edges"], hot["force_p99"], color=HOT, s=42, edgecolor="white", linewidth=0.6)
    for tag, row in hot.iterrows():
        ax.text(row["force_share_top5_edges"] + 0.003, row["force_p99"], REGIME_ID[tag], fontsize=6.8, va="center")
    ax.set_xlabel("top-5% force share")
    ax.set_ylabel(r"hot $f_{99}$")
    ax.set_title("tail scale vs force concentration", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(hot["giant_fraction_after_top5_edges"], hot["force_p99"], color=ACCENT, s=42, edgecolor="white", linewidth=0.6)
    for tag, row in hot.iterrows():
        ax.text(row["giant_fraction_after_top5_edges"] + 0.002, row["force_p99"], REGIME_ID[tag], fontsize=6.8, va="center")
    ax.set_xlabel("top-5% giant fraction")
    ax.set_ylabel(r"hot $f_{99}$")
    ax.set_title("tail scale vs force connectivity", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    t = tests.copy()
    names = {
        "bottom_side_percolation_edge_fraction": "perc.",
        "giant_fraction_after_top5_edges": "giant",
        "force_share_top5_edges": "share",
        "force_h1_birth_fraction": "H1 n",
        "force_h1_birth_force_share": "H1 f",
    }
    xx = np.arange(len(t))
    ax.bar(xx, t["spearman_rho"], color=["#8B929A", ACCENT, HOT, "#7F5AA2", "#D98C3A"], width=0.65)
    for i, row in enumerate(t.itertuples()):
        ax.text(i, row.spearman_rho + (0.06 if row.spearman_rho >= 0 else -0.08), f"p={row.permutation_p_two_sided:.2f}", ha="center", va="bottom" if row.spearman_rho >= 0 else "top", fontsize=5.5)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xticks(xx, [names[v] for v in t["predictor"]])
    ax.set_ylabel(r"Spearman $\rho$ with hot $f_{99}$")
    ax.set_title("six-regime diagnostics", fontsize=7.5, pad=5)
    finish(ax, "y")
    panel(ax, "f")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig7_true_force_percolation.{ext}", **kwargs)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, delta: pd.DataFrame, tests: pd.DataFrame) -> None:
    hot = metrics[metrics["phase"] == "hot"].copy()
    best = tests.sort_values("permutation_p_two_sided").iloc[0]
    lines = [
        "# True-force percolation report",
        "",
        "## Strict scope",
        "",
        "This analysis uses true `compute pair/gran/local` contact-force outputs, but only for cycle-10 hot/cold snapshots across six regimes. It is therefore a direct force-network test, not a multi-cycle memory test.",
        "",
        "## What was computed",
        "",
        "- Force-weighted percolation: pair contacts are added from largest to smallest true contact-force magnitude.",
        "- Force-subgraph growth: giant component after the strongest 1%, 5% and 10% of contacts.",
        "- Force-weighted graph-cycle births: edges that close graph cycles during force-ordered filtration.",
        "- Six-regime Spearman plus permutation diagnostics against hot force p99.",
        "",
        "## Main results",
        "",
        f"- Hot top-5% force share ranges from {hot['force_share_top5_edges'].min():.3f} to {hot['force_share_top5_edges'].max():.3f}.",
        f"- Hot giant component after the strongest 5% contacts ranges from {hot['giant_fraction_after_top5_edges'].min():.3f} to {hot['giant_fraction_after_top5_edges'].max():.3f}.",
        f"- The strongest diagnostic among tested force-network variables is `{best['predictor']}` with Spearman rho = {best['spearman_rho']:.2f}, permutation p = {best['permutation_p_two_sided']:.3f}.",
        "",
        "## Interpretation",
        "",
        "The true-force snapshots support force concentration as a real stress-sector feature, but the sample size is only six hot states. These diagnostics should be treated as regime-level evidence and as a target for multi-cycle force-local reruns, not as a universal scaling law.",
        "",
        "## What is not supported yet",
        "",
        "This analysis does not prove that a single force-percolation threshold controls hot overload. It also cannot test contact-memory causality because true force-local files are currently available only at cycle 10 for this six-regime set.",
        "",
    ]
    (ROOT / "nature_physics_true_force_percolation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    states = load_states()
    rows = [force_percolation(s) for s in states]
    metrics = pd.DataFrame(rows)
    delta = hot_cold_delta(metrics)
    tests = build_tests(metrics)
    metrics.to_csv(SRC / "nphys_true_force_percolation_metrics.csv", index=False)
    delta.to_csv(SRC / "nphys_true_force_percolation_hot_cold_delta.csv", index=False)
    tests.to_csv(SRC / "nphys_true_force_percolation_tests.csv", index=False)
    build_figure(metrics, delta, tests)
    write_report(metrics, delta, tests)
    print(metrics[["regime_id", "phase", "force_share_top5_edges", "giant_fraction_after_top5_edges", "force_h1_birth_force_share"]])
    print(tests)


if __name__ == "__main__":
    main()
