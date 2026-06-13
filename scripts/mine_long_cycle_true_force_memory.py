#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mine_true_force_percolation import (
    ACCENT,
    COLD,
    FIG,
    GRID,
    HOT,
    PROJECT,
    REGIME_ID,
    ROOT,
    SRC,
    ForceState,
    force_percolation,
    panel,
    read_dump,
    read_local,
    setup_style,
)


RUN_BASE = PROJECT / "runs" / "long_cycle_force_probe"
EXPECTED_CYCLES = 30
LONG_REGIMES = {
    "a050_mu010_g002": ("a050_mu010_g002_c30", "R1_low_expansion_low_friction"),
    "a100_mu030_g002": ("a100_mu030_g002_c30", "R3_intermediate"),
    "a150_mu060_g020": ("a150_mu060_g020_c30", "R6_high_expansion_high_friction"),
    "a150_mu060_g000": ("a150_mu060_g000_c30", "R6_closed_high_expansion_high_friction"),
    "a150_mu030_g002": ("a150_mu030_g002_c30", "R5_high_expansion_intermediate_friction"),
}

REGIME_ID.update(
    {
        "a150_mu060_g000": "R6c",
        "a150_mu030_g002": "R5",
    }
)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def discover_states() -> list[tuple[int, ForceState]]:
    states: list[tuple[int, ForceState]] = []
    pattern = re.compile(r"contacts_cycle_(\d+)_(hot|cold)\.local$")
    for base_tag, (folder_name, _run_name) in LONG_REGIMES.items():
        folder = RUN_BASE / folder_name
        if not folder.exists():
            continue
        for local in sorted(folder.glob("contacts_cycle_*_*.local")):
            match = pattern.match(local.name)
            if not match:
                continue
            cycle = int(match.group(1))
            phase = match.group(2)
            dump = folder / f"cycle_{cycle}_{phase}.dump"
            if not dump.exists():
                continue
            contacts = read_local(local)
            atoms = read_dump(dump)
            if contacts.empty or atoms.empty:
                continue
            states.append((cycle, ForceState(base_tag, phase, contacts, atoms)))
    return states


def build_metrics(states: list[tuple[int, ForceState]]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for cycle, state in states:
        row = force_percolation(state)
        row["cycle"] = cycle
        row["run"] = LONG_REGIMES[state.tag][1]
        row["folder"] = LONG_REGIMES[state.tag][0]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    cols = ["run", "tag", "regime_id", "folder", "cycle", "phase"]
    df = pd.DataFrame(rows)
    return df[cols + [c for c in df.columns if c not in cols]].sort_values(["regime_id", "cycle", "phase"])


def build_cycle_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    value_cols = [
        "force_p99",
        "force_share_top5_edges",
        "giant_fraction_after_top5_edges",
        "force_h1_birth_fraction",
        "force_h1_birth_force_share",
        "bottom_side_percolation_edge_fraction",
    ]
    cold = metrics[metrics["phase"] == "cold"].set_index(["run", "tag", "regime_id", "cycle"])
    hot = metrics[metrics["phase"] == "hot"].set_index(["run", "tag", "regime_id", "cycle"])
    common = cold.index.intersection(hot.index)
    rows: list[dict[str, float | str | int]] = []
    for idx in common:
        run, tag, regime_id, cycle = idx
        row: dict[str, float | str | int] = {"run": run, "tag": tag, "regime_id": regime_id, "cycle": int(cycle)}
        for col in value_cols:
            row[f"{col}_cold"] = float(cold.loc[idx, col])
            row[f"{col}_hot"] = float(hot.loc[idx, col])
            row[f"{col}_hot_minus_cold"] = float(hot.loc[idx, col] - cold.loc[idx, col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime_id", "cycle"])


def join_with_geometric_topology(metrics: pd.DataFrame) -> pd.DataFrame:
    topo_path = SRC / "nphys_deep_topology_percolation_metrics.csv"
    if metrics.empty or not topo_path.exists():
        return pd.DataFrame()
    topo = pd.read_csv(topo_path)
    joined = metrics.merge(
        topo,
        on=["run", "cycle", "phase"],
        how="inner",
        suffixes=("_true_force", "_geometry"),
    )
    return joined.sort_values(["regime_id", "cycle", "phase"])


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 21) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 6 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return rho, p


def build_tests(joined: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    n_regimes = 0 if joined.empty else int(joined["regime_id"].nunique())
    inferential_status = "registered_partial" if n_regimes < len(LONG_REGIMES) else "registered_complete_regime_set"
    if not joined.empty:
        hot = joined[joined["phase"] == "hot"].copy()
        for predictor in [
            "orientation_entropy",
            "bottom_side_percolation_edge_fraction_geometry",
            "cycle_birth_positive_fraction",
            "force_proxy_gini",
            "force_share_top5_edges",
            "force_h1_birth_force_share",
        ]:
            if predictor in hot.columns:
                rho, p = permutation_spearman(hot[predictor].to_numpy(float), hot["force_p99"].to_numpy(float))
                rows.append(
                    {
                        "scope": "hot_states",
                        "predictor": predictor,
                        "target": "true_force_p99",
                        "n": int(len(hot)),
                        "n_regimes": n_regimes,
                        "inferential_status": inferential_status,
                        "spearman_rho": rho,
                        "permutation_p_two_sided": p,
                    }
                )
    if not delta.empty:
        for predictor in [
            "force_share_top5_edges_hot_minus_cold",
            "giant_fraction_after_top5_edges_hot_minus_cold",
            "force_h1_birth_force_share_hot_minus_cold",
        ]:
            rho, p = permutation_spearman(delta[predictor].to_numpy(float), delta["force_p99_hot_minus_cold"].to_numpy(float))
            rows.append(
                {
                    "scope": "cycle_hot_minus_cold",
                    "predictor": predictor,
                    "target": "force_p99_hot_minus_cold",
                    "n": int(len(delta)),
                    "n_regimes": n_regimes,
                    "inferential_status": inferential_status,
                    "spearman_rho": rho,
                    "permutation_p_two_sided": p,
                }
            )
    return pd.DataFrame(rows)


def build_figure(metrics: pd.DataFrame, delta: pd.DataFrame, joined: pd.DataFrame, tests: pd.DataFrame) -> None:
    if metrics.empty:
        return
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    colors = {
        "R1": "#345995",
        "R3": "#D98C3A",
        "R5": "#7E6AAE",
        "R6": "#C95F3F",
        "R6c": "#9E3D34",
    }

    ax = fig.add_subplot(gs[0, 0])
    for rid, group in delta.groupby("regime_id"):
        ax.plot(group["cycle"], group["force_share_top5_edges_hot_minus_cold"], marker="o", ms=2.8, lw=1.0, color=colors.get(rid, ACCENT), label=rid)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Delta$ top-5% force share")
    ax.set_title("cycling of force concentration", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.8, ncol=3, loc="best")
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for rid, group in delta.groupby("regime_id"):
        ax.plot(group["cycle"], group["force_p99_hot_minus_cold"], marker="o", ms=2.8, lw=1.0, color=colors.get(rid, ACCENT))
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Delta f_{99}$")
    ax.set_title("tail-force memory", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for rid, group in delta.groupby("regime_id"):
        ax.plot(group["cycle"], group["force_h1_birth_force_share_hot_minus_cold"], marker="o", ms=2.8, lw=1.0, color=colors.get(rid, ACCENT))
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Delta$ cycle-closing force share")
    ax.set_title("force-loop activation", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    hot = joined[joined["phase"] == "hot"] if not joined.empty else pd.DataFrame()
    if not hot.empty and "orientation_entropy" in hot:
        for rid, group in hot.groupby("regime_id"):
            ax.scatter(group["orientation_entropy"], group["force_p99"], s=18, color=colors.get(rid, ACCENT), edgecolor="white", linewidth=0.35, label=rid)
    ax.set_xlabel("hot orientation entropy")
    ax.set_ylabel(r"true hot $f_{99}$")
    ax.set_title("geometry-force coupling", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    if not delta.empty:
        ax.scatter(delta["force_share_top5_edges_hot_minus_cold"], delta["force_p99_hot_minus_cold"], c=[colors.get(r, ACCENT) for r in delta["regime_id"]], s=22, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.axvline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel(r"$\Delta$ top-5% force share")
    ax.set_ylabel(r"$\Delta f_{99}$")
    ax.set_title("concentration vs overload", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    show = tests.dropna(subset=["spearman_rho"]).head(8) if not tests.empty else pd.DataFrame()
    if not show.empty:
        short_names = {
            "orientation_entropy": "entropy",
            "bottom_side_percolation_edge_fraction_geometry": "geom\nperc.",
            "cycle_birth_positive_fraction": "geom\nloops",
            "force_proxy_gini": "geom\nGini",
            "force_share_top5_edges": "top-5%\nshare",
            "force_h1_birth_force_share": "force\nloops",
            "force_share_top5_edges_hot_minus_cold": r"$\Delta$ top-5%",
            "giant_fraction_after_top5_edges_hot_minus_cold": r"$\Delta$ giant",
            "force_h1_birth_force_share_hot_minus_cold": r"$\Delta$ loops",
        }
        xx = np.arange(len(show))
        ax.bar(xx, show["spearman_rho"], color="#6F7C8A", width=0.68)
        ax.set_xticks(xx, [short_names.get(str(v), str(v)) for v in show["predictor"]], fontsize=5.8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
        ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("registered diagnostics", fontsize=7.5, pad=5)
    finish(ax, "y")
    panel(ax, "f")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig8_long_cycle_true_force_memory.{ext}", **kwargs)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, delta: pd.DataFrame, joined: pd.DataFrame, tests: pd.DataFrame) -> None:
    completed = 0 if metrics.empty else int(metrics.groupby(["run", "cycle", "phase"]).ngroups)
    expected = len(LONG_REGIMES) * EXPECTED_CYCLES * 2
    complete_cycles = 0 if delta.empty else int(delta.groupby(["run", "cycle"]).ngroups)
    parsed_regimes = 0 if metrics.empty else int(metrics["tag"].nunique())
    status = "complete" if completed == expected else "partial"
    lines = [
        "# Long-cycle true-force memory report",
        "",
        "## Scope",
        "",
        "This file is generated from true `compute pair/gran/local` force dumps from the long-cycle rerun. It is allowed to remain partial while the remote job is still running.",
        "",
        "## Current data status",
        "",
        f"- Status: `{status}`.",
        f"- Parsed states: {completed} / {expected}.",
        f"- Complete hot/cold cycle pairs: {complete_cycles} / {len(LONG_REGIMES) * EXPECTED_CYCLES}.",
        f"- Parsed targeted routes: {parsed_regimes} / {len(LONG_REGIMES)}.",
        "",
    ]
    if not metrics.empty:
        hot = metrics[metrics["phase"] == "hot"]
        lines += [
            "## Current measured ranges",
            "",
            f"- Hot true-force p99 range: {hot['force_p99'].min():.6g} to {hot['force_p99'].max():.6g}.",
            f"- Hot top-5% force-share range: {hot['force_share_top5_edges'].min():.3f} to {hot['force_share_top5_edges'].max():.3f}.",
            "",
        ]
    if not tests.empty and tests["spearman_rho"].notna().any():
        best = tests.dropna(subset=["spearman_rho"]).sort_values("permutation_p_two_sided").iloc[0]
        joined_routes = 0 if joined.empty else int(joined["tag"].nunique())
        lines += [
            "## Registered diagnostics",
            "",
            f"- Strongest current diagnostic: `{best['predictor']}` vs `{best['target']}`, rho = {best['spearman_rho']:.2f}, permutation p = {best['permutation_p_two_sided']:.3f}, n = {int(best['n'])}.",
            f"- Diagnostic status: `{best['inferential_status']}`. Geometry-to-true-force joined diagnostics currently cover {joined_routes} routes; the two newly added route tests are included in the true-force delta analysis but not yet in the geometric-topology join.",
            "",
        ]
    lines += ["## Conservative interpretation", ""]
    if status == "complete":
        lines += [
            f"The expected {expected} states are now parsed. The data can be used as completed evidence for route-resolved true-force memory, but the claim should remain route-specific: it supports force-loop activation as a topology-conditioned overload pathway, not a universal force-percolation threshold.",
            "",
        ]
    else:
        lines += [
            f"Do not use this as a manuscript claim until the parsed-state count reaches the expected {expected} states. Once complete, this analysis can test whether the previously observed geometric rewriting routes have a matching true-force memory signature across cycles.",
            "",
        ]
    (ROOT / "nature_physics_long_cycle_true_force_memory_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    states = discover_states()
    metrics = build_metrics(states)
    delta = build_cycle_delta(metrics)
    joined = join_with_geometric_topology(metrics)
    tests = build_tests(joined, delta)
    metrics.to_csv(SRC / "nphys_long_cycle_true_force_metrics.csv", index=False)
    delta.to_csv(SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv", index=False)
    joined.to_csv(SRC / "nphys_long_cycle_true_force_geometry_join.csv", index=False)
    tests.to_csv(SRC / "nphys_long_cycle_true_force_tests.csv", index=False)
    build_figure(metrics, delta, joined, tests)
    write_report(metrics, delta, joined, tests)
    print(f"parsed_states={0 if metrics.empty else len(metrics)}")
    print(f"complete_hot_cold_pairs={0 if delta.empty else len(delta)}")
    if not tests.empty:
        print(tests)


if __name__ == "__main__":
    main()
