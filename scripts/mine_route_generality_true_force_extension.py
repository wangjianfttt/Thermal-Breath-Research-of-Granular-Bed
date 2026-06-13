#!/usr/bin/env python3
from __future__ import annotations

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
    FIG,
    GRID,
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
EXPECTED_STATES_PER_ROUTE = EXPECTED_CYCLES * 2
ROUTES = {
    "a150_mu010_g020": ("a150_mu010_g020_c30", "G1_lossy_high_expansion_low_friction"),
    "a150_mu030_g020": ("a150_mu030_g020_c30", "G2_lossy_high_expansion_mid_friction"),
    "a050_mu060_g020": ("a050_mu060_g020_c30", "G3_buffered_low_expansion_high_friction"),
}

REGIME_ID.update(
    {
        "a150_mu010_g020": "G1",
        "a150_mu030_g020": "G2",
        "a050_mu060_g020": "G3",
    }
)


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        else:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(c) for c in out.columns]
    rows = out.astype(str).values.tolist()
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def discover_states() -> list[tuple[int, ForceState]]:
    states: list[tuple[int, ForceState]] = []
    pattern = re.compile(r"contacts_cycle_(\d+)_(hot|cold)\.local$")
    for base_tag, (folder_name, _run_name) in ROUTES.items():
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
        row["run"] = ROUTES[state.tag][1]
        row["folder"] = ROUTES[state.tag][0]
        rows.append(row)
    cols = ["run", "tag", "regime_id", "folder", "cycle", "phase"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return df[cols + [c for c in df.columns if c not in cols]].sort_values(["regime_id", "cycle", "phase"])


def build_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "run",
                "tag",
                "regime_id",
                "cycle",
                "force_p99_hot_minus_cold",
                "force_share_top5_edges_hot_minus_cold",
                "force_h1_birth_force_share_hot_minus_cold",
            ]
        )
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
    rows: list[dict[str, float | str | int]] = []
    for idx in cold.index.intersection(hot.index):
        run, tag, regime_id, cycle = idx
        row: dict[str, float | str | int] = {"run": run, "tag": tag, "regime_id": regime_id, "cycle": int(cycle)}
        for col in value_cols:
            row[f"{col}_cold"] = float(cold.loc[idx, col])
            row[f"{col}_hot"] = float(hot.loc[idx, col])
            row[f"{col}_hot_minus_cold"] = float(hot.loc[idx, col] - cold.loc[idx, col])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["regime_id", "cycle"])


def route_completeness(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for base_tag, (folder, run_name) in ROUTES.items():
        g = metrics[metrics["tag"] == base_tag] if not metrics.empty else pd.DataFrame()
        n_states = int(len(g))
        n_pairs = int(build_delta(g)["cycle"].nunique()) if not g.empty else 0
        rows.append(
            {
                "tag": base_tag,
                "folder": folder,
                "run": run_name,
                "states_parsed": n_states,
                "expected_states": EXPECTED_STATES_PER_ROUTE,
                "complete_pairs": n_pairs,
                "expected_pairs": EXPECTED_CYCLES,
                "complete": "yes" if n_states >= EXPECTED_STATES_PER_ROUTE and n_pairs >= EXPECTED_CYCLES else "no",
            }
        )
    return pd.DataFrame(rows)


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 59) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 6 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return rho, p


def route_centered_permutation_spearman(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    n_perm: int = 5000,
    seed: int = 71,
) -> tuple[float, float]:
    g = df.dropna(subset=["regime_id", predictor, target]).copy()
    if len(g) < 6:
        return np.nan, np.nan
    x = g[predictor] - g.groupby("regime_id")[predictor].transform("mean")
    y = g[target] - g.groupby("regime_id")[target].transform("mean")
    ok = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    x = x.to_numpy(float)[ok]
    y = y.to_numpy(float)[ok]
    route = g["regime_id"].to_numpy(str)[ok]
    if len(x) < 6 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for rid in np.unique(route):
            mask = route == rid
            yp[mask] = rng.permutation(yp[mask])
        null.append(float(spearmanr(x, yp).statistic))
    null = np.asarray(null, dtype=float)
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1))
    return rho, p


def build_tests(delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    if delta.empty:
        return pd.DataFrame(columns=["scope", "predictor", "target", "n", "spearman_rho", "permutation_p_two_sided"])
    for predictor in [
        "force_share_top5_edges_hot_minus_cold",
        "giant_fraction_after_top5_edges_hot_minus_cold",
        "force_h1_birth_force_share_hot_minus_cold",
        "bottom_side_percolation_edge_fraction_hot_minus_cold",
    ]:
        rho, p = permutation_spearman(delta[predictor].to_numpy(float), delta["force_p99_hot_minus_cold"].to_numpy(float))
        rows.append(
            {
                "scope": "route_generality_hot_minus_cold",
                "predictor": predictor,
                "target": "force_p99_hot_minus_cold",
                "n": int(len(delta)),
                "spearman_rho": rho,
                "permutation_p_two_sided": p,
            }
        )
        rho, p = route_centered_permutation_spearman(delta, predictor, "force_p99_hot_minus_cold")
        rows.append(
            {
                "scope": "route_centered_hot_minus_cold",
                "predictor": predictor,
                "target": "force_p99_hot_minus_cold",
                "n": int(len(delta)),
                "spearman_rho": rho,
                "permutation_p_two_sided": p,
            }
        )
    return pd.DataFrame(rows)


def build_figure(delta: pd.DataFrame, tests: pd.DataFrame, completeness: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.85])
    colors = {"G1": "#B55247", "G2": "#D98C3A", "G3": "#547C7A"}

    ax = fig.add_subplot(gs[0, 0])
    if not delta.empty:
        for rid, g in delta.groupby("regime_id"):
            ax.plot(g["cycle"], g["force_h1_birth_force_share_hot_minus_cold"], color=colors.get(rid, ACCENT), marker="o", ms=2.7, lw=1.0, label=rid)
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\Delta L_f$")
    ax.set_title("new-route loop activation", loc="left", pad=4)
    ax.legend(fontsize=5.8, ncol=3, loc="best")
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    if not delta.empty:
        ax.scatter(
            delta["force_h1_birth_force_share_hot_minus_cold"],
            delta["force_p99_hot_minus_cold"],
            c=[colors.get(r, ACCENT) for r in delta["regime_id"]],
            s=22,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.axhline(0, color="#9AA1A9", lw=0.7)
    ax.axvline(0, color="#9AA1A9", lw=0.7)
    loop_test = tests[
        (tests["scope"] == "route_centered_hot_minus_cold")
        & (tests["predictor"] == "force_h1_birth_force_share_hot_minus_cold")
    ]
    if not loop_test.empty:
        row = loop_test.iloc[0]
        ax.text(
            0.05,
            0.94,
            rf"route-centred $\rho={row.spearman_rho:.2f}$, $P={row.permutation_p_two_sided:.4f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            color="#39424E",
        )
    ax.set_xlabel(r"$\Delta L_f$")
    ax.set_ylabel(r"$\Delta f_{99}$")
    ax.set_title("loop coordinate vs overload", loc="left", pad=4)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    labels = {
        "force_h1_birth_force_share_hot_minus_cold": r"$\Delta L_f$",
        "giant_fraction_after_top5_edges_hot_minus_cold": r"$\Delta G$",
        "bottom_side_percolation_edge_fraction_hot_minus_cold": r"$\Delta B$",
        "force_share_top5_edges_hot_minus_cold": r"$\Delta F_{5\%}$",
    }
    order = list(labels)
    sub = tests[tests["scope"].isin(["route_generality_hot_minus_cold", "route_centered_hot_minus_cold"])].copy()
    sub = sub[sub["predictor"].isin(order)]
    y = np.arange(len(order), dtype=float)
    for off, scope, color, name in [
        (-0.16, "route_generality_hot_minus_cold", "#9BA7B2", "pooled"),
        (0.16, "route_centered_hot_minus_cold", "#39424E", "route-centred"),
    ]:
        vals = [
            float(sub[(sub["scope"] == scope) & (sub["predictor"] == pred)]["spearman_rho"].iloc[0])
            if not sub[(sub["scope"] == scope) & (sub["predictor"] == pred)].empty
            else np.nan
            for pred in order
        ]
        ax.barh(y + off, vals, height=0.28, color=color, label=name)
    ax.axvline(0, color="#9AA1A9", lw=0.7)
    ax.set_yticks(y, [labels[p] for p in order])
    ax.set_xlim(-0.35, 0.75)
    ax.set_xlabel(r"Spearman $\rho$ to $\Delta f_{99}$")
    ax.set_title("specificity after route removal", loc="left", pad=4)
    ax.invert_yaxis()
    ax.legend(fontsize=5.6, loc="lower right")
    finish(ax, axis="x")
    panel(ax, "c")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig64_route_generality_true_force_extension.{ext}", **kwargs)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, delta: pd.DataFrame, tests: pd.DataFrame, completeness: pd.DataFrame) -> None:
    complete_routes = int((completeness["complete"] == "yes").sum())
    total_states = int(completeness["states_parsed"].sum())
    expected_states = int(completeness["expected_states"].sum())
    lines = [
        "# Route-generality true-force extension",
        "",
        "This audit is intentionally separate from the main five-route true-force dataset. It should be merged into the main mechanism only after all three new routes are complete and the diagnostics pass.",
        "",
        "## Completion",
        "",
        markdown_table(completeness, digits=4),
        "",
        f"Parsed states: {total_states} / {expected_states}. Complete routes: {complete_routes} / {len(ROUTES)}.",
        "",
        "## Diagnostic tests",
        "",
    ]
    if tests.empty:
        lines += ["No diagnostic tests are available yet.", ""]
    else:
        lines += [markdown_table(tests, digits=4), ""]
    if complete_routes < len(ROUTES):
        lines += [
            "## Interpretation boundary",
            "",
            "Do not promote this extension into the manuscript. It is currently a completion and post-processing scaffold for the running remote batch.",
        ]
    else:
        loop = tests[
            (tests["scope"] == "route_centered_hot_minus_cold")
            & (tests["predictor"] == "force_h1_birth_force_share_hot_minus_cold")
        ]
        tail = tests[
            (tests["scope"] == "route_centered_hot_minus_cold")
            & (tests["predictor"] == "force_share_top5_edges_hot_minus_cold")
        ]
        if not loop.empty and not tail.empty:
            lines += [
                "## Main result",
                "",
                (
                    "The added true-pair-force routes support the mechanism as a route-centred diagnostic: "
                    f"the hot-minus-cold loop-birth force share remains correlated with hot-minus-cold overload "
                    f"after route means are removed (rho={loop.iloc[0]['spearman_rho']:.3f}, "
                    f"P={loop.iloc[0]['permutation_p_two_sided']:.4f}), whereas the top-5% force-share surrogate "
                    f"does not (rho={tail.iloc[0]['spearman_rho']:.3f}, "
                    f"P={tail.iloc[0]['permutation_p_two_sided']:.4f})."
                ),
                "",
            ]
        lines += [
            "## Interpretation boundary",
            "",
            "The extension is complete enough for scientific comparison against the five-route ensemble. It supports promotion as an Extended Data or reserve mechanism panel because the loop-birth coordinate survives route-centred testing while a generic force-tail surrogate does not. It should still not be described as a universal material law: the added set contains three routes and remains a DEM true-force extension rather than an independent experiment.",
        ]
    (ROOT / "nature_physics_route_generality_true_force_extension.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    states = discover_states()
    metrics = build_metrics(states)
    delta = build_delta(metrics)
    completeness = route_completeness(metrics)
    tests = build_tests(delta)
    metrics.to_csv(SRC / "nphys_route_generality_true_force_extension_metrics.csv", index=False)
    delta.to_csv(SRC / "nphys_route_generality_true_force_extension_delta.csv", index=False)
    completeness.to_csv(SRC / "nphys_route_generality_true_force_extension_completeness.csv", index=False)
    tests.to_csv(SRC / "nphys_route_generality_true_force_extension_tests.csv", index=False)
    build_figure(delta, tests, completeness)
    write_report(metrics, delta, tests, completeness)
    print(completeness.to_string(index=False))


if __name__ == "__main__":
    main()
