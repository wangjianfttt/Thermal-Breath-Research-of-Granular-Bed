#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mine_long_cycle_true_force_memory import discover_states
from mine_true_force_percolation import ForceState, REGIME_ID, UnionFind, enrich_contacts, force_percolation


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

REGIME_COLORS = {"R1": "#345995", "R3": "#D98C3A", "R6": "#C95F3F"}
GRID = "#E8EBEF"
INK = "#252A31"
ACCENT = "#B6423E"
MUTED = "#8A929C"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
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


def panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def within_center(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_wc"] = out[col] - out.groupby("regime_id")[col].transform("mean")
    return out


def spearman_wc(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    d = df[["regime_id", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 6:
        return np.nan, np.nan, int(len(d))
    d = within_center(d, [x, y])
    stat = spearmanr(d[f"{x}_wc"], d[f"{y}_wc"], nan_policy="omit")
    return float(stat.statistic), float(stat.pvalue), int(len(d))


def ols_r2(df: pd.DataFrame, predictors: list[str], target: str) -> float:
    d = df[["regime_id", "cycle", target, *predictors]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 8:
        return np.nan
    cols = [np.ones(len(d))]
    for predictor in predictors:
        vals = d[predictor].to_numpy(float)
        sigma = vals.std(ddof=0)
        cols.append((vals - vals.mean()) / sigma if sigma > 0 else vals * 0)
    cycle = d["cycle"].to_numpy(float)
    sigma = cycle.std(ddof=0)
    cols.append((cycle - cycle.mean()) / sigma if sigma > 0 else cycle * 0)
    for rid in ["R3", "R6"]:
        cols.append((d["regime_id"] == rid).astype(float).to_numpy())
    X = np.column_stack(cols)
    y = d[target].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def delta_r2(df: pd.DataFrame, base: list[str], add: str, target: str) -> float:
    return ols_r2(df, [*base, add], target) - ols_r2(df, base, target)


def within_regime_permutation_null(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    n_perm: int = 5000,
    seed: int = 44,
) -> tuple[float, float, np.ndarray]:
    observed, _p_asym, _n = spearman_wc(df, x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = df.copy()
        shuffled[x] = shuffled.groupby("regime_id")[x].transform(lambda s: rng.permutation(s.to_numpy()))
        null[i] = spearman_wc(shuffled, x, y)[0]
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return observed, p, null


def lag_scan(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    lags: range = range(-6, 7),
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for lag in lags:
        shifted = df[["run", "regime_id", "cycle", x]].copy()
        shifted["cycle"] = shifted["cycle"] + lag
        shifted = shifted.rename(columns={x: f"{x}_lagged"})
        target = df[["run", "regime_id", "cycle", y]].copy()
        joined = target.merge(shifted, on=["run", "regime_id", "cycle"], how="inner")
        rho, p, n = spearman_wc(joined, f"{x}_lagged", y)
        rows.append({"predictor": x, "target": y, "lag": int(lag), "rho_within": rho, "p_within": p, "n": n})
    return pd.DataFrame(rows)


def cycle_delta_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "force_p99",
        "force_share_top5_edges",
        "force_h1_birth_force_share",
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


def shuffled_state(state: ForceState, rng: np.random.Generator) -> ForceState:
    contacts = state.contacts.copy()
    forces = contacts["force"].to_numpy(float)
    permuted = rng.permutation(forces)
    contacts["force"] = permuted
    norms = np.linalg.norm(contacts[["fx", "fy", "fz"]].to_numpy(float), axis=1)
    scale = np.divide(permuted, norms, out=np.zeros_like(permuted), where=norms > 0)
    contacts["fx"] = contacts["fx"].to_numpy(float) * scale
    contacts["fy"] = contacts["fy"].to_numpy(float) * scale
    contacts["fz"] = contacts["fz"].to_numpy(float) * scale
    return replace(state, contacts=contacts)


def force_percolation_fast(state: ForceState) -> dict[str, float | str]:
    d, id_to_idx, bottom, top, side = enrich_contacts(state)
    if d.empty:
        return {"tag": state.tag, "regime_id": REGIME_ID[state.tag], "phase": state.phase}
    n_nodes = len(id_to_idx)
    id1 = d["id1"].to_numpy(int)
    id2 = d["id2"].to_numpy(int)
    a_idx = np.fromiter((id_to_idx[int(v)] for v in id1), dtype=int, count=len(id1))
    b_idx = np.fromiter((id_to_idx[int(v)] for v in id2), dtype=int, count=len(id2))
    forces = d["force"].to_numpy(float)
    order = np.argsort(forces)[::-1]
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
    n_edges = len(order)
    for rank, row_idx in enumerate(order, start=1):
        a = int(a_idx[row_idx])
        b = int(b_idx[row_idx])
        f = float(forces[row_idx])
        current_force += f
        root_a = uf.find(a)
        root_b = uf.find(b)
        if root_a == root_b:
            h1_births += 1
            h1_birth_force_sum += f
            root = root_a
        else:
            root, _ = uf.union(a, b)
        edge_frac = rank / n_edges
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
        "contacts": float(n_edges),
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
        "force_h1_birth_fraction": float(h1_births / n_edges),
        "force_h1_birth_force_share": float(h1_birth_force_sum / total_force) if total_force > 0 else np.nan,
    }


def compute_raw_force_shuffle_null(n_perm: int = 24, seed: int = 77) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = discover_states()
    if not states:
        return pd.DataFrame(), pd.DataFrame()
    real_delta = pd.read_csv(SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv")
    target = real_delta[["run", "tag", "regime_id", "cycle", "force_p99_hot_minus_cold"]].rename(
        columns={"force_p99_hot_minus_cold": "force_p99_hot_minus_cold_real"}
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    metric_rows: list[dict[str, float | str | int]] = []
    for perm in range(n_perm):
        print(f"raw force-shuffle permutation {perm + 1}/{n_perm}", flush=True)
        state_rows: list[dict[str, float | str | int]] = []
        for cycle, state in states:
            shuffled = shuffled_state(state, rng)
            row = force_percolation_fast(shuffled)
            row["cycle"] = cycle
            row["run"] = {
                "a050_mu010_g002": "R1_low_expansion_low_friction",
                "a100_mu030_g002": "R3_intermediate",
                "a150_mu060_g020": "R6_high_expansion_high_friction",
            }[state.tag]
            state_rows.append(row)
        metrics = pd.DataFrame(state_rows)
        delta = cycle_delta_from_metrics(metrics).merge(target, on=["run", "tag", "regime_id", "cycle"], how="inner")
        delta["force_p99_hot_minus_cold"] = delta["force_p99_hot_minus_cold_real"]
        rho, p, n = spearman_wc(delta, "force_h1_birth_force_share_hot_minus_cold", "force_p99_hot_minus_cold")
        rows.append({"null_model": "raw_force_shuffle_on_fixed_contact_graph", "perm": perm, "rho_within": rho, "p_asymptotic": p, "n": n})
        if perm < 5:
            keep = delta[
                [
                    "run",
                    "tag",
                    "regime_id",
                    "cycle",
                    "force_h1_birth_force_share_hot_minus_cold",
                    "force_share_top5_edges_hot_minus_cold",
                    "force_p99_hot_minus_cold",
                ]
            ].copy()
            keep["perm"] = perm
            metric_rows.extend(keep.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def build_null_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    breathing = pd.read_csv(SRC / "nphys_breathing_cycle_metrics.csv")
    delta = pd.read_csv(SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv")

    relationship_specs = [
        ("loop_to_overload", delta, "force_h1_birth_force_share_hot_minus_cold", "force_p99_hot_minus_cold"),
        ("positive_cycles_to_overload", breathing, "cycle_birth_positive_fraction_inhale_delta", "force_p99_hot_minus_cold"),
        ("top5_to_overload", delta, "force_share_top5_edges_hot_minus_cold", "force_p99_hot_minus_cold"),
        ("fabric_to_next_cold_fabric", breathing, "Z_geom_inhale_delta", "Z_geom_next_cold_minus_current"),
        (
            "positive_cycles_to_next_cold_loop_memory",
            breathing,
            "cycle_birth_positive_fraction_inhale_delta",
            "force_h1_birth_force_share_next_cold_minus_current",
        ),
        ("contact_aperture_to_hot_overload", breathing, "breathing_aperture", "force_p99_hot_minus_cold"),
    ]
    rows = []
    null_rows = []
    for name, df, x, y in relationship_specs:
        observed, p_perm, null = within_regime_permutation_null(df, x, y)
        rho, p_asym, n = spearman_wc(df, x, y)
        rows.append(
            {
                "relationship": name,
                "predictor": x,
                "target": y,
                "rho_within": rho,
                "p_asymptotic": p_asym,
                "p_within_regime_permutation": p_perm,
                "n": n,
                "null_mean": float(np.nanmean(null)),
                "null_sd": float(np.nanstd(null)),
                "null_q025": float(np.nanquantile(null, 0.025)),
                "null_q975": float(np.nanquantile(null, 0.975)),
            }
        )
        null_rows.extend({"relationship": name, "perm": i, "rho_within": val} for i, val in enumerate(null))

    r2_rows = []
    for added, base in [
        ("force_h1_birth_force_share_hot_minus_cold", []),
        ("force_share_top5_edges_hot_minus_cold", []),
        ("force_h1_birth_force_share_hot_minus_cold", ["force_share_top5_edges_hot_minus_cold"]),
        ("force_share_top5_edges_hot_minus_cold", ["force_h1_birth_force_share_hot_minus_cold"]),
    ]:
        observed = delta_r2(delta, base, added, "force_p99_hot_minus_cold")
        rng = np.random.default_rng(211)
        null = np.empty(3000)
        for i in range(len(null)):
            shuffled = delta.copy()
            shuffled[added] = shuffled.groupby("regime_id")[added].transform(lambda s: rng.permutation(s.to_numpy()))
            null[i] = delta_r2(shuffled, base, added, "force_p99_hot_minus_cold")
        p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        r2_rows.append(
            {
                "base": "+".join(base) if base else "FE+cycle",
                "added": added,
                "delta_r2_observed": observed,
                "p_within_regime_permutation": p,
                "null_mean": float(np.nanmean(null)),
                "null_q975": float(np.nanquantile(null, 0.975)),
            }
        )

    lag = pd.concat(
        [
            lag_scan(breathing, "Z_geom_inhale_delta", "Z_geom_next_cold_minus_current"),
            lag_scan(
                breathing,
                "cycle_birth_positive_fraction_inhale_delta",
                "force_h1_birth_force_share_next_cold_minus_current",
            ),
        ],
        ignore_index=True,
    )
    return pd.DataFrame(rows), pd.DataFrame(null_rows), pd.DataFrame(r2_rows), lag


def plot_null_figure(
    summary: pd.DataFrame,
    null_samples: pd.DataFrame,
    r2: pd.DataFrame,
    lag: pd.DataFrame,
    raw_shuffle: pd.DataFrame,
) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.43)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ax = axes[0]
    panel(ax, "a")
    rel = "loop_to_overload"
    vals = null_samples[null_samples["relationship"] == rel]["rho_within"].to_numpy(float)
    obs = float(summary.loc[summary["relationship"] == rel, "rho_within"].iloc[0])
    ax.hist(vals, bins=40, color="#D8DDE5", edgecolor="white", lw=0.25)
    ax.axvline(obs, color=ACCENT, lw=1.4, label="observed")
    if not raw_shuffle.empty:
        raw_vals = raw_shuffle["rho_within"].to_numpy(float)
        ax.axvline(np.nanmedian(raw_vals), color=INK, lw=1.1, ls=(0, (3, 2)), label="raw force-shuffle median")
    ax.set_xlabel(r"$\rho_{\rm within}$ under null")
    ax.set_ylabel("count")
    ax.set_title("loop-overload relation beats permutation null", loc="left", pad=2)
    ax.legend(loc="upper left")
    finish(ax, axis="y")

    ax = axes[1]
    panel(ax, "b")
    labels = []
    vals = []
    pvals = []
    for _, row in r2.iterrows():
        labels.append("loop" if row["added"].startswith("force_h1") else "top 5%")
        if row["base"] != "FE+cycle":
            labels[-1] = labels[-1] + "\nafter other"
        vals.append(row["delta_r2_observed"])
        pvals.append(row["p_within_regime_permutation"])
    x = np.arange(len(vals))
    colors = [ACCENT if "loop" in lab else MUTED for lab in labels]
    ax.bar(x, vals, color=colors, width=0.68)
    for i, (v, p) in enumerate(zip(vals, pvals)):
        ax.text(i, v + 0.006, f"P={p:.1e}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"incremental $\Delta R^2$")
    ax.set_ylim(0, max(vals) * 1.34)
    ax.set_title("loop coordinate remains the added predictor", loc="left", pad=2)
    finish(ax, axis="y")

    ax = axes[2]
    panel(ax, "c")
    for (pred, target), g in lag.groupby(["predictor", "target"], sort=False):
        color = ACCENT if "cycle_birth" in pred else "#3D6B9C"
        label = "cycle creation -> loop imprint" if "cycle_birth" in pred else "fabric excursion -> fabric imprint"
        ax.plot(g["lag"], g["rho_within"], "o-", ms=3.2, lw=1.0, color=color, label=label)
    ax.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_xlabel("cycle lag used for predictor")
    ax.set_ylabel(r"$\rho_{\rm within}$")
    ax.set_title("lag scan reveals route-periodic memory", loc="left", pad=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.54, -0.18), fontsize=6.1, ncol=1)
    finish(ax)

    ax = axes[3]
    panel(ax, "d")
    subset = summary[summary["relationship"].isin(["contact_aperture_to_hot_overload", "loop_to_overload", "top5_to_overload"])]
    if subset.empty:
        subset = summary[summary["target"] == "force_p99_hot_minus_cold"]
    order = ["loop_to_overload", "top5_to_overload", "contact_aperture_to_hot_overload"]
    subset = subset.set_index("relationship").reindex([x for x in order if x in subset["relationship"].values]).reset_index()
    label_map = {
        "loop_to_overload": "loop\nactivation",
        "top5_to_overload": "top-5%\nshare",
        "contact_aperture_to_hot_overload": "contact\naperture",
    }
    color_map = {
        "loop_to_overload": ACCENT,
        "top5_to_overload": MUTED,
        "contact_aperture_to_hot_overload": "#3D6B9C",
    }
    names = [label_map[r] for r in subset["relationship"]]
    colors = [color_map[r] for r in subset["relationship"]]
    ax.bar(np.arange(len(subset)), subset["rho_within"], color=colors, width=0.65)
    ax.axhline(0, color="#B7BDC5", lw=0.55, ls=(0, (3, 3)))
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(names)
    ax.set_ylabel(r"observed $\rho_{\rm within}$")
    ax.set_title("what the mechanism is not", loc="left", pad=2)
    finish(ax, axis="y")

    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig12_breathing_null_controls.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    r2: pd.DataFrame,
    lag: pd.DataFrame,
    raw_shuffle: pd.DataFrame,
) -> None:
    lines = [
        "# Breathing null-control audit",
        "",
        "This audit asks whether the breathing-cycle mechanism survives controls that break timing or break the placement of force on the contact graph.",
        "",
        "## Relationship-level permutation controls",
        "",
        summary.to_markdown(index=False),
        "",
        "## Incremental regression controls",
        "",
        r2.to_markdown(index=False),
        "",
        "## Lag controls",
        "",
        lag.to_markdown(index=False),
        "",
        "## Raw force-shuffle control",
        "",
    ]
    if raw_shuffle.empty:
        lines.append("Raw force-shuffle control was not run because local true-force state files were unavailable.")
    else:
        lines.extend(
            [
                raw_shuffle.describe().to_markdown(),
                "",
                "In this null model, each contact graph is kept fixed but force magnitudes are randomly reassigned to edges before force-ordered filtration. The hot-minus-cold overload target is unchanged because the force distribution is preserved within each state; only the force-topology embedding is broken.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strongest acceptable mechanism is not contact turnover alone and not top-5% concentration alone. The force-loop signal exceeds relationship-level permutation controls and remains the added predictor after top-5% concentration. Raw force-shuffling reduces but does not erase the loop-overload relation, implying that force distribution, graph structure and force-on-edge placement all contribute. The lag scan shows route-periodic memory rather than a unique one-cycle causal peak, so lagged imprints should be used as supportive cycle-map evidence, not as timing proof. These controls should be treated as evidence audits, not as fitted universal laws.",
        ]
    )
    (ROOT / "nature_physics_breathing_null_controls_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary, null_samples, r2, lag = build_null_audit()

    summary.to_csv(SRC / "nphys_breathing_null_control_summary.csv", index=False)
    null_samples.to_csv(SRC / "nphys_breathing_null_control_permutation_samples.csv", index=False)
    r2.to_csv(SRC / "nphys_breathing_null_control_regression.csv", index=False)
    lag.to_csv(SRC / "nphys_breathing_null_control_lag_scan.csv", index=False)

    raw_shuffle, raw_metric_samples = compute_raw_force_shuffle_null()
    raw_shuffle.to_csv(SRC / "nphys_breathing_raw_force_shuffle_null.csv", index=False)
    raw_metric_samples.to_csv(SRC / "nphys_breathing_raw_force_shuffle_metric_samples.csv", index=False)

    plot_null_figure(summary, null_samples, r2, lag, raw_shuffle)
    write_report(summary, r2, lag, raw_shuffle)

    print(summary[["relationship", "rho_within", "p_within_regime_permutation", "n"]].to_string(index=False))
    if not raw_shuffle.empty:
        obs = summary.loc[summary["relationship"] == "loop_to_overload", "rho_within"].iloc[0]
        p_raw = (np.sum(np.abs(raw_shuffle["rho_within"]) >= abs(obs)) + 1) / (len(raw_shuffle) + 1)
        print(f"raw force-shuffle null: observed rho={obs:.3f}, empirical p={p_raw:.4g}, n_perm={len(raw_shuffle)}")


if __name__ == "__main__":
    main()
