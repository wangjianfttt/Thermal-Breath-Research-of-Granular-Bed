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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from mine_long_cycle_true_force_memory import LONG_REGIMES, RUN_BASE
from mine_route_generality_true_force_extension import ROUTES as EXTENSION_ROUTES
from mine_true_force_percolation import ForceState, UnionFind, read_dump, read_local


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
BASE_DELTA = SRC / "nphys_force_loop_conduit_gating_cycle_metrics.csv"
EXTENSION_DELTA = SRC / "nphys_route_generality_true_force_extension_delta.csv"
OVERLOAD_SCALE = 0.003

INK = "#252A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
HOT = "#B6423E"
GOLD = "#D98C3A"
BLUE = "#3D6B9C"
GREEN = "#4F8B67"
VIOLET = "#7E6AAE"
ROUTE_COLORS = {
    "R1": "#345995",
    "R3": "#D98C3A",
    "R5": "#7E6AAE",
    "R6": "#C95F3F",
    "R6c": "#8D3138",
    "G1": "#B55247",
    "G2": "#C79A3A",
    "G3": "#547C7A",
}
MARKERS = {"R1": "o", "R3": "s", "R5": "D", "R6": "^", "R6c": "v", "G1": "P", "G2": "X", "G3": "h"}


@dataclass(frozen=True)
class StateKey:
    tag: str
    run: str
    regime_id: str
    cycle: int
    phase: str


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


def panel(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float, int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 4 or np.nanstd(xx[ok]) == 0 or np.nanstd(yy[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    out = spearmanr(xx[ok], yy[ok])
    return float(out.statistic), float(out.pvalue), int(ok.sum())


def discover_keys() -> list[StateKey]:
    keys: list[StateKey] = []
    for tag, (folder_name, run_name) in LONG_REGIMES.items():
        folder = RUN_BASE / folder_name
        if not folder.exists():
            continue
        for cycle in range(1, 31):
            for phase in ["cold", "hot"]:
                if (folder / f"contacts_cycle_{cycle}_{phase}.local").exists() and (folder / f"cycle_{cycle}_{phase}.dump").exists():
                    regime_id = {"a150_mu060_g000": "R6c", "a150_mu030_g002": "R5"}.get(tag)
                    if regime_id is None:
                        if "a050" in tag:
                            regime_id = "R1"
                        elif "a100" in tag:
                            regime_id = "R3"
                        else:
                            regime_id = "R6"
                    keys.append(StateKey(tag, run_name, regime_id, cycle, phase))
    for tag, (folder_name, run_name) in EXTENSION_ROUTES.items():
        folder = RUN_BASE / folder_name
        if not folder.exists():
            continue
        regime_id = {"a150_mu010_g020": "G1", "a150_mu030_g020": "G2", "a050_mu060_g020": "G3"}[tag]
        for cycle in range(1, 31):
            for phase in ["cold", "hot"]:
                if (folder / f"contacts_cycle_{cycle}_{phase}.local").exists() and (folder / f"cycle_{cycle}_{phase}.dump").exists():
                    keys.append(StateKey(tag, run_name, regime_id, cycle, phase))
    return keys


def read_state(key: StateKey) -> ForceState:
    if key.tag in LONG_REGIMES:
        folder = RUN_BASE / LONG_REGIMES[key.tag][0]
    else:
        folder = RUN_BASE / EXTENSION_ROUTES[key.tag][0]
    return ForceState(
        key.tag,
        key.phase,
        read_local(folder / f"contacts_cycle_{key.cycle}_{key.phase}.local"),
        read_dump(folder / f"cycle_{key.cycle}_{key.phase}.dump"),
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or np.sum(weights) <= 0:
        return np.nan
    return float(np.sum(values * weights) / np.sum(weights))


def entropy_from_bins(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> float:
    if len(values) == 0 or np.sum(weights) <= 0:
        return np.nan
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    p = hist / hist.sum() if hist.sum() > 0 else hist
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(max(len(bins) - 1, 2)))


def state_spatial_metrics(key: StateKey) -> dict[str, float | str | int]:
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
    if d.empty:
        return base

    ids = np.array(sorted(set(d["id1"]).union(set(d["id2"]))))
    id_to_idx = {int(v): i for i, v in enumerate(ids)}
    x = atoms.loc[ids, "x"].to_numpy(float)
    y = atoms.loc[ids, "y"].to_numpy(float)
    z = atoms.loc[ids, "z"].to_numpy(float)
    r = np.sqrt(x**2 + y**2)
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-12)
    r_norm = (r - r.min()) / max(r.max() - r.min(), 1e-12)
    bottom = z_norm <= 0.06
    top = z_norm >= 0.94
    side = r_norm >= 0.94
    uf = UnionFind(len(ids), bottom, top, side)

    idx1 = np.array([id_to_idx[int(v)] for v in d["id1"]], dtype=int)
    idx2 = np.array([id_to_idx[int(v)] for v in d["id2"]], dtype=int)
    edge_z = 0.5 * (z_norm[idx1] + z_norm[idx2])
    edge_r = 0.5 * (r_norm[idx1] + r_norm[idx2])
    edge_boundary = (edge_z <= 0.10) | (edge_z >= 0.90) | (edge_r >= 0.90)
    edge_bottom = edge_z <= 0.12
    edge_side = edge_r >= 0.88

    forces = d["force"].to_numpy(float)
    order = np.argsort(forces)[::-1]
    total_force = float(forces.sum())
    cycle_closing = np.zeros(len(d), dtype=bool)
    conduit_closing = np.zeros(len(d), dtype=bool)
    for row_idx in order:
        row = d.iloc[int(row_idx)]
        a = id_to_idx[int(row["id1"])]
        b = id_to_idx[int(row["id2"])]
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            cycle_closing[int(row_idx)] = True
            root = ra
        else:
            root, _merged = uf.union(a, b)
        conduit_closing[int(row_idx)] = bool(uf.bottom[root] and uf.side[root] and cycle_closing[int(row_idx)])

    loop_force = forces[cycle_closing]
    loop_z = edge_z[cycle_closing]
    loop_r = edge_r[cycle_closing]
    loop_boundary = edge_boundary[cycle_closing]
    loop_bottom = edge_bottom[cycle_closing]
    loop_side = edge_side[cycle_closing]
    conduit_force = forces[conduit_closing]

    base.update(
        {
            "force_sum": total_force,
            "force_p99": float(np.percentile(forces, 99)),
            "loop_edge_fraction": float(cycle_closing.mean()),
            "loop_force_share": float(loop_force.sum() / total_force) if total_force > 0 else np.nan,
            "loop_boundary_force_share": float(loop_force[loop_boundary].sum() / total_force) if total_force > 0 and len(loop_force) else np.nan,
            "loop_bottom_force_share": float(loop_force[loop_bottom].sum() / total_force) if total_force > 0 and len(loop_force) else np.nan,
            "loop_side_force_share": float(loop_force[loop_side].sum() / total_force) if total_force > 0 and len(loop_force) else np.nan,
            "loop_conduit_force_share": float(conduit_force.sum() / total_force) if total_force > 0 else np.nan,
            "loop_force_weighted_z": weighted_mean(loop_z, loop_force),
            "loop_force_weighted_r": weighted_mean(loop_r, loop_force),
            "loop_z_entropy": entropy_from_bins(loop_z, loop_force, np.linspace(0, 1, 9)),
            "loop_r_entropy": entropy_from_bins(loop_r, loop_force, np.linspace(0, 1, 9)),
            "top5_force_share": float(forces[order[: max(1, int(np.ceil(0.05 * len(d))))]].sum() / total_force),
        }
    )
    return base


def build_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "force_p99",
        "loop_edge_fraction",
        "loop_force_share",
        "loop_boundary_force_share",
        "loop_bottom_force_share",
        "loop_side_force_share",
        "loop_conduit_force_share",
        "loop_force_weighted_z",
        "loop_force_weighted_r",
        "loop_z_entropy",
        "loop_r_entropy",
        "top5_force_share",
    ]
    cold = metrics[metrics["phase"] == "cold"].set_index(["run", "tag", "regime_id", "cycle"])
    hot = metrics[metrics["phase"] == "hot"].set_index(["run", "tag", "regime_id", "cycle"])
    rows = []
    for idx in cold.index.intersection(hot.index):
        run, tag, regime_id, cycle = idx
        row: dict[str, float | str | int] = {"run": run, "tag": tag, "regime_id": regime_id, "cycle": int(cycle)}
        for col in cols:
            row[f"{col}_cold"] = float(cold.loc[idx, col])
            row[f"{col}_hot"] = float(hot.loc[idx, col])
            row[f"{col}_delta"] = float(hot.loc[idx, col] - cold.loc[idx, col])
        rows.append(row)
    base = pd.DataFrame(rows).sort_values(["regime_id", "cycle"])
    existing = pd.read_csv(BASE_DELTA)[["tag", "regime_id", "cycle", "overload_asinh", "loop_activation", "force_p99_hot_minus_cold"]].copy()
    existing["dataset"] = "five-route basis"
    if EXTENSION_DELTA.exists():
        ext = pd.read_csv(EXTENSION_DELTA)[["tag", "regime_id", "cycle", "force_p99_hot_minus_cold", "force_h1_birth_force_share_hot_minus_cold"]].copy()
        ext["overload_asinh"] = np.arcsinh(ext["force_p99_hot_minus_cold"] / OVERLOAD_SCALE)
        ext["loop_activation"] = ext["force_h1_birth_force_share_hot_minus_cold"]
        ext["dataset"] = "new-route extension"
        existing = pd.concat([existing, ext[existing.columns]], ignore_index=True)
    return base.merge(existing, on=["tag", "regime_id", "cycle"], how="left")


def correlations(delta: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        ("loop total force share", "loop_force_share_delta"),
        ("loop boundary force share", "loop_boundary_force_share_delta"),
        ("loop bottom force share", "loop_bottom_force_share_delta"),
        ("loop side force share", "loop_side_force_share_delta"),
        ("loop conduit force share", "loop_conduit_force_share_delta"),
        ("loop height centroid", "loop_force_weighted_z_delta"),
        ("loop radial centroid", "loop_force_weighted_r_delta"),
        ("loop vertical entropy", "loop_z_entropy_delta"),
        ("top-5 force share", "top5_force_share_delta"),
    ]
    df = delta.copy()
    df["overload_asinh_rc"] = route_center(df, "overload_asinh")
    rows = []
    for name, col in predictors:
        df[col + "_rc"] = route_center(df, col)
        raw_r, raw_p, n = safe_spearman(df[col], df["overload_asinh"])
        rc_r, rc_p, _n = safe_spearman(df[col + "_rc"], df["overload_asinh_rc"])
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


def leave_one_route_models(delta: pd.DataFrame) -> pd.DataFrame:
    models = {
        "loop total": ["loop_force_share_delta"],
        "boundary loops": ["loop_boundary_force_share_delta"],
        "conduit loops": ["loop_conduit_force_share_delta"],
        "loop + boundary": ["loop_force_share_delta", "loop_boundary_force_share_delta"],
        "loop + spatial centroids": ["loop_force_share_delta", "loop_force_weighted_z_delta", "loop_force_weighted_r_delta"],
        "top5 tail": ["top5_force_share_delta"],
    }
    rows = []
    for name, features in models.items():
        y_all: list[float] = []
        yh_all: list[float] = []
        base_all: list[float] = []
        for rid in sorted(delta["regime_id"].unique()):
            train = delta[delta["regime_id"] != rid].dropna(subset=["overload_asinh", *features])
            test = delta[delta["regime_id"] == rid].dropna(subset=["overload_asinh", *features])
            scaler = StandardScaler().fit(train[features].to_numpy(float))
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(
                scaler.transform(train[features].to_numpy(float)), train["overload_asinh"].to_numpy(float)
            )
            pred = model.predict(scaler.transform(test[features].to_numpy(float)))
            y_all.extend(test["overload_asinh"].to_numpy(float))
            yh_all.extend(pred)
            base_all.extend(np.repeat(float(train["overload_asinh"].mean()), len(test)))
        y = np.asarray(y_all)
        yh = np.asarray(yh_all)
        base = np.asarray(base_all)
        rho, p, _n = safe_spearman(y, yh)
        rows.append(
            {
                "model": name,
                "features": ";".join(features),
                "validation": "leave_one_route_out",
                "n": int(len(y)),
                "r2_vs_training_mean": float(1 - np.sum((y - yh) ** 2) / np.sum((y - base) ** 2)),
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
            mean_loop_force_share_delta=("loop_force_share_delta", "mean"),
            mean_loop_boundary_force_share_delta=("loop_boundary_force_share_delta", "mean"),
            mean_loop_conduit_force_share_delta=("loop_conduit_force_share_delta", "mean"),
            mean_top5_force_share_delta=("top5_force_share_delta", "mean"),
        )
        .sort_values("mean_overload_asinh")
    )


def make_figure(delta: pd.DataFrame, corr: pd.DataFrame, models: pd.DataFrame, routes: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 0.92], height_ratios=[1.0, 0.9])

    ax = fig.add_subplot(gs[0, 0])
    for rid, g in delta.groupby("regime_id", sort=True):
        ax.scatter(
            g["loop_conduit_force_share_delta"],
            g["overload_asinh"],
            s=24,
            color=ROUTE_COLORS.get(rid, INK),
            marker=MARKERS.get(rid, "o"),
            edgecolor="white",
            lw=0.35,
            alpha=0.90,
            label=rid,
        )
    ax.axvline(0, color="#AEB6C0", lw=0.75, ls=(0, (3, 3)))
    ax.axhline(0, color="#AEB6C0", lw=0.75, ls=(0, (3, 3)))
    ax.set_xlabel("hot-cold conduit-loop force-share")
    ax.set_ylabel("asinh overload")
    ax.set_title("wall-coupled loops mark the dangerous branch across routes", loc="left", pad=4)
    ax.legend(fontsize=5.8, ncol=3, loc="upper left")
    finish(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    order = [
        "loop total force share",
        "loop boundary force share",
        "loop bottom force share",
        "loop side force share",
        "loop conduit force share",
        "loop radial centroid",
        "top-5 force share",
    ]
    cc = corr.set_index("predictor").loc[order].reset_index()
    y = np.arange(len(cc))
    colors = [HOT, HOT, GOLD, BLUE, VIOLET, GREEN, "#6F7C8A"]
    ax.barh(y, cc["spearman_route_centered"], color=colors, alpha=0.90)
    ax.axvline(0, color="#AEB6C0", lw=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(cc["predictor"], fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlim(-0.62, 0.98)
    ax.set_xlabel("route-centred Spearman")
    ax.set_title("eight-route spatial loop fingerprint", loc="left", pad=4)
    finish(ax, "x")
    panel(ax, "b", x=-0.18)

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(routes))
    ax.plot(x, routes["mean_loop_force_share_delta"], color=HOT, marker="o", lw=1.2, label="all loops")
    ax.plot(x, routes["mean_loop_conduit_force_share_delta"], color=VIOLET, marker="s", lw=1.2, label="conduit loops")
    ax.plot(x, routes["mean_top5_force_share_delta"], color="#6F7C8A", marker="^", lw=1.2, label="top-5%")
    ax.set_xticks(x, routes["regime_id"])
    ax.set_ylabel("mean hot-cold force-share")
    ax.set_title("route severity follows the loop sector", loc="left", pad=4)
    ax.legend(fontsize=6.0, ncol=3, loc="upper left")
    finish(ax, "y")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    model_order = ["loop total", "boundary loops", "conduit loops", "loop + boundary", "loop + spatial centroids", "top5 tail"]
    mm = models.set_index("model").loc[model_order].reset_index()
    x = np.arange(len(mm))
    ax.bar(x, mm["r2_vs_training_mean"], color=[HOT, GOLD, VIOLET, HOT, GREEN, "#6F7C8A"], width=0.68)
    ax.axhline(0, color="#AEB6C0", lw=0.75)
    ax.set_xticks(x, ["loops", "boundary", "conduit", "loop+\nboundary", "loop+\ncentroids", "top-5%"], fontsize=5.6)
    ax.set_ylabel(r"leave-route-out $R^2$")
    ax.set_ylim(-0.25, 0.96)
    ax.set_title("spatial features bound, not replace", loc="left", pad=4)
    finish(ax, "y")
    panel(ax, "d", x=-0.18)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig48_force_loop_spatial_fingerprint.{ext}", **kwargs)
    plt.close(fig)


def write_report(corr: pd.DataFrame, models: pd.DataFrame, routes: pd.DataFrame) -> None:
    def corr_val(name: str) -> float:
        return float(corr.loc[corr["predictor"] == name, "spearman_route_centered"].iloc[0])

    def model_val(name: str) -> float:
        return float(models.loc[models["model"] == name, "r2_vs_training_mean"].iloc[0])

    lines = [
        "# Force-loop spatial-fingerprint audit",
        "",
        "This reserve audit tests whether the force-loop mechanism has a reproducible spatial fingerprint across all eight true-force routes, rather than being visible only in representative snapshots or in the original five-route basis.",
        "",
        "## Spatial correlations",
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
        f"The spatial audit supports a bounded refinement of the loop mechanism. The wall-coupled conduit subset of cycle-closing loop force is associated with overload within routes (rho={corr_val('loop conduit force share'):.3f}), whereas generic boundary-loop force is weak and oppositely signed (rho={corr_val('loop boundary force share'):.3f}). The all-loop force share remains the stronger and more transferable coordinate (rho={corr_val('loop total force share'):.3f}; leave-one-route R2={model_val('loop total'):.3f}).",
        "",
        f"Spatial variables therefore make the mechanism inspectable without replacing it: boundary loops alone give leave-one-route R2={model_val('boundary loops'):.3f}, conduit loops give R2={model_val('conduit loops'):.3f}, and loop plus boundary features give R2={model_val('loop + boundary'):.3f}. The top-5% force-tail control remains weaker and oppositely signed within routes.",
        "",
        "Allowed wording: dangerous hot states enrich force-carrying graph loops near boundary-coupled parts of the force network, but the transferable overload coordinate is still the total cycle-closing loop sector. Not allowed: the data do not establish a universal spatial morphology, single wall-spanning chain law or boundary-loop replacement for the force-loop coordinate.",
    ]
    (ROOT / "nature_physics_force_loop_spatial_fingerprint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    keys = discover_keys()
    metrics = pd.DataFrame([state_spatial_metrics(k) for k in keys]).sort_values(["regime_id", "cycle", "phase"])
    delta = build_delta(metrics)
    corr = correlations(delta)
    models = leave_one_route_models(delta)
    routes = route_summary(delta)
    metrics.to_csv(SRC / "nphys_force_loop_spatial_fingerprint_state_metrics.csv", index=False)
    delta.to_csv(SRC / "nphys_force_loop_spatial_fingerprint_cycle_delta.csv", index=False)
    corr.to_csv(SRC / "nphys_force_loop_spatial_fingerprint_correlations.csv", index=False)
    models.to_csv(SRC / "nphys_force_loop_spatial_fingerprint_transfer_tests.csv", index=False)
    routes.to_csv(SRC / "nphys_force_loop_spatial_fingerprint_route_summary.csv", index=False)
    make_figure(delta, corr, models, routes)
    write_report(corr, models, routes)
    print(f"states={len(metrics)} pairs={len(delta)}")
    print(corr[["predictor", "spearman_route_centered", "p_route_centered"]].round(3).to_string(index=False))
    print(models[["model", "r2_vs_training_mean", "spearman_y_yhat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
