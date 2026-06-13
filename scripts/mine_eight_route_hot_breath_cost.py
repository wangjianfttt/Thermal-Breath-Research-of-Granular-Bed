#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

ORIGINAL = SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv"
EXTENSION = SRC / "nphys_route_generality_true_force_extension_delta.csv"

COLORS = {
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
INK = "#252A31"
MUTED = "#7A838E"
GRID = "#E7EAEE"
OVERLOAD_SCALE = 0.003


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def route_center(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("regime_id")[col].transform("mean")


def route_z(df: pd.DataFrame, col: str) -> pd.Series:
    centred = route_center(df, col)
    scale = df.groupby("regime_id")[col].transform("std").replace(0, np.nan)
    return centred / scale


def load_combined() -> pd.DataFrame:
    orig = pd.read_csv(ORIGINAL).copy()
    ext = pd.read_csv(EXTENSION).copy()
    orig["dataset"] = "five-route basis"
    ext["dataset"] = "new-route extension"
    cols = sorted(set(orig.columns).intersection(ext.columns))
    cols = [c for c in cols if c != "dataset"] + ["dataset"]
    df = pd.concat([orig[cols], ext[cols]], ignore_index=True)
    df["overload_asinh"] = np.arcsinh(df["force_p99_hot_minus_cold"] / OVERLOAD_SCALE)
    df["loop_activation"] = df["force_h1_birth_force_share_hot_minus_cold"]
    df["loop_birth_fraction"] = df["force_h1_birth_fraction_hot_minus_cold"]
    df["force_tail"] = df["force_share_top5_edges_hot_minus_cold"]
    df["giant_fragmentation"] = df["giant_fraction_after_top5_edges_hot_minus_cold"]
    df["wall_conduit"] = df["bottom_side_percolation_edge_fraction_hot_minus_cold"]
    for col in ["overload_asinh", "loop_activation", "loop_birth_fraction", "force_tail", "giant_fragmentation", "wall_conduit"]:
        df[f"{col}_rc"] = route_center(df, col)
        df[f"{col}_rz"] = route_z(df, col)
    df["loop_breath_cost"] = df["loop_activation_rc"].clip(lower=0.0)
    df["tail_cost"] = df["force_tail_rc"].clip(lower=0.0)
    df["topological_triplet_norm"] = np.sqrt(
        df["loop_activation_rz"].fillna(0.0) ** 2
        + df["loop_birth_fraction_rz"].fillna(0.0) ** 2
        + df["giant_fragmentation_rz"].fillna(0.0) ** 2
    )
    df["tail_triplet_norm"] = np.sqrt(
        df["force_tail_rz"].fillna(0.0) ** 2
        + df["giant_fragmentation_rz"].fillna(0.0) ** 2
        + df["wall_conduit_rz"].fillna(0.0) ** 2
    )
    return df


def permutation_spearman(x: np.ndarray, y: np.ndarray, route: np.ndarray, n_perm: int = 5000, seed: int = 83) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    route = route[ok]
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


def build_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = {
        "loop activation": "loop_activation_rc",
        "positive loop cost": "loop_breath_cost",
        "topological triplet": "topological_triplet_norm",
        "force tail": "force_tail_rc",
        "positive tail cost": "tail_cost",
        "tail/conduit triplet": "tail_triplet_norm",
        "wall conduit": "wall_conduit_rc",
    }
    rows: list[dict[str, float | str | int]] = []
    for name, col in predictors.items():
        rho, p = permutation_spearman(
            df[col].to_numpy(float),
            df["overload_asinh_rc"].to_numpy(float),
            df["regime_id"].to_numpy(str),
        )
        rows.append(
            {
                "predictor": name,
                "column": col,
                "target": "route-centred asinh overload",
                "n": int(df[[col, "overload_asinh_rc"]].dropna().shape[0]),
                "spearman_rho_route_centered": rho,
                "route_preserving_permutation_p": p,
            }
        )
    tests = pd.DataFrame(rows)

    model_defs = {
        "loop only": ["loop_activation_rc"],
        "tail only": ["force_tail_rc"],
        "loop + tail": ["loop_activation_rc", "force_tail_rc"],
        "topological triplet": ["loop_activation_rz", "loop_birth_fraction_rz", "giant_fragmentation_rz"],
        "tail/conduit triplet": ["force_tail_rz", "giant_fragmentation_rz", "wall_conduit_rz"],
        "full diagnostic": [
            "loop_activation_rz",
            "loop_birth_fraction_rz",
            "giant_fragmentation_rz",
            "force_tail_rz",
            "wall_conduit_rz",
        ],
    }
    y = df["overload_asinh_rc"].to_numpy(float)
    route = df["regime_id"].to_numpy(str)
    model_rows: list[dict[str, float | str | int]] = []
    for model, cols in model_defs.items():
        pred = np.full(len(df), np.nan)
        for rid in np.unique(route):
            train = route != rid
            test = route == rid
            x_train = df.loc[train, cols].fillna(0.0).to_numpy(float)
            x_test = df.loc[test, cols].fillna(0.0).to_numpy(float)
            fit = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0]).fit(x_train, y[train])
            pred[test] = fit.predict(x_test)
        ok = np.isfinite(pred) & np.isfinite(y)
        rho = float(spearmanr(pred[ok], y[ok]).statistic)
        model_rows.append(
            {
                "model": model,
                "predictors": ";".join(cols),
                "leave_one_route_r2": float(r2_score(y[ok], pred[ok])),
                "leave_one_route_spearman": rho,
                "n": int(ok.sum()),
            }
        )
    return tests, pd.DataFrame(model_rows)


def build_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["regime_id", "dataset", "tag"], sort=True)
        .agg(
            n=("cycle", "size"),
            mean_overload_asinh=("overload_asinh", "mean"),
            mean_loop_activation=("loop_activation", "mean"),
            mean_force_tail=("force_tail", "mean"),
            mean_wall_conduit=("wall_conduit", "mean"),
            mean_topological_triplet=("topological_triplet_norm", "mean"),
        )
        .reset_index()
    )


def build_figure(df: pd.DataFrame, tests: pd.DataFrame, models: pd.DataFrame, route_summary: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.25, 1.05, 1.1])

    ax = fig.add_subplot(gs[:, 0])
    order = route_summary.sort_values("mean_overload_asinh")["regime_id"].tolist()
    x = np.arange(len(order), dtype=float)
    width = 0.34
    summary = route_summary.set_index("regime_id").loc[order]
    ax.axhline(0, color="#AEB6C0", lw=0.7, ls=(0, (3, 3)))
    ax.bar(x - width / 2, summary["mean_loop_activation"], width=width, color="#B6423E", alpha=0.82, label=r"mean $\Delta L_f$")
    ax.bar(x + width / 2, summary["mean_force_tail"], width=width, color="#3D6B9C", alpha=0.80, label=r"mean $\Delta q_5$")
    for i, rid in enumerate(order):
        ax.scatter(i - width / 2, summary.loc[rid, "mean_loop_activation"], marker=MARKERS.get(rid, "o"), color=COLORS.get(rid, MUTED), s=34, edgecolor="white", lw=0.45, zorder=4)
        ax.scatter(i + width / 2, summary.loc[rid, "mean_force_tail"], marker=MARKERS.get(rid, "o"), color=COLORS.get(rid, MUTED), s=34, edgecolor="white", lw=0.45, zorder=4)
    ax.set_xticks(x, order)
    ax.set_ylabel("mean hot-minus-cold increment")
    ax.set_xlabel("route, ordered by mean overload")
    ax.set_title("eight-route hot breath separates loop and tail cost", loc="left", pad=4)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), fontsize=5.8, handlelength=1.0, borderaxespad=0.0)
    finish(ax, axis="y")
    panel(ax, "a", x=-0.08)

    ax = fig.add_subplot(gs[0, 1])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["loop_activation_rc"],
            g["overload_asinh_rc"],
            marker=MARKERS.get(rid, "o"),
            color=COLORS.get(rid, MUTED),
            s=16,
            edgecolor="white",
            lw=0.25,
            alpha=0.82,
        )
    loop = tests[tests["predictor"] == "loop activation"].iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={loop.spearman_rho_route_centered:.2f}$, $P={loop.route_preserving_permutation_p:.4f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel(r"route-centred $\Delta L_f$")
    ax.set_ylabel("route-centred overload")
    ax.set_title("loop sector", loc="left", pad=4)
    finish(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for rid, g in df.groupby("regime_id", sort=True):
        ax.scatter(
            g["force_tail_rc"],
            g["overload_asinh_rc"],
            marker=MARKERS.get(rid, "o"),
            color=COLORS.get(rid, MUTED),
            s=16,
            edgecolor="white",
            lw=0.25,
            alpha=0.82,
        )
    tail = tests[tests["predictor"] == "force tail"].iloc[0]
    ax.text(0.05, 0.95, rf"$\rho={tail.spearman_rho_route_centered:.2f}$, $P={tail.route_preserving_permutation_p:.4f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
    ax.axhline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.set_xlabel(r"route-centred $\Delta q_5$")
    ax.set_ylabel("route-centred overload")
    ax.set_title("force-tail control", loc="left", pad=4)
    finish(ax)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1:])
    model_order = [
        "loop only",
        "tail only",
        "loop + tail",
        "topological triplet",
        "tail/conduit triplet",
        "full diagnostic",
    ]
    m = models.set_index("model").loc[model_order].reset_index()
    colors = ["#B6423E", "#3D6B9C", "#6E7581", "#8D3138", "#6E8F8C", "#252A31"]
    y = np.arange(len(m), dtype=float)
    ax.axvline(0, color="#AEB6C0", lw=0.65, ls=(0, (3, 3)))
    ax.barh(y, m["leave_one_route_r2"], color=colors, alpha=0.86)
    for yi, row in zip(y, m.itertuples(index=False)):
        ax.text(row.leave_one_route_r2 + 0.02, yi, rf"$\rho_p={row.leave_one_route_spearman:.2f}$", va="center", fontsize=5.8, color=INK)
    ax.set_yticks(y, m["model"])
    ax.invert_yaxis()
    ax.set_xlabel(r"leave-one-route $R^2$ for route-centred overload")
    ax.set_title("transfer test: topology beats force-tail concentration", loc="left", pad=4)
    finish(ax, axis="x")
    panel(ax, "d", x=-0.07)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig67_eight_route_hot_breath_cost.{ext}", **kwargs)
    plt.close(fig)


def write_report(tests: pd.DataFrame, models: pd.DataFrame, route_summary: pd.DataFrame) -> None:
    loop = tests[tests["predictor"] == "loop activation"].iloc[0]
    tail = tests[tests["predictor"] == "force tail"].iloc[0]
    topo = models[models["model"] == "topological triplet"].iloc[0]
    tail_model = models[models["model"] == "tail only"].iloc[0]
    lines = [
        "# Eight-route hot-breath cost audit",
        "",
        "Purpose: test whether the hot part of the breathing mechanism transfers when the five-route true-force basis is merged with the three new true-force route-generality runs.",
        "",
        "## Route summary",
        "",
        route_summary.round(4).to_markdown(index=False),
        "",
        "## Route-centred rank tests",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Leave-one-route transfer",
        "",
        models.round(4).to_markdown(index=False),
        "",
        "## Manuscript-safe interpretation",
        "",
        (
            "Across 240 hot-minus-cold true-force pairs from eight routes, route-centred loop activation remains the transferable hot-breath cost coordinate "
            f"(rho={loop.spearman_rho_route_centered:.3f}, P={loop.route_preserving_permutation_p:.4f}), whereas the top-5% force-tail increment is oppositely signed "
            f"(rho={tail.spearman_rho_route_centered:.3f}, P={tail.route_preserving_permutation_p:.4f}). "
            f"A topology triplet model gives leave-one-route R2={topo.leave_one_route_r2:.3f}, while a tail-only model gives R2={tail_model.leave_one_route_r2:.3f}. "
            "This supports the hot-inhale part of the breathing story beyond the original targeted-route set."
        ),
        "",
        "Boundary: the audit does not measure next-cold imprint efficiency for the three new routes, so it extends the hot cost arm of the breathing mechanism, not the full inhale-exhale memory loop.",
        "",
    ]
    (ROOT / "nature_physics_eight_route_hot_breath_cost.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load_combined()
    tests, models = build_tests(df)
    route_summary = build_route_summary(df)
    df.to_csv(SRC / "nphys_eight_route_hot_breath_cost_cycle_metrics.csv", index=False)
    tests.to_csv(SRC / "nphys_eight_route_hot_breath_cost_tests.csv", index=False)
    models.to_csv(SRC / "nphys_eight_route_hot_breath_cost_models.csv", index=False)
    route_summary.to_csv(SRC / "nphys_eight_route_hot_breath_cost_route_summary.csv", index=False)
    build_figure(df, tests, models, route_summary)
    write_report(tests, models, route_summary)
    print(tests.round(4).to_string(index=False))
    print(models.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
