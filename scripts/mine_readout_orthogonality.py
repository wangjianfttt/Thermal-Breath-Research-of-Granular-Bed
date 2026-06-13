#!/usr/bin/env python3
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

COLD = "#345995"
HOT = "#C95F3F"
INK = "#252A31"
NEUTRAL = "#8B929A"
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


def zscore(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    sd = np.std(arr, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return arr * 0.0
    return (arr - np.mean(arr)) / sd


def panel(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided exact permutation P for n <= 8 without assuming asymptotic normality."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    obs = float(spearmanr(x, y).statistic)
    if len(x) > 8:
        return obs, float(spearmanr(x, y).pvalue)
    vals = []
    for perm in permutations(y):
        vals.append(float(spearmanr(x, np.asarray(perm)).statistic))
    vals = np.asarray(vals)
    p = float(np.mean(np.abs(vals) >= abs(obs) - 1e-12))
    return obs, p


def regression_coefficients(df: pd.DataFrame, target: str) -> tuple[np.ndarray, float]:
    x = np.column_stack([zscore(df["fabric_memory_index"]), zscore(df["loop_activation_index"])])
    y = zscore(df[target])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    pred = x @ beta
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return beta, r2


def loo_predict(df: pd.DataFrame, predictors: list[str], target: str) -> tuple[float, float]:
    y = zscore(df[target])
    pred = np.full(len(df), np.nan)
    for holdout in range(len(df)):
        train = np.arange(len(df)) != holdout
        x_train = np.column_stack([zscore(df.loc[train, p]) for p in predictors])
        x_test = np.column_stack([zscore(df[p]) for p in predictors])[holdout : holdout + 1]
        y_train = zscore(df.loc[train, target])
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        pred[holdout] = (x_test @ beta).item()
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    rho = float(spearmanr(y, pred).statistic)
    return r2, rho


def make_tests(six: pd.DataFrame, long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []

    channel_pairs = [
        ("six_regime", "matched_cold", "fabric_memory_index", "cold_load"),
        ("six_regime", "matched_hot", "loop_activation_index", "hot_overload_force_delta"),
        ("six_regime", "cross_fabric_to_hot", "fabric_memory_index", "hot_overload_force_delta"),
        ("six_regime", "cross_loop_to_cold", "loop_activation_index", "cold_load"),
        ("six_regime", "hot_load_fabric", "fabric_memory_index", "hot_load"),
        ("six_regime", "hot_load_loop", "loop_activation_index", "hot_load"),
    ]
    for scope, test, predictor, target in channel_pairs:
        rho, p = exact_spearman_p(six[predictor].to_numpy(float), six[target].to_numpy(float))
        pr = pearsonr(six[predictor], six[target])
        rows.append(
            {
                "scope": scope,
                "test": test,
                "predictor": predictor,
                "target": target,
                "n": len(six),
                "spearman_rho": rho,
                "spearman_p_exact": p,
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
            }
        )

    centered = long_df.copy()
    for col in centered.select_dtypes("number").columns:
        centered[col] = centered[col] - centered.groupby("regime_id")[col].transform("mean")

    long_pairs = [
        ("long_cycle_within_route", "matched_loop_to_overload", "loop_activation", "overload_delta"),
        ("long_cycle_within_route", "matched_rewrite_to_overload", "thermal_rewrite", "overload_delta"),
        ("long_cycle_within_route", "cross_fabric_to_overload", "fabric_reservoir", "overload_delta"),
        ("long_cycle_within_route", "cross_loop_to_cold_force", "loop_activation", "force_p99_cold"),
        ("long_cycle_within_route", "fabric_to_cold_force", "fabric_reservoir", "force_p99_cold"),
    ]
    for scope, test, predictor, target in long_pairs:
        sp = spearmanr(centered[predictor], centered[target])
        pr = pearsonr(centered[predictor], centered[target])
        rows.append(
            {
                "scope": scope,
                "test": test,
                "predictor": predictor,
                "target": target,
                "n": len(centered),
                "spearman_rho": float(sp.statistic),
                "spearman_p_exact": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
            }
        )

    cold_beta, cold_r2 = regression_coefficients(six, "cold_load")
    hot_beta, hot_r2 = regression_coefficients(six, "hot_overload_force_delta")
    hot_load_beta, hot_load_r2 = regression_coefficients(six, "hot_load")
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(cold_beta, hot_beta) / (np.linalg.norm(cold_beta) * np.linalg.norm(hot_beta)),
                    -1,
                    1,
                )
            )
        )
    )
    reg_rows = [
        {
            "scope": "six_regime",
            "target": "cold_load",
            "beta_fabric": cold_beta[0],
            "beta_loop": cold_beta[1],
            "dominant_channel": "fabric",
            "r2": cold_r2,
        },
        {
            "scope": "six_regime",
            "target": "hot_overload_force_delta",
            "beta_fabric": hot_beta[0],
            "beta_loop": hot_beta[1],
            "dominant_channel": "loop",
            "r2": hot_r2,
        },
        {
            "scope": "six_regime",
            "target": "hot_load",
            "beta_fabric": hot_load_beta[0],
            "beta_loop": hot_load_beta[1],
            "dominant_channel": "mixed",
            "r2": hot_load_r2,
        },
        {
            "scope": "six_regime",
            "target": "cold_hot_gradient_angle",
            "beta_fabric": np.nan,
            "beta_loop": np.nan,
            "dominant_channel": "angle_degrees",
            "r2": angle,
        },
    ]
    for target, matched, cross in [
        ("cold_load", ["fabric_memory_index"], ["loop_activation_index"]),
        ("hot_overload_force_delta", ["loop_activation_index"], ["fabric_memory_index"]),
    ]:
        for label, predictors in [("matched", matched), ("cross", cross), ("two_channel", ["fabric_memory_index", "loop_activation_index"])]:
            r2, rho = loo_predict(six, predictors, target)
            reg_rows.append(
                {
                    "scope": "six_regime_loo",
                    "target": f"{target}_{label}",
                    "beta_fabric": np.nan,
                    "beta_loop": np.nan,
                    "dominant_channel": "+".join(predictors),
                    "r2": r2,
                    "loo_spearman_rho": rho,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(reg_rows)


def bootstrap_angle(six: pd.DataFrame, n_boot: int = 10000, seed: int = 119) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_boot):
        idx = rng.integers(0, len(six), len(six))
        sample = six.iloc[idx].reset_index(drop=True)
        if sample[["fabric_memory_index", "loop_activation_index"]].drop_duplicates().shape[0] < 3:
            continue
        cold_beta, cold_r2 = regression_coefficients(sample, "cold_load")
        hot_beta, hot_r2 = regression_coefficients(sample, "hot_overload_force_delta")
        denom = np.linalg.norm(cold_beta) * np.linalg.norm(hot_beta)
        if denom == 0 or not np.isfinite(denom):
            continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(cold_beta, hot_beta) / denom, -1, 1))))
        rows.append({"bootstrap_id": i, "angle_degrees": angle, "cold_r2": cold_r2, "hot_r2": hot_r2})
    return pd.DataFrame(rows)


def build_figure(six: pd.DataFrame, long_df: pd.DataFrame, tests: pd.DataFrame, regression: pd.DataFrame, boot: pd.DataFrame) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    x = zscore(six["fabric_memory_index"])
    y = zscore(six["loop_activation_index"])
    sizes = 70 + 260 * (six["cold_load"] - six["cold_load"].min()) / (six["cold_load"].max() - six["cold_load"].min())
    sc = ax.scatter(x, y, s=sizes, c=six["hot_overload_force_delta"], cmap="coolwarm", edgecolor="white", lw=0.7, zorder=4)
    for pos, (_, row) in enumerate(six.iterrows()):
        ax.text(x[pos] + 0.09, y[pos], row["regime_id"], fontsize=6.2, va="center")
    cold_beta = regression.query("target == 'cold_load'").iloc[0][["beta_fabric", "beta_loop"]].to_numpy(float)
    hot_beta = regression.query("target == 'hot_overload_force_delta'").iloc[0][["beta_fabric", "beta_loop"]].to_numpy(float)
    origin = np.array([-1.85, -2.08])
    for beta, color, label, off in [(cold_beta, COLD, "cold gradient", 0.0), (hot_beta, HOT, "hot-overload gradient", 0.12)]:
        vec = beta / np.linalg.norm(beta) * 0.95
        ax.arrow(origin[0], origin[1] + off, vec[0], vec[1], color=color, width=0.015, head_width=0.12, length_includes_head=True)
        ax.text(origin[0] + vec[0] * 1.18, origin[1] + off + vec[1] * 1.18, label, color=color, fontsize=6.3, ha="center")
    angle = regression.query("target == 'cold_hot_gradient_angle'")["r2"].iloc[0]
    ax.text(
        0.96,
        0.96,
        rf"gradient angle $={angle:.0f}^\circ$",
        transform=ax.transAxes,
        fontsize=7.1,
        va="top",
        ha="right",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )
    ax.set_xlabel("fabric coordinate, $z(M)$")
    ax.set_ylabel("loop coordinate, $z(\\Lambda)$")
    ax.set_xlim(-2.25, 2.05)
    ax.set_ylim(-2.35, 1.68)
    ax.set_title("two nearly orthogonal projections", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "a")
    cb = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label(r"hot overload, $\Omega$", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.7)

    ax = fig.add_subplot(gs[0, 1])
    bars = [
        tests.query("test == 'matched_cold'")["spearman_rho"].iloc[0],
        tests.query("test == 'cross_loop_to_cold'")["spearman_rho"].iloc[0],
        tests.query("test == 'matched_hot'")["spearman_rho"].iloc[0],
        tests.query("test == 'cross_fabric_to_hot'")["spearman_rho"].iloc[0],
    ]
    labels = ["M→cold", "Λ→cold", "Λ→hot", "M→hot"]
    colors = [COLD, NEUTRAL, HOT, NEUTRAL]
    ax.axhline(0, color="#B8BDC4", lw=0.7)
    ax.bar(np.arange(4), bars, color=colors, width=0.68)
    ax.set_xticks(np.arange(4), labels, rotation=35, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Spearman rho")
    ax.set_title("matched vs crossed channels", fontsize=7.5, pad=5)
    finish(ax, axis="y")
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    ordered = ["cold_load", "hot_overload_force_delta", "hot_load"]
    mat = regression.set_index("target").loc[ordered, ["beta_fabric", "beta_loop"]].to_numpy(float)
    im = ax.imshow(mat, cmap="PuOr", vmin=-1.35, vmax=1.35, aspect="auto")
    ax.set_yticks(np.arange(3), ["cold load", "hot overload", "hot load"])
    ax.set_xticks(np.arange(2), ["fabric", "loop"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.4, color=INK)
    ax.set_title("standardized projection weights", fontsize=7.5, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel(ax, "c")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5.7)

    ax = fig.add_subplot(gs[1, 1])
    centered = long_df.copy()
    for col in centered.select_dtypes("number").columns:
        centered[col] = centered[col] - centered.groupby("regime_id")[col].transform("mean")
    ax.scatter(centered["loop_activation"], centered["overload_delta"], s=15, color=HOT, alpha=0.72, edgecolor="white", lw=0.25)
    rho = tests.query("test == 'matched_loop_to_overload'")["spearman_rho"].iloc[0]
    ax.text(0.05, 0.94, rf"$\rho={rho:.2f}$", transform=ax.transAxes, va="top", color=HOT, fontsize=6.8)
    ax.set_xlabel(r"within-route $\Delta L_f$")
    ax.set_ylabel(r"within-route $\Omega$")
    ax.set_title("cycle-resolved hot channel", fontsize=7.5, pad=5)
    finish(ax)
    panel(ax, "d")

    ax = fig.add_subplot(gs[1, 2])
    vals = boot["angle_degrees"].to_numpy(float)
    ax.hist(vals, bins=np.linspace(0, 180, 31), color="#B9B0C9", edgecolor="white", lw=0.35)
    q = np.nanpercentile(vals, [2.5, 50, 97.5])
    ax.axvline(q[1], color="#6D4C8D", lw=1.0)
    ax.axvspan(q[0], q[2], color="#6D4C8D", alpha=0.15, lw=0)
    ax.text(0.05, 0.94, f"median {q[1]:.0f}°\\n95% CI {q[0]:.0f}-{q[2]:.0f}°", transform=ax.transAxes, va="top", fontsize=6.6, color=INK)
    ax.set_xlabel("bootstrap gradient angle")
    ax.set_ylabel("count")
    ax.set_title("finite-sample stability", fontsize=7.5, pad=5)
    finish(ax, axis="y")
    panel(ax, "e")

    out = FIG / "nphys_fig28_readout_orthogonality"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(tests: pd.DataFrame, regression: pd.DataFrame, boot: pd.DataFrame) -> None:
    get = lambda name, col: tests.query("test == @name")[col].iloc[0]
    angle = regression.query("target == 'cold_hot_gradient_angle'")["r2"].iloc[0]
    q = np.nanpercentile(boot["angle_degrees"], [2.5, 50, 97.5])
    report = f"""# Readout-orthogonality audit

Date: 2026-06-12

## Question

The manuscript claims that cold retained load and hot overload are different readouts of one trained contact-network state. A skeptical alternative is that both are merely monotonic responses to one hidden scalar. This audit tests that alternative by comparing matched and crossed projections in the existing six-regime ensemble and in the 90 paired long-cycle states.

## Main result

In the six-regime state space, cold load is almost purely aligned with the fabric coordinate, whereas hot overload is almost purely aligned with the loop-activation coordinate. The standardized gradients are separated by {angle:.1f} degrees. Bootstrap resampling gives a median angle of {q[1]:.1f} degrees with a 95% interval of {q[0]:.1f}--{q[2]:.1f} degrees.

Matched rank correlations are strong: fabric to cold load has Spearman rho = {get('matched_cold','spearman_rho'):.3f} (exact P = {get('matched_cold','spearman_p_exact'):.4f}, n = 6), and loop activation to hot overload has rho = {get('matched_hot','spearman_rho'):.3f} (exact P = {get('matched_hot','spearman_p_exact'):.4f}, n = 6). Crossed channels have the opposite sign: loop activation to cold load has rho = {get('cross_loop_to_cold','spearman_rho'):.3f}, and fabric to hot overload has rho = {get('cross_fabric_to_hot','spearman_rho'):.3f}. Thus the matched channels do not simply duplicate the same scalar readout.

The long-cycle audit is consistent with this interpretation but also sets the boundary. Within routes, loop activation predicts hot overload with Spearman rho = {get('matched_loop_to_overload','spearman_rho'):.3f} (n = 90), while fabric reservoir leakage into hot overload is weaker and negative (rho = {get('cross_fabric_to_overload','spearman_rho'):.3f}). The fabric coordinate is not a strong cycle-resolved predictor of the cold force tail in this reduced subset (rho = {get('fabric_to_cold_force','spearman_rho'):.3f}, P = {get('fabric_to_cold_force','spearman_p_exact'):.3g}), so the robust cold-channel evidence should remain the six-regime retained-load ensemble rather than a cycle-resolved force-tail claim.

## Interpretation allowed in the manuscript

Allowed: cold retained load and hot overload are two phase-dependent projections of a trained state; in the reduced two-coordinate diagnostic, the cold and overload gradients are nearly orthogonal.

Not allowed: the two-coordinate map is a universal constitutive law, or hot load itself is only a pure loop coordinate. Direct hot load is mixed, whereas the force-overload increment is the cleaner loop-sector readout.

## Generated files

- `figures/nphys_fig28_readout_orthogonality.*`
- `source_data/nphys_readout_orthogonality_tests.csv`
- `source_data/nphys_readout_orthogonality_regression.csv`
- `source_data/nphys_readout_orthogonality_bootstrap.csv`
"""
    (ROOT / "nature_physics_readout_orthogonality.md").write_text(report, encoding="utf-8")


def main() -> None:
    six = pd.read_csv(SRC / "nphys_unified_projection_six_regime.csv")
    long_df = pd.read_csv(SRC / "nphys_unified_projection_long_cycle.csv")
    tests, regression = make_tests(six, long_df)
    boot = bootstrap_angle(six)

    tests.to_csv(SRC / "nphys_readout_orthogonality_tests.csv", index=False)
    regression.to_csv(SRC / "nphys_readout_orthogonality_regression.csv", index=False)
    boot.to_csv(SRC / "nphys_readout_orthogonality_bootstrap.csv", index=False)
    build_figure(six, long_df, tests, regression, boot)
    write_report(tests, regression, boot)

    angle = regression.query("target == 'cold_hot_gradient_angle'")["r2"].iloc[0]
    print(f"Readout orthogonality audit complete. Gradient angle = {angle:.1f} deg.")
    print("Wrote figures/nphys_fig28_readout_orthogonality.[svg,pdf,png,tiff]")


if __name__ == "__main__":
    main()
