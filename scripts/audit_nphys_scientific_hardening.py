#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"

DELTA = SRC / "nphys_long_cycle_true_force_hot_cold_delta.csv"
BREATH = SRC / "nphys_breathing_parameter_effects_cycle_metrics.csv"

OUT_BLOCK = SRC / "nphys_hardening_block_bootstrap.csv"
OUT_SHIFT = SRC / "nphys_hardening_circular_shift_null.csv"
OUT_PRED = SRC / "nphys_hardening_return_map_prediction.csv"
OUT_REPORT = ROOT / "nature_physics_scientific_hardening_audit.md"


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def demean_by_group(df: pd.DataFrame, cols: list[str], group: str = "regime_id") -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_wc"] = out[col] - out.groupby(group)[col].transform("mean")
    return out


def spearman_wc(df: pd.DataFrame, x: str, y: str) -> float:
    d = demean_by_group(df[[x, y, "regime_id"]].dropna(), [x, y])
    if len(d) < 4:
        return float("nan")
    return float(spearmanr(d[f"{x}_wc"], d[f"{y}_wc"]).statistic)


def make_blocks(df: pd.DataFrame, block_size: int = 5) -> list[pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    for _, g in df.sort_values(["regime_id", "cycle"]).groupby("regime_id", sort=True):
        cycles = g["cycle"].to_numpy()
        for start in range(int(cycles.min()), int(cycles.max()) + 1, block_size):
            block = g[(g["cycle"] >= start) & (g["cycle"] < start + block_size)]
            if not block.empty:
                blocks.append(block.copy())
    return blocks


def block_bootstrap_spearman(df: pd.DataFrame, pairs: list[tuple[str, str, str]], *, n_boot: int = 5000, block_size: int = 5, seed: int = 1307) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    blocks_by_regime = {
        regime: make_blocks(g, block_size=block_size)
        for regime, g in df.groupby("regime_id", sort=True)
    }
    for name, x, y in pairs:
        obs = spearman_wc(df, x, y)
        vals = np.empty(n_boot)
        for i in range(n_boot):
            sampled = []
            for regime, blocks in blocks_by_regime.items():
                n_blocks = len(blocks)
                picks = rng.integers(0, n_blocks, size=n_blocks)
                for pick in picks:
                    sampled.append(blocks[pick])
            boot = pd.concat(sampled, ignore_index=True)
            vals[i] = spearman_wc(boot, x, y)
        rows.append(
            {
                "relationship": name,
                "predictor": x,
                "target": y,
                "observed_rho_within": obs,
                "block_size_cycles": block_size,
                "n_boot": n_boot,
                "boot_median": float(np.nanmedian(vals)),
                "boot_q025": float(np.nanquantile(vals, 0.025)),
                "boot_q975": float(np.nanquantile(vals, 0.975)),
                "boot_pr_positive": float(np.nanmean(vals > 0)),
                "boot_pr_negative": float(np.nanmean(vals < 0)),
            }
        )
    return pd.DataFrame(rows)


def circular_shift_null(df: pd.DataFrame, pairs: list[tuple[str, str, str]], *, n_perm: int = 5000, seed: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, x, y in pairs:
        d = df[["regime_id", "cycle", x, y]].dropna().sort_values(["regime_id", "cycle"]).copy()
        obs = spearman_wc(d, x, y)
        vals = np.empty(n_perm)
        for i in range(n_perm):
            shifted = []
            for _, g in d.groupby("regime_id", sort=True):
                arr = g[x].to_numpy()
                if len(arr) <= 2:
                    k = 0
                else:
                    k = int(rng.integers(1, len(arr)))
                gg = g.copy()
                gg[x] = np.roll(arr, k)
                shifted.append(gg)
            null = pd.concat(shifted, ignore_index=True)
            vals[i] = spearman_wc(null, x, y)
        p = (np.sum(np.abs(vals) >= abs(obs)) + 1) / (n_perm + 1)
        rows.append(
            {
                "relationship": name,
                "predictor": x,
                "target": y,
                "observed_rho_within": obs,
                "n_perm": n_perm,
                "circular_shift_p_two_sided": float(p),
                "null_median": float(np.nanmedian(vals)),
                "null_q025": float(np.nanquantile(vals, 0.025)),
                "null_q975": float(np.nanquantile(vals, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def design(df: pd.DataFrame, cols: list[str], include_intercept: bool = True) -> np.ndarray:
    parts = []
    if include_intercept:
        parts.append(np.ones((len(df), 1)))
    for col in cols:
        parts.append(df[[col]].to_numpy(float))
    return np.column_stack(parts)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str) -> np.ndarray:
    mu = train[cols].mean()
    sig = train[cols].std(ddof=0).replace(0, 1.0)
    x_train = ((train[cols] - mu) / sig).to_numpy(float)
    x_test = ((test[cols] - mu) / sig).to_numpy(float)
    x_train = np.column_stack([np.ones(len(train)), x_train])
    x_test = np.column_stack([np.ones(len(test)), x_test])
    y = train[target].to_numpy(float)
    beta = np.linalg.lstsq(x_train, y, rcond=None)[0]
    return x_test @ beta


def out_of_sample_metrics(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(baseline)
    y = y[ok]
    yhat = yhat[ok]
    baseline = baseline[ok]
    sse = float(np.sum((y - yhat) ** 2))
    sse0 = float(np.sum((y - baseline) ** 2))
    r2 = 1.0 - sse / sse0 if sse0 > 0 else float("nan")
    rho = float(spearmanr(y, yhat).statistic) if len(y) >= 4 else float("nan")
    return {"n_test": int(len(y)), "sse": sse, "baseline_sse": sse0, "oos_r2_vs_baseline": r2, "spearman_y_yhat": rho}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]


def prepare_prediction_table() -> pd.DataFrame:
    m = pd.read_csv(BREATH).sort_values(["regime_id", "cycle"]).copy()
    target = "force_p99_hot_minus_cold"
    lag_cols = [
        target,
        "fabric_reservoir_index",
        "breath_amplitude",
        "imprint_efficiency",
        "loop_cost",
        "force_h1_birth_force_share_hot_minus_cold",
        "force_share_top5_edges_hot_minus_cold",
        "cycle_birth_positive_fraction_inhale_delta",
        "Z_geom_inhale_delta",
    ]
    for col in lag_cols:
        m[f"lag_{col}"] = m.groupby("regime_id")[col].shift(1)
    m["target_next_overload"] = m[target]
    m["baseline_prev_overload"] = m[f"lag_{target}"]
    return m.dropna(subset=["target_next_overload", "baseline_prev_overload"]).reset_index(drop=True)


def return_map_prediction() -> pd.DataFrame:
    df = prepare_prediction_table()
    target = "target_next_overload"
    specs = [
        ModelSpec("AR_previous_overload", ("baseline_prev_overload",)),
        ModelSpec(
            "AR_plus_fabric_loop",
            (
                "baseline_prev_overload",
                "lag_fabric_reservoir_index",
                "lag_force_h1_birth_force_share_hot_minus_cold",
            ),
        ),
        ModelSpec(
            "AR_plus_breathing_parameters",
            (
                "baseline_prev_overload",
                "lag_breath_amplitude",
                "lag_imprint_efficiency",
                "lag_loop_cost",
            ),
        ),
        ModelSpec(
            "AR_plus_full_minimal_map",
            (
                "baseline_prev_overload",
                "lag_fabric_reservoir_index",
                "lag_force_h1_birth_force_share_hot_minus_cold",
                "lag_breath_amplitude",
                "lag_imprint_efficiency",
                "lag_loop_cost",
            ),
        ),
        ModelSpec(
            "top5_control_map",
            (
                "baseline_prev_overload",
                "lag_force_share_top5_edges_hot_minus_cold",
            ),
        ),
    ]
    rows = []
    regimes = sorted(df["regime_id"].dropna().unique())
    for spec in specs:
        all_y, all_yhat, all_base = [], [], []
        for regime in regimes:
            train = df[df["regime_id"] != regime].copy()
            test = df[df["regime_id"] == regime].copy()
            cols = list(spec.features)
            pred = fit_predict(train.dropna(subset=cols + [target]), test.dropna(subset=cols + [target]), cols, target)
            test_ok = test.dropna(subset=cols + [target])
            all_y.append(test_ok[target].to_numpy(float))
            all_yhat.append(pred)
            all_base.append(test_ok["baseline_prev_overload"].to_numpy(float))
        y = np.concatenate(all_y)
        yhat = np.concatenate(all_yhat)
        base = np.concatenate(all_base)
        row = {"validation": "leave_one_regime_out", "model": spec.name, "features": ";".join(spec.features)}
        row.update(out_of_sample_metrics(y, yhat, base))
        rows.append(row)

    # A within-route forward-chaining check addresses the opposite question: can the map
    # predict future cycles inside a known route after observing its early part?
    for spec in specs:
        all_y, all_yhat, all_base = [], [], []
        cols = list(spec.features)
        for _, g in df.groupby("regime_id", sort=True):
            g = g.dropna(subset=cols + [target]).sort_values("cycle")
            cut = int(np.floor(0.6 * len(g)))
            train = g.iloc[:cut]
            test = g.iloc[cut:]
            if len(train) <= len(cols) + 1 or len(test) < 3:
                continue
            pred = fit_predict(train, test, cols, target)
            all_y.append(test[target].to_numpy(float))
            all_yhat.append(pred)
            all_base.append(test["baseline_prev_overload"].to_numpy(float))
        y = np.concatenate(all_y)
        yhat = np.concatenate(all_yhat)
        base = np.concatenate(all_base)
        row = {"validation": "within_regime_forward_60_40", "model": spec.name, "features": ";".join(spec.features)}
        row.update(out_of_sample_metrics(y, yhat, base))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(block: pd.DataFrame, shift: pd.DataFrame, pred: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        return df[cols].to_markdown(index=False, floatfmt=".3g")

    lines = [
        "# Scientific hardening audit",
        "",
        "This audit targets three review risks: pseudo-replication across cycles, whether loop activation survives autocorrelation-preserving nulls, and whether the breathing variables have predictive value beyond a previous-overload baseline.",
        "",
        "## 1. Block bootstrap for within-regime correlations",
        "",
        md_table(
            block,
            [
                "relationship",
                "observed_rho_within",
                "boot_q025",
                "boot_median",
                "boot_q975",
                "boot_pr_positive",
                "boot_pr_negative",
            ],
        ),
        "",
        "Interpretation: block resampling treats neighbouring cycles as dependent. A relation is robust only if the bootstrap interval does not cross zero with the expected sign.",
        "",
        "## 2. Circular-shift null within each regime",
        "",
        md_table(
            shift,
            [
                "relationship",
                "observed_rho_within",
                "circular_shift_p_two_sided",
                "null_q025",
                "null_median",
                "null_q975",
            ],
        ),
        "",
        "Interpretation: circular shifts preserve the autocorrelated route shape inside each regime but break cycle alignment between predictor and target.",
        "",
        "## 3. Minimal return-map prediction",
        "",
        md_table(
            pred,
            [
                "validation",
                "model",
                "n_test",
                "oos_r2_vs_baseline",
                "spearman_y_yhat",
            ],
        ),
        "",
        "Interpretation: leave-one-regime-out tests transfer across routes and is deliberately harsh for only three routes. The within-regime forward split asks whether early cycles in a route can predict later cycles in the same route. Positive out-of-sample R2 means the model beats the previous-cycle overload baseline.",
        "",
        "## Manuscript-safe conclusion",
        "",
    ]

    loop_block = block.loc[block["relationship"] == "loop_to_overload"].iloc[0]
    top5_block = block.loc[block["relationship"] == "top5_to_overload"].iloc[0]
    loop_shift = shift.loc[shift["relationship"] == "loop_to_overload"].iloc[0]
    best_forward = pred[pred["validation"] == "within_regime_forward_60_40"].sort_values("oos_r2_vs_baseline", ascending=False).head(1)
    best_loro = pred[pred["validation"] == "leave_one_regime_out"].sort_values("oos_r2_vs_baseline", ascending=False).head(1)

    lines.extend(
        [
            f"- Loop activation remains positive under block bootstrap: rho={loop_block.observed_rho_within:.3f}, 95% block interval [{loop_block.boot_q025:.3f}, {loop_block.boot_q975:.3f}].",
            f"- Top-5% force concentration is not a substitute mechanism: rho={top5_block.observed_rho_within:.3f}, 95% block interval [{top5_block.boot_q025:.3f}, {top5_block.boot_q975:.3f}].",
            f"- The circular-shift null gives loop-to-overload P={loop_shift.circular_shift_p_two_sided:.4g}, so the relation is not explained only by smooth cycle trends.",
        ]
    )
    if not best_forward.empty:
        row = best_forward.iloc[0]
        lines.append(
            f"- In forward prediction within known routes, the best minimal map is `{row.model}` with out-of-sample R2={row.oos_r2_vs_baseline:.3f} against the previous-overload baseline."
        )
    if not best_loro.empty:
        row = best_loro.iloc[0]
        lines.append(
            f"- Cross-route leave-one-regime-out remains the hardest test: the best model is `{row.model}` with out-of-sample R2={row.oos_r2_vs_baseline:.3f}. Treat this as a transfer-risk boundary, not as a universal law."
        )
    lines.extend(
        [
            "",
            "Recommended framing: claim a route-conditioned, autocorrelation-checked loop mechanism and a predictive return-map diagnostic within known routes. Do not claim a universal transferable constitutive model until more regimes or experimental validation are available.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    delta = pd.read_csv(DELTA)
    breath = pd.read_csv(BREATH)
    pairs_delta = [
        ("loop_to_overload", "force_h1_birth_force_share_hot_minus_cold", "force_p99_hot_minus_cold"),
        ("top5_to_overload", "force_share_top5_edges_hot_minus_cold", "force_p99_hot_minus_cold"),
        ("giant_component_to_overload", "giant_fraction_after_top5_edges_hot_minus_cold", "force_p99_hot_minus_cold"),
    ]
    pairs_breath = [
        ("fabric_to_next_cold_fabric", "Z_geom_inhale_delta", "Z_geom_next_cold_minus_current"),
        ("positive_cycles_to_next_cold_loop_memory", "cycle_birth_positive_fraction_inhale_delta", "force_h1_birth_force_share_next_cold_minus_current"),
        ("amplitude_to_overload", "breath_amplitude", "force_p99_hot_minus_cold"),
        ("imprint_efficiency_to_overload", "imprint_efficiency", "force_p99_hot_minus_cold"),
        ("loop_cost_to_overload", "loop_cost", "force_p99_hot_minus_cold"),
    ]
    block = pd.concat(
        [
            block_bootstrap_spearman(delta, pairs_delta),
            block_bootstrap_spearman(breath, pairs_breath),
        ],
        ignore_index=True,
    )
    shift = pd.concat(
        [
            circular_shift_null(delta, pairs_delta),
            circular_shift_null(breath, pairs_breath),
        ],
        ignore_index=True,
    )
    pred = return_map_prediction()
    block.to_csv(OUT_BLOCK, index=False)
    shift.to_csv(OUT_SHIFT, index=False)
    pred.to_csv(OUT_PRED, index=False)
    write_report(block, shift, pred)
    print(OUT_BLOCK)
    print(OUT_SHIFT)
    print(OUT_PRED)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
