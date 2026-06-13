#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"

INK = "#262B31"
MUTED = "#6D7480"
GRID = "#E7EAEE"
BLUE = "#355F91"
ORANGE = "#C65F42"
TEAL = "#267C78"
PURPLE = "#7A679E"
GOLD = "#C9963B"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
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


def panel(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color="black")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def get_model_score(model_tests: pd.DataFrame, model: str, validation: str) -> float:
    row = model_tests[(model_tests["model"].eq(model)) & (model_tests["validation"].eq(validation))]
    return float(row["r2_vs_training_mean"].iloc[0])


def build_prediction_table() -> pd.DataFrame:
    memory = pd.read_csv(SRC / "nphys_fig1_memory_source.csv")
    semi = pd.read_csv(SRC / "nphys_fig2_boundary_semi_confined_source.csv")
    pre = pd.read_csv(SRC / "nphys_fig2_boundary_precompressed_source.csv")
    two = pd.read_csv(SRC / "nature_physics_two_channel_summary_source.csv")
    breath = pd.read_csv(SRC / "nphys_breathing_cycle_correlations.csv")
    hierarchy = pd.read_csv(SRC / "nphys_mechanism_hierarchy_model_tests.csv")
    robust = pd.read_csv(SRC / "nphys_force_loop_robustness_conditional.csv")

    cold_r, cold_p = stats.pearsonr(two["Z_cold_N_mean"], two["cold_bottom_pN_mean"])
    hot_r, hot_p = stats.pearsonr(two["hot_force_mean"], two["hot_bottom_pN_mean"])

    heat_semi = semi[semi["Phase"].eq("Heating")]
    heat_pre = pre[pre["Phase"].eq("Heating")]
    side_amp = heat_semi["SidePressure_Pa"].iloc[-1] / heat_semi["SidePressure_Pa"].iloc[0]
    bottom_release = heat_pre["BottomPressure_Pa"].iloc[-1] / heat_pre["BottomPressure_Pa"].iloc[0]
    lid_release = heat_pre["LidPressure_Pa"].iloc[-1] / heat_pre["LidPressure_Pa"].iloc[0]

    z_gain = (memory["z_mean"].iloc[-1] - memory["z_mean"].iloc[0]) / memory["z_mean"].iloc[0]
    survival_gain = memory["contact_survival"].iloc[-1] - memory["contact_survival"].iloc[0]
    rearrange_drop = memory["rms_displacement_m"].iloc[-1] / memory["rms_displacement_m"].iloc[0]

    lag_fabric = breath[breath["relationship"].eq("inhaled_geometry_to_next_cold_fabric")].iloc[0]
    lag_loop = breath[breath["relationship"].eq("inhaled_positive_cycles_to_next_cold_loop_memory")].iloc[0]
    loop_after_tail = robust[robust["test"].eq("loop_after_top5_and_cycle")].iloc[0]

    rows = [
        {
            "prediction_id": "P1",
            "claim": "Cold cycling trains a fabric register",
            "experimental_readout": "height/porosity plus tomography or acoustic fabric",
            "dem_evidence": f"Z gain {z_gain:.2f}; survival +{100*survival_gain:.1f} pp; rearrangement ratio {rearrange_drop:.2f}",
            "primary_metric": z_gain,
            "evidence_strength": min(1.0, abs(z_gain) / 0.10),
            "observability": 0.90,
            "falsifier": "No persistent rise in fabric/contact-survival proxy under repeated identical cycles",
        },
        {
            "prediction_id": "P2",
            "claim": "Boundary projection changes the sign of load response",
            "experimental_readout": "phase-resolved side, bottom and lid force arrays",
            "dem_evidence": f"semi side amplification {side_amp:.2f}x; pre bottom {bottom_release:.2f}x; pre lid {lid_release:.3f}x",
            "primary_metric": np.log10(side_amp / max(lid_release, 1e-6)),
            "evidence_strength": 0.95,
            "observability": 0.85,
            "falsifier": "All boundary protocols collapse onto one monotonic pressure-growth curve",
        },
        {
            "prediction_id": "P3",
            "claim": "Cold and hot readouts require different state variables",
            "experimental_readout": "cold fabric imaging plus hot wall-force intermittency",
            "dem_evidence": f"cold load vs Z r={cold_r:.2f}; hot load vs hot force scale r={hot_r:.2f}",
            "primary_metric": min(abs(cold_r), abs(hot_r)),
            "evidence_strength": min(1.0, min(abs(cold_r), abs(hot_r))),
            "observability": 0.75,
            "falsifier": "Residual cold load and transient hot overload collapse onto the same scalar density or pressure coordinate",
        },
        {
            "prediction_id": "P4",
            "claim": "Hot overload is organised by loop activation, not only force tails",
            "experimental_readout": "photoelastic or calibrated DEM-contact graph matched to wall-force arrays",
            "dem_evidence": f"loop residual rho={loop_after_tail['spearman']:.2f}; dimensionless loop LORO R2={get_model_score(hierarchy, 'dimensionless loop', 'leave_one_route_out'):.2f}",
            "primary_metric": float(loop_after_tail["spearman"]),
            "evidence_strength": 0.82,
            "observability": 0.45,
            "falsifier": "Force-tail metrics predict overload after graph-loop embedding is destroyed or randomised",
        },
        {
            "prediction_id": "P5",
            "claim": "A hot excursion leaves a next-cold imprint",
            "experimental_readout": "cycle-resolved tomography, acoustic memory or unload-reload probes",
            "dem_evidence": f"fabric imprint rho={lag_fabric['spearman_within_regime_centered']:.2f}; loop imprint rho={lag_loop['spearman_within_regime_centered']:.2f}",
            "primary_metric": min(abs(lag_fabric["spearman_within_regime_centered"]), abs(lag_loop["spearman_within_regime_centered"])),
            "evidence_strength": min(abs(lag_fabric["spearman_within_regime_centered"]), abs(lag_loop["spearman_within_regime_centered"])),
            "observability": 0.55,
            "falsifier": "Hot-phase excursions do not predict any next-cold fabric or loop-memory observable",
        },
    ]
    df = pd.DataFrame(rows)
    df["difficulty"] = 1 - df["observability"]
    df.to_csv(SRC / "nphys_experimental_prediction_matrix.csv", index=False)
    return df


def draw_prediction_flow(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "a", x=0.0, y=1.04)
    ax.text(0.02, 0.95, "falsifiable experimental ladder", fontsize=9.8, fontweight="bold", color=INK)
    items = [
        (0.12, "image fabric", "M"),
        (0.34, "drive cycle", "Theta"),
        (0.56, "wall-force phase", "P_w"),
        (0.78, "next-cold imprint", "X_{n+1}"),
    ]
    colors = [BLUE, GOLD, ORANGE, TEAL]
    for (x, title, symbol), c in zip(items, colors):
        ax.scatter([x], [0.58], s=1150, color=c, alpha=0.18, edgecolor=c, lw=1.1)
        ax.text(x, 0.60, rf"${symbol}$", ha="center", va="center", fontsize=12, color=c, fontweight="bold")
        ax.text(x, 0.35, title, ha="center", va="top", fontsize=7.4, color=INK)
    for x0, x1 in [(0.19, 0.27), (0.41, 0.49), (0.63, 0.71)]:
        ax.annotate("", xy=(x1, 0.58), xytext=(x0, 0.58), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=MUTED))
    ax.text(0.50, 0.13, "failure of any link falsifies the breathing map", ha="center", fontsize=7.4, color=MUTED)


def draw_score_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    y = np.arange(len(df))[::-1]
    colors = [BLUE, GOLD, TEAL, ORANGE, PURPLE]
    ax.barh(y, df["evidence_strength"], color=colors, height=0.52, alpha=0.88)
    ax.scatter(df["observability"], y, s=42, color="white", edgecolor=INK, lw=0.8, zorder=3, label="observability")
    ax.set_yticks(y, df["prediction_id"] + "  " + df["claim"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("evidence strength / observability")
    ax.set_title("what the present DEM already constrains", loc="left")
    ax.legend(loc="lower right", fontsize=5.8)
    finish(ax, axis="x")
    panel(ax, "b", x=-0.04)


def draw_difficulty_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    y = np.arange(len(df))[::-1]
    colors = [BLUE, GOLD, TEAL, ORANGE, PURPLE]
    ax.barh(y, df["difficulty"], color=colors, height=0.54, alpha=0.82)
    ax.set_yticks(y, df["prediction_id"])
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("experimental difficulty")
    ax.set_title("hardest test is loop-resolved force topology", loc="left")
    finish(ax, axis="x")
    panel(ax, "c", x=-0.20)


def draw_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    cols = ["fabric", "wall force", "graph loops", "lagged memory"]
    mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.3],
            [0.4, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.3, 0.0],
            [0.2, 1.0, 1.0, 0.0],
            [0.8, 0.2, 0.6, 1.0],
        ]
    )
    im = ax.imshow(mat, aspect="auto", cmap=mpl.colors.LinearSegmentedColormap.from_list("obs", ["#F6F7F8", "#A7CED1", TEAL]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df)), df["prediction_id"])
    ax.set_title("observable needed to test each prediction", loc="left")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.75:
                ax.text(j, i, "required", ha="center", va="center", fontsize=5.6, color=INK)
            elif mat[i, j] > 0:
                ax.text(j, i, "helpful", ha="center", va="center", fontsize=5.6, color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel(ax, "d", x=-0.10)


def draw_falsifiers(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "e", x=0.0, y=1.03)
    ax.text(0.02, 0.94, "decisive failure modes", fontsize=7.8, fontweight="bold", color=INK)
    y = 0.82
    for _, row in df.iterrows():
        ax.text(0.02, y, row["prediction_id"], fontsize=7.0, fontweight="bold", color=INK)
        ax.text(0.12, y, row["falsifier"], fontsize=6.2, color=INK, va="top", wrap=True)
        y -= 0.17


def save_report(df: pd.DataFrame) -> None:
    report_cols = [
        "prediction_id",
        "claim",
        "experimental_readout",
        "dem_evidence",
        "evidence_strength",
        "observability",
        "falsifier",
    ]
    table = ["| " + " | ".join(report_cols) + " |", "| " + " | ".join(["---"] * len(report_cols)) + " |"]
    for _, row in df[report_cols].iterrows():
        cells = []
        for col in report_cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3g}")
            else:
                cells.append(str(val).replace("|", "/"))
        table.append("| " + " | ".join(cells) + " |")
    lines = [
        "# Experimental prediction matrix",
        "",
        "This audit converts the DEM mechanism into falsifiable experimental predictions. It does not claim that these experiments have already been performed.",
        "",
        "\n".join(table),
        "",
        "## Manuscript-safe interpretation",
        "",
        "- The strongest near-term experimental tests are phase-resolved wall-force arrays plus independent fabric observables.",
        "- Force-loop observables are the hardest in opaque ceramic beds; the safer wording is that they can be tested in photoelastic or index-matched analogue beds, or by calibrated DEM constrained by wall-force arrays.",
        "- A scalar pressure-growth law would be falsified by the precompressed release and by non-collapse of cold and hot readouts onto one state coordinate.",
        "- A decisive positive experiment would show that a cold fabric observable predicts next-cold imprint, while a graph-loop or calibrated stress-network observable predicts hot overload.",
    ]
    (ROOT / "nature_physics_experimental_prediction_matrix.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup()
    df = build_prediction_table()
    fig = plt.figure(figsize=(7.25, 5.9))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.95, wspace=0.52, hspace=0.58)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.18, 1.08, 0.92], height_ratios=[0.95, 1.05, 0.92])
    draw_prediction_flow(fig.add_subplot(gs[0, :2]))
    draw_matrix(fig.add_subplot(gs[0, 2]), df)
    draw_score_panel(fig.add_subplot(gs[1, :2]), df)
    draw_difficulty_panel(fig.add_subplot(gs[1, 2]), df)
    draw_falsifiers(fig.add_subplot(gs[2, :]), df)
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"nphys_fig22_experimental_prediction_matrix.{ext}", dpi=600 if ext in {"png", "tiff"} else None, bbox_inches="tight")
    plt.close(fig)
    save_report(df)
    print("wrote figures/nphys_fig22_experimental_prediction_matrix.*")
    print("wrote source_data/nphys_experimental_prediction_matrix.csv")
    print("wrote nature_physics_experimental_prediction_matrix.md")


if __name__ == "__main__":
    main()
