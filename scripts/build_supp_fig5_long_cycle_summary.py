#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
INFILE = SRC / "fig5_long_cycle_source.csv"

INK = "#242A31"
MUTED = "#737D89"
GRID = "#E7EAEE"
CYCLE10 = "#AEB6C0"
CYCLE30 = "#242A31"
COLORS = {
    "memory-rich": "#345995",
    "intermediate": "#D98C3A",
    "hot-load": "#C95F3F",
}
LABELS = {
    "a050_mu010_g002_c30": "memory-rich",
    "a100_mu030_g002_c30": "intermediate",
    "a150_mu060_g020_c30": "hot-load",
}


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def prepare() -> pd.DataFrame:
    df = pd.read_csv(INFILE).copy()
    df["case"] = df["tag"].map(LABELS).fillna(df["tag"])
    order = ["memory-rich", "intermediate", "hot-load"]
    df["case"] = pd.Categorical(df["case"], categories=order, ordered=True)
    df = df.sort_values("case").reset_index(drop=True)
    df["Z_change_frac"] = df["Z_cycle30"] / df["Z_cycle10"] - 1.0
    df["Gini_change_frac"] = df["Gini_cycle30"] / df["Gini_cycle10"] - 1.0
    df["survival_change_frac"] = df["survival_cycle30"] / df["survival_cycle10"] - 1.0
    return df


def draw_dumbbell(
    ax: plt.Axes,
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    change_col: str,
    xlabel: str,
    title: str,
    logx: bool = False,
) -> None:
    y = np.arange(len(df))
    for i, row in df.iterrows():
        color = COLORS[str(row["case"])]
        x0 = float(row[start_col])
        x1 = float(row[end_col])
        ax.plot([x0, x1], [i, i], color=color, lw=2.0, alpha=0.72, solid_capstyle="round", zorder=2)
        ax.scatter(x0, i, s=34, facecolor="white", edgecolor=color, lw=0.9, zorder=3)
        ax.scatter(x1, i, s=38, facecolor=color, edgecolor="white", lw=0.55, zorder=4)
        x_text = np.sqrt(x0 * x1) if logx else 0.5 * (x0 + x1)
        ax.text(
            x_text,
            i - 0.18,
            f"{row[change_col] * 100:+.0f}%",
            color=color,
            ha="center",
            va="center",
            fontsize=6.3,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78),
            clip_on=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(df["case"])
    for tick, case in zip(ax.get_yticklabels(), df["case"]):
        tick.set_color(COLORS[str(case)])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=4)
    if logx:
        ax.set_xscale("log")
    ax.margins(x=0.16, y=0.20)
    finish(ax, axis="x")


def draw_fabric_arrow(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        case = str(row["case"])
        color = COLORS[case]
        x0 = float(row["survival_cycle10"])
        y0 = float(row["Z_cycle10"])
        x1 = float(row["survival_cycle30"])
        y1 = float(row["Z_cycle30"])
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.25, shrinkA=2, shrinkB=2),
            zorder=3,
        )
        ax.scatter(x0, y0, s=30, facecolor="white", edgecolor=color, lw=0.8, zorder=4)
        ax.scatter(x1, y1, s=36, facecolor=color, edgecolor="white", lw=0.55, zorder=5)
        ax.text(x1 + 0.006, y1, case, color=color, fontsize=6.2, va="center")
    ax.set_xlabel("hot-cold contact survival")
    ax.set_ylabel(r"cold coordination $Z_c$")
    ax.set_title("fabric register from cycle 10 to 30", loc="left", pad=4)
    finish(ax)


def draw_drift_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    cols = [
        ("cold\nload", "cold_bottom_change_10_30_frac"),
        ("hot\nload", "hot_bottom_change_10_30_frac"),
        (r"$Z_c$", "Z_change_frac"),
        (r"$G_f$", "Gini_change_frac"),
        ("survival", "survival_change_frac"),
    ]
    mat = df[[c for _, c in cols]].to_numpy(float)
    vmax = 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["case"])
    for tick, case in zip(ax.get_yticklabels(), df["case"]):
        tick.set_color(COLORS[str(case)])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for label, _ in cols])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v * 100:+.0f}%", ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 0.45 else INK)
    ax.set_title("post-cycle-10 drift fingerprint", loc="left", pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("fractional change", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8, length=2)


def draw_work_bar(ax: plt.Axes, df: pd.DataFrame) -> None:
    y = np.arange(len(df))
    values = df["work_total_30"].to_numpy(float) * 1e3
    ax.barh(y, values, color=[COLORS[str(c)] for c in df["case"]], height=0.48, alpha=0.88)
    for i, val in enumerate(values):
        ax.text(val + max(values) * 0.035, i, f"{val:.2f}", va="center", fontsize=6.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(df["case"])
    for tick, case in zip(ax.get_yticklabels(), df["case"]):
        tick.set_color(COLORS[str(case)])
    ax.invert_yaxis()
    ax.set_xlabel(r"$10^3\sum W_{\rm irr}$ (J)")
    ax.set_title("irreversible work accumulated", loc="left", pad=4)
    finish(ax, axis="x")


def build_figure(df: pd.DataFrame) -> None:
    setup_style()
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_dumbbell(
        ax_a,
        df,
        "cold_bottom_cycle10",
        "cold_bottom_cycle30",
        "cold_bottom_change_10_30_frac",
        r"retained cold bottom load $p_{b,c}$ (Pa)",
        "cold load after the regime map",
    )
    panel(ax_a, "a")
    draw_dumbbell(
        ax_b,
        df,
        "hot_bottom_cycle10",
        "hot_bottom_cycle30",
        "hot_bottom_change_10_30_frac",
        r"instantaneous hot bottom load $p_{b,h}$ (Pa)",
        "hot overload relaxes or persists",
        logx=True,
    )
    panel(ax_b, "b")
    draw_fabric_arrow(ax_c, df)
    panel(ax_c, "c")
    draw_drift_matrix(ax_d, df)
    panel(ax_d, "d")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"fig5.{ext}", **kwargs)
    plt.close(fig)


def write_report(df: pd.DataFrame) -> None:
    out = ROOT / "supp_fig5_long_cycle_summary_report.md"
    lines = [
        "# Supplementary Fig. 5 long-cycle summary",
        "",
        "Purpose: replace the visually crowded 30-cycle spaghetti plot with a source-data-traceable summary of what changes between cycle 10 and cycle 30.",
        "",
        df[
            [
                "case",
                "cold_bottom_change_10_30_frac",
                "hot_bottom_change_10_30_frac",
                "Z_change_frac",
                "Gini_change_frac",
                "survival_change_frac",
                "work_total_30",
            ]
        ].round(4).to_markdown(index=False),
        "",
        "Interpretation: the memory-rich and intermediate cases are close to saturated in retained cold load, while the hot-load case has a large residual cold-load drift after cycle 10 and a strong hot-load relaxation. The figure is a validation summary, not a substitute for the main five-route true-force mechanism.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = prepare()
    build_figure(df)
    write_report(df)
    print("Wrote supplementary Fig. 5 long-cycle summary")
    print(
        df[
            [
                "case",
                "cold_bottom_change_10_30_frac",
                "hot_bottom_change_10_30_frac",
                "Z_change_frac",
                "survival_change_frac",
                "work_total_30",
            ]
        ].round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
