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

INK = "#262B31"
MUTED = "#6D7480"
GRID = "#E7EAEE"
SELECTED = "#2F5E8E"
CANDIDATE = "#C85E45"
OTHER = "#B6BEC8"
BOUNDARY = "#7A679E"


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


def panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def finish(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.tick_params(width=0.65)


def zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    return (out - out.mean()) / out.std(ddof=0).replace(0, np.nan)


def choose_candidates(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[df["long_cycle_true_force_route"].astype(bool)].copy()
    pool = df[~df["long_cycle_true_force_route"].astype(bool)].copy()
    coords = [
        "alpha_mult",
        "friction",
        "lid_gap_radii",
        "cold_memory_index",
        "hot_susceptibility_index",
        "loss_index",
    ]
    z = zscore(df, coords)
    sel_z = z.loc[selected.index].to_numpy(float)
    rows = []
    class_counts = selected["route_class"].value_counts().to_dict()
    hot_threshold = float(selected["hot_susceptibility_index"].median())
    cold_threshold = float(selected["cold_memory_index"].median())
    for idx, row in pool.iterrows():
        zi = z.loc[idx].to_numpy(float)
        dist = np.linalg.norm(sel_z - zi, axis=1)
        nearest = float(dist.min())
        novelty = float(dist.mean())
        class_weight = 1.0 / (1.0 + class_counts.get(row["route_class"], 0))
        boundary_score = 0.0
        if row["route_class"] == "lossy transitional":
            boundary_score += 1.0
        if row["alpha_mult"] == 1.0 and row["friction"] == 0.6:
            boundary_score += 0.55
        if row["alpha_mult"] == 0.5 and row["friction"] == 0.6:
            boundary_score += 0.45
        if abs(float(row["hot_susceptibility_index"]) - hot_threshold) < 0.35:
            boundary_score += 0.35
        if abs(float(row["cold_memory_index"]) - cold_threshold) < 0.45:
            boundary_score += 0.25
        if bool(row["selected_for_targeted_ensemble"]):
            boundary_score += 0.35
        risk_span = float(row["hot_load_risk"]) / (abs(float(row["cold_load_risk"])) + 1e-9)
        risk_score = float(np.clip(np.log10(max(risk_span, 1e-9)) + 1.0, 0.0, 2.0))
        score = 0.38 * nearest + 0.22 * novelty + 0.70 * class_weight + 0.55 * boundary_score + 0.18 * risk_score
        rows.append(
            {
                "tag": row["tag"],
                "alpha_mult": row["alpha_mult"],
                "friction": row["friction"],
                "lid_gap_radii": row["lid_gap_radii"],
                "route_class": row["route_class"],
                "selected_for_targeted_ensemble": bool(row["selected_for_targeted_ensemble"]),
                "nearest_selected_distance": nearest,
                "mean_selected_distance": novelty,
                "class_undercoverage_weight": class_weight,
                "boundary_score": boundary_score,
                "risk_score": risk_score,
                "candidate_score": score,
                "cold_memory_index": row["cold_memory_index"],
                "hot_susceptibility_index": row["hot_susceptibility_index"],
                "loss_index": row["loss_index"],
                "hot_load_risk": row["hot_load_risk"],
                "cold_load_risk": row["cold_load_risk"],
                "rationale": rationale(row),
            }
        )
    out = pd.DataFrame(rows).sort_values("candidate_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def rationale(row: pd.Series) -> str:
    bits: list[str] = []
    if row["route_class"] == "lossy transitional":
        bits.append("tests lossy boundary missing from true-force set")
    if row["alpha_mult"] == 1.0 and row["friction"] == 0.6:
        bits.append("tests high-friction overload at intermediate expansion")
    if row["alpha_mult"] == 0.5 and row["friction"] == 0.6:
        bits.append("tests whether friction alone activates loops at low expansion")
    if bool(row["selected_for_targeted_ensemble"]):
        bits.append("already targeted in three-seed ensemble")
    if not bits:
        bits.append("fills parameter/mechanism gap around current route manifold")
    return "; ".join(bits)


def draw(df: pd.DataFrame, candidates: pd.DataFrame) -> None:
    setup()
    FIG.mkdir(exist_ok=True)
    top = candidates.head(6).copy()
    fig = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1:])

    colors = df["route_class"].map(
        {"buffered reservoir": "#8FB4D8", "weakly coupled": "#C5CAD3", "lossy transitional": BOUNDARY, "excitable overload": "#D98663"}
    ).fillna(OTHER)
    ax_a.scatter(
        df["cold_memory_index"],
        df["hot_susceptibility_index"],
        s=38,
        c=colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    selected = df[df["long_cycle_true_force_route"].astype(bool)]
    ax_a.scatter(
        selected["cold_memory_index"],
        selected["hot_susceptibility_index"],
        s=78,
        facecolor="none",
        edgecolor=SELECTED,
        linewidth=1.2,
        zorder=4,
        label="existing true-force route",
    )
    top3_tags = set(top.head(3)["tag"])
    top3 = df[df["tag"].isin(top3_tags)]
    ax_a.scatter(
        top3["cold_memory_index"],
        top3["hot_susceptibility_index"],
        s=82,
        marker="*",
        color=CANDIDATE,
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
        label="recommended rerun",
    )
    offsets = {
        "a150_mu010_g020": (0.08, -0.04),
        "a150_mu030_g020": (0.08, 0.08),
        "a050_mu060_g020": (0.08, -0.02),
    }
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        dx, dy = offsets.get(row["tag"], (0.06, 0.04))
        ax_a.text(
            row["cold_memory_index"] + dx,
            row["hot_susceptibility_index"] + dy,
            f"C{rank}",
            fontsize=6.4,
            fontweight="bold",
            color=CANDIDATE,
            va="center",
            ha="left",
        )
    ax_a.axhline(0, color="#B8C0CA", lw=0.65, ls=(0, (3, 3)))
    ax_a.axvline(0, color="#B8C0CA", lw=0.65, ls=(0, (3, 3)))
    ax_a.set_xlabel("cold-memory index")
    ax_a.set_ylabel("hot-susceptibility index")
    ax_a.set_title("27-case route space and true-force gaps", loc="left")
    ax_a.legend(loc="lower left", fontsize=5.7)
    finish(ax_a)
    panel(ax_a, "a", x=-0.10)

    y = np.arange(len(top))[::-1]
    ax_b.barh(y, top["candidate_score"], color=[CANDIDATE if i < 3 else OTHER for i in range(len(top))], height=0.58)
    ax_b.set_yticks(y, top["tag"])
    ax_b.set_xlabel("generality-test score")
    ax_b.set_title("which new route best tests the mechanism?", loc="left")
    finish(ax_b, axis="x")
    panel(ax_b, "b", x=-0.04)

    terms = ["nearest_selected_distance", "boundary_score", "class_undercoverage_weight", "risk_score"]
    heat = top.head(6)[terms].copy()
    heat = (heat - heat.min()) / (heat.max() - heat.min()).replace(0, np.nan)
    ax_c.imshow(heat.to_numpy(float), cmap=mpl.colors.LinearSegmentedColormap.from_list("score", ["#F6F7F8", "#F2C99E", CANDIDATE]), aspect="auto", vmin=0, vmax=1)
    ax_c.set_yticks(np.arange(len(top)), top["tag"])
    ax_c.set_xticks(np.arange(len(terms)), ["distance", "boundary", "undercoverage", "risk"], rotation=25, ha="right")
    ax_c.set_title("why each candidate was selected", loc="left")
    ax_c.tick_params(length=0)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    panel(ax_c, "c", x=-0.04)

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"nphys_fig63_route_generality_candidates.{ext}", **kwargs)
    plt.close(fig)


def write_report(candidates: pd.DataFrame) -> None:
    top = candidates.head(6).copy()
    lines = [
        "# Route-generality candidate audit",
        "",
        "Question: if one or two additional true-force long-cycle routes can be run, which routes most directly test whether the slow-susceptibility times fast-loop-activation mechanism generalises beyond the present five-route set?",
        "",
        "## Recommended routes",
        "",
        top[
            [
                "rank",
                "tag",
                "alpha_mult",
                "friction",
                "lid_gap_radii",
                "route_class",
                "candidate_score",
                "nearest_selected_distance",
                "boundary_score",
                "rationale",
            ]
        ]
        .round(3)
        .to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The current five true-force routes already span buffered, weak/intermediate and excitable-overload behaviour, but they leave two vulnerable gaps: lossy transitional routes and high-friction/intermediate-expansion overload routes. The top candidates are therefore not chosen for largest overload alone. They are chosen because they can falsify the two-scale story: a new route should either preserve the ordering between slow route severity, loop-to-overload gain and mean overload, or show that the current five-route monotonicity is a selected-route artefact.",
        "",
        "Recommended immediate remote reruns:",
        "",
    ]
    for _, row in top.head(3).iterrows():
        lines.append(
            f"- `{row.tag}`: {row.rationale}; class={row.route_class}; score={row.candidate_score:.2f}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is a design-of-reruns audit, not new mechanism evidence. Until at least one candidate is rerun with true pair-force dumps, the manuscript should continue to describe route generality as supported within the observed five-route ensemble.",
            "",
            "## Outputs",
            "",
            "- `source_data/nphys_route_generality_candidates.csv`",
            "- `figures/nphys_fig63_route_generality_candidates.*`",
        ]
    )
    (ROOT / "nature_physics_route_generality_candidates.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_csv(SRC / "nphys_route_phase_space_27case.csv")
    candidates = choose_candidates(df)
    candidates.to_csv(SRC / "nphys_route_generality_candidates.csv", index=False)
    draw(df, candidates)
    write_report(candidates)
    print("Top route-generality candidates")
    print(candidates.head(8)[["rank", "tag", "route_class", "candidate_score", "rationale"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
