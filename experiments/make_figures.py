"""Figures for the report, drawn from the CSV files written by the experiment scripts.

Usage: python experiments/make_figures.py [--results results] [--out report/figures]
Needs ``rq1_runs_linear.csv``, ``rq2_stages.csv``, ``rq3_runs_linear.csv`` and
``sensitivity_oat.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from rae.stats import mean_ci  # noqa: E402

# Reference palette (light mode): categorical slots in fixed order, a one-hue sequential ramp,
# and recessive chrome. Text never takes a series colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = ["#86b6ef", "#2a78d6", "#104281"]  # blue 250 / 450 / 650
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.edgecolor": AXIS,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)


def _ci(block: pd.DataFrame, column: str) -> tuple[float, float]:
    return mean_ci(block[column])


def _label_ends(
    ax, ends: list[tuple[float, float, str]], gap: float = 0.06, left: frozenset[str] = frozenset()
) -> None:
    """Direct labels at the end of each series, nudged apart vertically so they never collide.

    Call after plotting. ``gap`` is the minimum distance between labels as a share of the y
    range; labels named in ``left`` go on the left of their point instead of the right.
    """
    low, high = ax.get_ylim()
    min_gap = gap * (high - low)
    right_of = offset_copy(ax.transData, fig=ax.figure, x=6, y=0, units="points")
    left_of = offset_copy(ax.transData, fig=ax.figure, x=-6, y=0, units="points")
    last = None
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        y_label = y if last is None else max(y, last + min_gap)
        last = y_label
        on_left = text in left
        ax.text(
            x,
            y_label,
            text,
            transform=left_of if on_left else right_of,
            va="center",
            ha="right" if on_left else "left",
            color=INK_2,
            fontsize=8,
        )


def rq1_figure(runs: pd.DataFrame, out: Path) -> None:
    """Dot plot of the Q1/Q4 ratio per selector, loop off, one panel per historical selector."""
    data = runs[runs["exposure_weight"] == 0.0]
    order = [s for s in dict.fromkeys(data["selector"]) if s != "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True, sharey=True)
    for ax, hist_name in zip(axes, ["scout", "biased scout"], strict=True):
        block = data[data["historical"] == hist_name]
        stats = [_ci(block[block["selector"] == s], "selection_rate_ratio_q1_q4") for s in order]
        y = range(len(order))
        ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=":")
        ax.axvline(stats[0][0], color=AXIS, linewidth=1, linestyle="--")
        ax.errorbar(
            [m for m, _ in stats],
            y,
            xerr=[h for _, h in stats],
            fmt="o",
            color=SERIES[0],
            ecolor=SERIES[0],
            elinewidth=1.5,
            capsize=0,
        )
        for yi, (m, _) in zip(y, stats, strict=True):
            ax.annotate(
                f"{m:.1f}",
                (m, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color=INK_2,
                fontsize=8,
            )
        ax.set_title(f"Historical data from the {hist_name}", loc="left")
        ax.set_yticks(list(y), order)
        ax.set_ylim(len(order) - 0.5, -0.8)  # inverted, with room for the top value label
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Q1/Q4 selection-rate ratio (1 = no RAE)")
    fig.suptitle(
        "RQ1. ML selectors vs the historical selector (feedback loop off)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        -0.02,
        "Dots: mean over seeds; bars: 95% CI; dashed: historical selector; dotted: parity.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / "rq1_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rq2_figure(stages: pd.DataFrame, out: Path) -> None:
    """Q1/Q4 ratio and equal-opportunity gap by stage, for growing exposure weight (scout)."""
    data = stages[stages["selector"] == "scout with memory"]
    weights = [0.0, 1.0, 2.0]
    panels = [
        ("ratio_q1_q4", "Q1/Q4 selection-rate ratio"),
        ("equal_opportunity_gap", "Equal-opportunity gap"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        ends = []
        for colour, c in zip(SEQUENTIAL, weights, strict=True):
            means = data[data["exposure_weight"] == c].groupby("stage")[metric].mean()
            ax.plot(means.index, means.values, marker="o", color=colour, label=f"c = {c:g}")
            ends.append((means.index[-1], means.values[-1], f"c = {c:g}"))
        _label_ends(ax, ends)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Stage")
        ax.set_xticks(sorted(data["stage"].unique()))
        ax.set_xlim(right=data["stage"].max() + 0.45)
    axes[0].legend(loc="upper left", title="Exposure weight", title_fontsize=8)
    fig.suptitle(
        "RQ2. The feedback loop across stages (scout with memory)", x=0.01, ha="left", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / "rq2_stages.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rq3_figure(runs: pd.DataFrame, out: Path, exposure: float = 1.0) -> None:
    """Talent loss among late-born athletes vs regret, for each mitigation and strength."""
    data = runs[(runs["exposure_weight"] == exposure) & (runs["base"] != "oracle")]
    means = data.groupby(["base", "family", "strength"])[["regret", "talent_loss_q4"]].mean()
    families = [
        ("RA-adjusted", SERIES[0]),
        ("quota", SERIES[1]),
        ("bio-banding", SERIES[2]),
        ("rotating", MUTED),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    all_ends = []
    for ax, base in zip(axes, ["scout", "ML later perf. + birth"], strict=True):
        start = means.loc[(base, "none", 0.0)]
        ends = []
        for family, colour in families:
            path = pd.concat([start.to_frame().T, means.loc[(base, family)]])
            ax.plot(path["regret"], path["talent_loss_q4"], marker="o", color=colour, label=family)
            end = path.iloc[-1]
            ends.append((end["regret"], end["talent_loss_q4"], family))
        all_ends.append(ends)
        ax.plot(
            start["regret"],
            start["talent_loss_q4"],
            marker="o",
            color=INK,
            linestyle="",
            markersize=7,
            zorder=5,
        )
        ax.annotate(
            "no mitigation",
            (start["regret"], start["talent_loss_q4"]),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            color=INK,
            fontsize=8,
        )
        ax.set_title(f"Base selector: {base}", loc="left")
        ax.set_xlabel("Regret (lower = more efficient)")
    for ax, ends in zip(axes, all_ends, strict=True):
        _label_ends(ax, ends, left=frozenset({"RA-adjusted"}))
    axes[0].set_ylabel("Talent loss among Q4 athletes")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"RQ3. Mitigations at increasing strength (feedback loop on, c = {exposure:g})",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        -0.02,
        "Each path starts at the unmitigated selector; points are mean over "
        "seeds. Lower-left is better on both axes; the oracle is at (0, 0).",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / "rq3_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sensitivity_figure(runs: pd.DataFrame, out: Path) -> None:
    """Paired change in regret (cost) and in Q4 talent loss (benefit) of each mitigation, as the
    persistent share kappa of the maturation advantage grows."""
    data = runs[runs["parameter"] == "adult_maturation_weight"]
    families = [("RA-adjusted", SERIES[0]), ("quota", SERIES[1]), ("bio-banding", SERIES[2])]
    panels = [
        ("regret", "Change in regret (above 0 = efficiency cost)"),
        ("talent_loss_q4", "Change in talent loss among Q4 athletes"),
    ]
    scout = data[data["selector"] == "scout"].set_index(["level", "seed"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        ax.axhline(0.0, color=AXIS, linewidth=1)
        ends = []
        for family, colour in families:
            mitigated = data[data["selector"] == family].set_index(["level", "seed"])
            delta = (mitigated[metric] - scout[metric]).groupby("level").mean()
            ax.plot(delta.index, delta.values, marker="o", color=colour, label=family)
            ends.append((delta.index[-1], delta.values[-1], family))
        _label_ends(ax, ends, gap=0.08)
        ax.set_title(title, loc="left")
        ax.set_xlabel("kappa (share of the maturation advantage that persists)")
        ax.set_xlim(right=data["level"].max() * 1.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Sensitivity. Mitigations become costly when maturation persists into adulthood",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        -0.02,
        "Paired change with respect to the scout, mean over seeds; feedback loop on (c = 1).",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / "sensitivity_kappa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rq1_figure(pd.read_csv(args.results / "rq1_runs_linear.csv"), args.out)
    rq2_figure(pd.read_csv(args.results / "rq2_stages.csv"), args.out)
    rq3_figure(pd.read_csv(args.results / "rq3_runs_linear.csv"), args.out)
    sensitivity_figure(pd.read_csv(args.results / "sensitivity_oat.csv"), args.out)
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
