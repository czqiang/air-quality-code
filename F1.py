#!/usr/bin/env python3
"""Figure 1: station-mean pollutant time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "look" / "wrfchem_aurora_cnemc_station_timeseries_wrf1000hpa.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "F1.pdf"

POLLUTANTS = [
    ("pm25", "PM$_{2.5}$"),
    ("pm10", "PM$_{10}$"),
    ("co", "CO"),
    ("so2", "SO$_2$"),
    ("o3", "O$_3$"),
    ("no2", "NO$_2$"),
]

MODEL_STYLES = {
    "CNEMC": {
        "column_prefix": "obs",
        "color": "#111111",
        "linewidth": 2.8,
        "linestyle": "-",
        "marker": None,
        "zorder": 3,
    },
    "WRF-Chem": {
        "column_prefix": "wrf",
        "color": "#1f77b4",
        "linewidth": 2.6,
        "linestyle": "-",
        "marker": None,
        "zorder": 2,
    },
    "Aurora": {
        "column_prefix": "aurora",
        "color": "#d62728",
        "linewidth": 2.2,
        "linestyle": "--",
        "marker": "o",
        "markersize": 4.8,
        "markeredgewidth": 0.0,
        "zorder": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def set_bold_ticklabels(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def plot_figure(input_csv: Path, output: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv, parse_dates=["time_utc"])
    configure_matplotlib()

    fig, axes = plt.subplots(3, 2, figsize=(14.2, 9.6), sharex=True)
    axes = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, (key, label), panel in zip(axes, POLLUTANTS, panel_labels):
        for model_name, style in MODEL_STYLES.items():
            column = f"{style['column_prefix']}_{key}_ug_m3"
            plot_kwargs = {
                "color": style["color"],
                "linewidth": style["linewidth"],
                "label": model_name,
                "zorder": style["zorder"],
            }
            if style.get("marker"):
                plot_kwargs.update(
                    {
                        "marker": style["marker"],
                        "markersize": style["markersize"],
                        "markeredgewidth": style["markeredgewidth"],
                    }
                )
            plot_df = df[["time_utc", column]].dropna() if model_name == "Aurora" else df[["time_utc", column]]
            ax.plot(plot_df["time_utc"], plot_df[column], linestyle=style["linestyle"], **plot_kwargs)

        ax.set_title(f"{panel} {label}", loc="left", pad=7)
        ax.set_ylabel("($\\mu$g m$^{-3}$)")
        ax.grid(True, which="major", color="0.86", linewidth=0.9)
        ax.grid(True, which="minor", axis="x", color="0.92", linewidth=0.65)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax.tick_params(axis="both", which="major", width=1.45, length=5.5, direction="out")
        ax.tick_params(axis="x", which="minor", width=1.1, length=3.5, direction="out")
        for spine in ax.spines.values():
            spine.set_linewidth(1.35)
        set_bold_ticklabels(ax)

    axes[0].legend(loc="upper right", frameon=False, handlelength=2.4, borderaxespad=0.2)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.65, h_pad=1.0, w_pad=0.8)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_figure(args.input_csv, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
