#!/usr/bin/env python3
"""Figure 0: WRF-Chem simulation domain map."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from netCDF4 import Dataset
from shapely.geometry import Polygon, box


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WRFOUT = Path(
    "/home/xeon/wrf/cases/eastchina_20240206_20240212_20km/output/"
    "wrfchem_latest/wrfout_d01_2024-02-06_00:00:00"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "F0.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrfout", type=Path, default=DEFAULT_WRFOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_wrf_grid(wrfout: Path) -> tuple[np.ndarray, np.ndarray]:
    if not wrfout.exists():
        raise FileNotFoundError(wrfout)

    with Dataset(wrfout) as ds:
        lats = np.asarray(ds.variables["XLAT"][0], dtype=float)
        lons = np.asarray(ds.variables["XLONG"][0], dtype=float)
    return lons, lats


def wrf_boundary(lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.concatenate([lons[0, :], lons[1:, -1], lons[-1, -2::-1], lons[-2:0:-1, 0], lons[0, :1]])
    lat = np.concatenate([lats[0, :], lats[1:, -1], lats[-1, -2::-1], lats[-2:0:-1, 0], lats[0, :1]])
    return lon, lat


def padded_extent(lons: np.ndarray, lats: np.ndarray) -> list[float]:
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    return [lon_min - 0.85, lon_max + 0.85, lat_min - 0.75, lat_max + 0.75]


def china_province_records(extent: list[float]) -> list:
    shp = shapereader.natural_earth("10m", "cultural", "admin_1_states_provinces_lakes")
    area = box(extent[0], extent[2], extent[1], extent[3])
    records = []
    for record in shapereader.Reader(shp).records():
        attrs = record.attributes
        geom = record.geometry
        if attrs.get("adm0_a3") != "CHN" or geom is None or not geom.intersects(area):
            continue
        records.append(record)
    return records


def province_label_points(records: list, domain_polygon: Polygon) -> list[tuple[str, float, float]]:
    labels = []
    for record in records:
        geom = record.geometry.intersection(domain_polygon)
        if geom.is_empty or geom.area < 0.03:
            continue
        name = record.attributes.get("name_en") or record.attributes.get("name")
        point = geom.representative_point()
        labels.append((name, float(point.x), float(point.y)))
    return labels


def add_base_map(ax, province_records: list) -> None:
    ocean = "#d9edf8"
    land = "#f3efe3"
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor=ocean, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=land, edgecolor="none", zorder=0.5)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor=ocean, edgecolor="0.45", linewidth=0.35, zorder=1)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#78a9c9", linewidth=0.45, alpha=0.75, zorder=1.5)
    ax.coastlines(resolution="10m", color="0.10", linewidth=1.45, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="0.10", linewidth=1.15, zorder=3)
    ax.add_geometries(
        [record.geometry for record in province_records],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="0.18",
        linewidth=1.15,
        zorder=3.4,
    )


def add_gridlines(ax) -> None:
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(114, 127, 2),
        ylocs=np.arange(26, 37, 2),
        linewidth=0.75,
        color="0.45",
        alpha=0.42,
        linestyle="--",
        zorder=2,
    )


def add_axis_ticks(ax, extent: list[float]) -> None:
    ax.set_xticks(np.arange(114, 127, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(28, 36, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.6, length=6.5, direction="out")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.45)


def draw_domain(ax, boundary_lon: np.ndarray, boundary_lat: np.ndarray) -> None:
    domain_color = "#b2182b"
    ax.fill(
        boundary_lon,
        boundary_lat,
        facecolor=domain_color,
        edgecolor="none",
        alpha=0.08,
        transform=ccrs.PlateCarree(),
        zorder=2.7,
    )
    ax.plot(
        boundary_lon,
        boundary_lat,
        color=domain_color,
        linewidth=3.2,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.text(
        124.35,
        34.35,
        "WRF-Chem domain",
        transform=ccrs.PlateCarree(),
        ha="right",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=domain_color,
        path_effects=[pe.withStroke(linewidth=3.6, foreground="white")],
        zorder=6,
    )


def add_province_labels(ax, labels: list[tuple[str, float, float]]) -> None:
    for name, lon, lat in labels:
        size = 11.5 if name == "Shanghai" else 14
        ax.text(
            lon,
            lat,
            name,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold",
            color="0.16",
            path_effects=[pe.withStroke(linewidth=3.2, foreground="white")],
            zorder=6,
        )


def plot_domain_map(wrfout: Path, output: Path) -> None:
    configure_matplotlib()
    lons, lats = read_wrf_grid(wrfout)
    extent = padded_extent(lons, lats)
    boundary_lon, boundary_lat = wrf_boundary(lons, lats)
    domain_polygon = Polygon(zip(boundary_lon, boundary_lat))
    province_records = china_province_records(extent)
    labels = province_label_points(province_records, domain_polygon)

    fig = plt.figure(figsize=(8.5, 6.9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    add_base_map(ax, province_records)
    add_gridlines(ax)
    add_axis_ticks(ax, extent)
    draw_domain(ax, boundary_lon, boundary_lat)
    add_province_labels(ax, labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_domain_map(args.wrfout, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
