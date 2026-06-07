#!/usr/bin/env python3
"""Plot the combined PM2.5, NO2, and O3 RMSE station-map figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr

from S1 import (
    CNEMC_ROOT,
    DEFAULT_DOMAIN,
    INPUT_ROOT,
    OUT_DIR,
    R_DRY_AIR,
    STATION_LIST,
    STUDY_END,
    STUDY_START,
    add_base_map,
    cams_date_dirs,
    cnemc_city_columns,
    load_china_geoms,
    load_china_province_lines,
    load_city_points,
    load_cnemc_for_date,
    load_country_geoms,
)


OUTPUT_NAME = "S2.pdf"
UNIT_LABEL = r"$\mathbf{(\mu g\ m^{-3})}$"
SECOND_COLUMN_SHIFT = -0.026


@dataclass(frozen=True)
class Species:
    key: str
    label: str
    obs_type: str
    cams_file: str
    cams_var: str
    obs_scale: float = 1.0
    pressure_level: int | None = None


SPECIES = [
    Species("pm25", r"$\mathbf{PM}_{\mathbf{2.5}}$", "PM2.5_24h", "sfc", "pm2p5"),
    Species("no2", r"$\mathbf{NO}_{\mathbf{2}}$", "NO2_24h", "atmos", "no2", pressure_level=1000),
    Species("o3", r"$\mathbf{O}_{\mathbf{3}}$", "O3_8h", "atmos", "go3", pressure_level=1000),
]


def lonlat_to_km(lon: np.ndarray, lat: np.ndarray, ref_lat: float) -> np.ndarray:
    x = lon * 111.32 * np.cos(np.deg2rad(ref_lat))
    y = lat * 110.57
    return np.column_stack([x, y])


def gaussian_smooth_points(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    valid = np.isfinite(values)
    smoothed = np.full_like(values, np.nan, dtype=float)
    if valid.sum() < 2:
        smoothed[valid] = values[valid]
        return smoothed

    ref_lat = float(np.nanmean(lat[valid]))
    target_xy = lonlat_to_km(lon, lat, ref_lat)
    source_xy = target_xy[valid]
    source_values = values[valid]
    diff = target_xy[:, None, :] - source_xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    weights = np.exp(-0.5 * (dist / radius_km) ** 2)
    weights[dist > radius_km * 3.0] = 0.0
    weight_sum = weights.sum(axis=1)
    ok = weight_sum > 0
    smoothed[ok] = (weights[ok] @ source_values) / weight_sum[ok]
    return smoothed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--cnemc-root", type=Path, default=CNEMC_ROOT)
    parser.add_argument("--station-list", type=Path, default=STATION_LIST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--start", default=STUDY_START)
    parser.add_argument("--end", default=STUDY_END)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_DOMAIN[0])
    parser.add_argument("--lon-max", type=float, default=DEFAULT_DOMAIN[1])
    parser.add_argument("--lat-min", type=float, default=DEFAULT_DOMAIN[2])
    parser.add_argument("--lat-max", type=float, default=DEFAULT_DOMAIN[3])
    parser.add_argument("--smooth-radius-km", type=float, default=180.0)
    parser.add_argument("--paired-values", default=None)
    parser.add_argument("--station-values", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output", default=OUTPUT_NAME)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def interpolate_cams_for_date(date_dir: Path, points: pd.DataFrame) -> dict[str, np.ndarray]:
    lat_points = xr.DataArray(points["lat"].to_numpy(dtype=float), dims="city")
    lon_points = xr.DataArray(points["lon"].to_numpy(dtype=float), dims="city")
    out: dict[str, np.ndarray] = {}

    if any(spec.cams_file == "sfc" for spec in SPECIES):
        with xr.open_dataset(date_dir / "sfc.nc") as sfc:
            out["times"] = pd.to_datetime(sfc["forecast_reference_time"].values).to_numpy(dtype="datetime64[ns]")
            for spec in SPECIES:
                if spec.cams_file != "sfc":
                    continue
                values = (
                    sfc[spec.cams_var]
                    .isel(forecast_period=0)
                    .interp(latitude=lat_points, longitude=lon_points)
                    .to_numpy()
                    * 1.0e9
                )
                out[spec.key] = np.asarray(values, dtype=float)

    if any(spec.cams_file == "atmos" for spec in SPECIES):
        with xr.open_dataset(date_dir / "atmos.nc") as atmos:
            if "times" not in out:
                out["times"] = pd.to_datetime(atmos["forecast_reference_time"].values).to_numpy(dtype="datetime64[ns]")
            for spec in SPECIES:
                if spec.cams_file != "atmos":
                    continue
                if spec.pressure_level is None:
                    raise ValueError(f"{spec.key} requires a pressure level")
                temperature = (
                    atmos["t"]
                    .isel(forecast_period=0)
                    .sel(pressure_level=spec.pressure_level)
                    .interp(latitude=lat_points, longitude=lon_points)
                    .to_numpy()
                )
                values = (
                    atmos[spec.cams_var]
                    .isel(forecast_period=0)
                    .sel(pressure_level=spec.pressure_level)
                    .interp(latitude=lat_points, longitude=lon_points)
                    .to_numpy()
                )
                air_density = (spec.pressure_level * 100.0) / (R_DRY_AIR * temperature)
                out[spec.key] = np.asarray(values * air_density * 1.0e9, dtype=float)
    return out


def build_timewise_paired_values(args: argparse.Namespace) -> pd.DataFrame:
    domain = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    cities_all = cnemc_city_columns(args.cnemc_root, args.start, args.end)
    points = load_city_points(args.station_list, cities_all, domain)
    cities = points["city"].tolist()
    lon = points["lon"].to_numpy(dtype=float)
    lat = points["lat"].to_numpy(dtype=float)
    obs_types = [spec.obs_type for spec in SPECIES]

    records = []
    cnemc_cache: dict[str, pd.DataFrame] = {}
    for date_dir in cams_date_dirs(args.input_root, args.start, args.end):
        cams = interpolate_cams_for_date(date_dir, points)
        for time_index, utc_time64 in enumerate(cams["times"]):
            utc_time = pd.Timestamp(utc_time64)
            bj_time = utc_time + pd.Timedelta(hours=8)
            bj_date = bj_time.strftime("%Y%m%d")
            bj_hour = int(bj_time.hour)

            if bj_date not in cnemc_cache:
                cnemc_cache[bj_date] = load_cnemc_for_date(args.cnemc_root, bj_date, cities, obs_types)
            obs_df = cnemc_cache[bj_date]

            for spec in SPECIES:
                row = obs_df[(obs_df["hour"] == bj_hour) & (obs_df["type"] == spec.obs_type)]
                if row.empty:
                    continue
                obs_values = row.iloc[0][cities].to_numpy(dtype=float) * spec.obs_scale
                smoothed_values = gaussian_smooth_points(lon, lat, obs_values, args.smooth_radius_km)
                cams_values = np.asarray(cams[spec.key][time_index, :], dtype=float)
                valid = np.isfinite(cams_values) & np.isfinite(obs_values) & np.isfinite(smoothed_values)

                for idx in np.where(valid)[0]:
                    raw_error = cams_values[idx] - obs_values[idx]
                    smoothed_error = cams_values[idx] - smoothed_values[idx]
                    records.append(
                        {
                            "pollutant": spec.key,
                            "label": spec.label,
                            "city": cities[idx],
                            "lon": lon[idx],
                            "lat": lat[idx],
                            "utc_time": utc_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "bj_date": bj_date,
                            "bj_hour": bj_hour,
                            "cams": float(cams_values[idx]),
                            "cnemc": float(obs_values[idx]),
                            "cnemc_smoothed": float(smoothed_values[idx]),
                            "error_raw": float(raw_error),
                            "error_smoothed": float(smoothed_error),
                        }
                    )

    paired = pd.DataFrame(records)
    if paired.empty:
        raise RuntimeError("No paired samples were found")
    return paired


def rmse(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    return float(np.sqrt(np.nanmean(values * values)))


def summarize_station_rmse(paired: pd.DataFrame) -> pd.DataFrame:
    station = (
        paired.groupby(["pollutant", "label", "city", "lon", "lat"], as_index=False)
        .agg(
            n=("error_raw", "count"),
            cams_mean=("cams", "mean"),
            cnemc_mean=("cnemc", "mean"),
            cnemc_smoothed_mean=("cnemc_smoothed", "mean"),
            bias_raw_mean=("error_raw", "mean"),
            bias_smoothed_mean=("error_smoothed", "mean"),
            rmse_raw=("error_raw", rmse),
            rmse_smoothed=("error_smoothed", rmse),
        )
        .sort_values(["pollutant", "city"])
        .reset_index(drop=True)
    )
    station["rmse_smoothed_minus_raw"] = station["rmse_smoothed"] - station["rmse_raw"]
    station["rmse_improved"] = station["rmse_smoothed_minus_raw"] < 0
    return station


def summarize_overall(paired: pd.DataFrame, station: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SPECIES:
        data = paired[paired["pollutant"].eq(spec.key)]
        stat = station[station["pollutant"].eq(spec.key)]
        for comparison, error_col, obs_col in [
            ("raw_cnemc", "error_raw", "cnemc"),
            ("timewise_smoothed_cnemc", "error_smoothed", "cnemc_smoothed"),
        ]:
            rows.append(
                {
                    "pollutant": spec.key,
                    "comparison": comparison,
                    "paired_samples": len(data),
                    "cities": stat["city"].nunique(),
                    "times": data["utc_time"].nunique(),
                    "cams_mean": data["cams"].mean(),
                    "obs_mean": data[obs_col].mean(),
                    "mean_bias": data[error_col].mean(),
                    "rmse_all_pairs": rmse(data[error_col]),
                    "mean_station_rmse": stat[f"rmse_{'raw' if error_col == 'error_raw' else 'smoothed'}"].mean(),
                    "station_improved_fraction": float(stat["rmse_improved"].mean())
                    if comparison == "timewise_smoothed_cnemc"
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def load_or_build_station_rmse(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_path = args.out_dir / args.station_values if args.station_values else None
    summary_path = args.out_dir / args.summary if args.summary else None
    if station_path and summary_path and station_path.exists() and summary_path.exists() and not args.rebuild:
        station = pd.read_csv(station_path)
        summary = pd.read_csv(summary_path)
    else:
        paired = build_timewise_paired_values(args)
        station = summarize_station_rmse(paired)
        summary = summarize_overall(paired, station)
        if args.paired_values:
            paired.to_csv(args.out_dir / args.paired_values, index=False)
        if station_path:
            station.to_csv(station_path, index=False)
        if summary_path:
            summary.to_csv(summary_path, index=False)

    required = ["lon", "lat", "rmse_raw", "rmse_smoothed", "rmse_smoothed_minus_raw"]
    for col in required:
        station[col] = pd.to_numeric(station[col], errors="coerce")
    station = station.dropna(subset=required).reset_index(drop=True)
    if station.empty:
        raise RuntimeError("No valid station RMSE values were found")
    return station, summary


def map_context(args: argparse.Namespace):
    map_extent = (args.lon_min - 1.5, args.lon_max + 1.5, args.lat_min - 1.0, args.lat_max + 1.0)
    return (
        map_extent,
        load_country_geoms(map_extent),
        load_china_geoms(),
        load_china_province_lines(map_extent),
    )


def nice_step(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0**exponent)
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0**exponent))


def positive_levels(values: np.ndarray, target_intervals: int = 8) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, 6)
    vmax = float(np.nanmax(finite))
    step = nice_step(vmax / target_intervals)
    vmax = max(step, np.ceil(vmax / step) * step)
    return np.arange(0.0, vmax + step * 0.5, step)


def diverging_limit(values: np.ndarray, target_intervals: int = 5) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    vmax = float(np.nanmax(finite))
    step = nice_step(vmax / target_intervals)
    return float(max(step, np.ceil(vmax / step) * step))


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.3,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def style_colorbar(cb, unit_position: str = "center") -> None:
    cb.locator = MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=12.5, pad=3, length=4.5, width=1.15)
    for label in cb.ax.get_yticklabels():
        label.set_fontweight("bold")
    if unit_position == "right":
        cb.ax.text(
            1.22,
            1.018,
            UNIT_LABEL,
            transform=cb.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.8,
            fontweight="bold",
        )
    else:
        cb.ax.set_title(UNIT_LABEL, fontsize=10.8, fontweight="bold", pad=7)


def scatter_station_map(
    ax,
    station: pd.DataFrame,
    values: np.ndarray,
    cmap,
    norm,
    label: str,
    map_extent: tuple[float, float, float, float],
    country_geoms: list,
    china_geoms: list,
    province_lines: list,
    show_bottom_labels: bool,
    show_left_labels: bool,
) -> None:
    pc = ccrs.PlateCarree()
    add_base_map(
        ax,
        map_extent,
        country_geoms,
        china_geoms,
        province_lines,
        show_bottom_labels=show_bottom_labels,
        show_left_labels=show_left_labels,
    )
    ax.scatter(
        station["lon"],
        station["lat"],
        c=values,
        s=36,
        marker="o",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.18,
        alpha=0.95,
        transform=pc,
        zorder=10,
    )
    add_panel_label(ax, label)


def shift_axes_horizontally(axes, dx: float) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])


def plot_combined(args: argparse.Namespace, station: pd.DataFrame) -> Path:
    map_extent, country_geoms, china_geoms, province_lines = map_context(args)
    pc = ccrs.PlateCarree()

    row_count = len(SPECIES)
    fig = plt.figure(figsize=(11.8, 11.4))
    gs = GridSpec(
        row_count,
        5,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.048, 1.0, 0.048],
        height_ratios=[1.0] * row_count,
        wspace=0.012,
        hspace=0.18,
    )

    panel_letters = iter("abcdefghi")
    for row_index, spec in enumerate(SPECIES):
        data = station[station["pollutant"].eq(spec.key)].copy()
        rmse_raw = data["rmse_raw"].to_numpy(dtype=float)
        rmse_smoothed = data["rmse_smoothed"].to_numpy(dtype=float)
        rmse_diff = data["rmse_smoothed_minus_raw"].to_numpy(dtype=float)

        rmse_cmap = plt.get_cmap("YlOrRd")
        rmse_levels = positive_levels(np.concatenate([rmse_raw, rmse_smoothed]))
        rmse_norm = BoundaryNorm(rmse_levels, rmse_cmap.N)
        diff_cmap = plt.get_cmap("RdBu_r")
        diff_vlim = diverging_limit(rmse_diff)
        diff_norm = TwoSlopeNorm(vmin=-diff_vlim, vcenter=0.0, vmax=diff_vlim)

        ax_raw = fig.add_subplot(gs[row_index, 0], projection=pc)
        ax_smooth = fig.add_subplot(gs[row_index, 1], projection=pc)
        ax_diff = fig.add_subplot(gs[row_index, 3], projection=pc)
        cax_rmse = fig.add_subplot(gs[row_index, 2])
        cax_diff = fig.add_subplot(gs[row_index, 4])

        show_bottom = row_index == row_count - 1
        diff_label = r"$\mathbf{RMSE}_{\mathbf{smooth}}-\mathbf{RMSE}_{\mathbf{raw}}$"
        labels = [
            f"({next(panel_letters)}) {spec.label} raw RMSE",
            f"({next(panel_letters)}) {spec.label} smoothed RMSE",
            f"({next(panel_letters)}) {spec.label} {diff_label}",
        ]
        scatter_station_map(
            ax_raw,
            data,
            rmse_raw,
            rmse_cmap,
            rmse_norm,
            labels[0],
            map_extent,
            country_geoms,
            china_geoms,
            province_lines,
            show_bottom_labels=show_bottom,
            show_left_labels=True,
        )
        scatter_station_map(
            ax_smooth,
            data,
            rmse_smoothed,
            rmse_cmap,
            rmse_norm,
            labels[1],
            map_extent,
            country_geoms,
            china_geoms,
            province_lines,
            show_bottom_labels=show_bottom,
            show_left_labels=False,
        )
        scatter_station_map(
            ax_diff,
            data,
            rmse_diff,
            diff_cmap,
            diff_norm,
            labels[2],
            map_extent,
            country_geoms,
            china_geoms,
            province_lines,
            show_bottom_labels=show_bottom,
            show_left_labels=False,
        )

        rmse_sm = ScalarMappable(norm=rmse_norm, cmap=rmse_cmap)
        rmse_sm.set_array([])
        cb_rmse = fig.colorbar(rmse_sm, cax=cax_rmse, orientation="vertical")
        style_colorbar(cb_rmse, unit_position="center")
        shift_axes_horizontally([ax_smooth, cax_rmse], SECOND_COLUMN_SHIFT)

        diff_sm = ScalarMappable(norm=diff_norm, cmap=diff_cmap)
        diff_sm.set_array([])
        cb_diff = fig.colorbar(diff_sm, cax=cax_diff, orientation="vertical")
        style_colorbar(cb_diff, unit_position="right")

    out_path = args.out_dir / args.output
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    station, summary = load_or_build_station_rmse(args)
    out_path = plot_combined(args, station)

    print(summary.to_string(index=False))
    for spec in SPECIES:
        data = station[station["pollutant"].eq(spec.key)]
        diff = data["rmse_smoothed_minus_raw"].to_numpy(dtype=float)
        print(
            f"{spec.key}: cities={len(data)}, "
            f"raw_rmse_mean={data['rmse_raw'].mean():.3f}, "
            f"smoothed_rmse_mean={data['rmse_smoothed'].mean():.3f}, "
            f"smoothed-minus-raw={np.nanmean(diff):.3f}, "
            f"improved={int(np.sum(diff < 0))}/{len(data)}"
        )
    print(f"PDF written: {out_path}")


if __name__ == "__main__":
    main()
