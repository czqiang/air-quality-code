#!/usr/bin/env python3
"""Plot gridded PM2.5 RMSE maps for CAMS against CNEMC and CHAP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon
import xarray as xr

from S1 import (
    AVERAGE_OBS_TYPES,
    CNEMC_ROOT,
    DEFAULT_DOMAIN,
    INPUT_ROOT,
    OUT_DIR,
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


PROJECT_ROOT = Path("/home/xeon/lcy-front")
CHAP_ROOT = PROJECT_ROOT / "chap"
OUTPUT_NAME = "S3.pdf"
PM25_OBS_TYPE = AVERAGE_OBS_TYPES["pm25"]
UNIT_LABEL = r"$\mathbf{(\mu g\ m^{-3})}$"
TAIWAN_POLYGON = Polygon(
    [
        (120.00, 21.75),
        (120.22, 22.30),
        (120.45, 22.85),
        (120.65, 23.50),
        (120.88, 24.05),
        (121.12, 24.55),
        (121.42, 25.20),
        (121.85, 25.35),
        (122.05, 25.05),
        (121.82, 24.45),
        (121.55, 23.70),
        (121.25, 23.05),
        (120.92, 22.35),
        (120.48, 21.88),
    ]
).buffer(0.16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--cnemc-root", type=Path, default=CNEMC_ROOT)
    parser.add_argument("--chap-root", type=Path, default=CHAP_ROOT)
    parser.add_argument("--station-list", type=Path, default=STATION_LIST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--start", default=STUDY_START)
    parser.add_argument("--end", default=STUDY_END)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_DOMAIN[0])
    parser.add_argument("--lon-max", type=float, default=DEFAULT_DOMAIN[1])
    parser.add_argument("--lat-min", type=float, default=DEFAULT_DOMAIN[2])
    parser.add_argument("--lat-max", type=float, default=DEFAULT_DOMAIN[3])
    parser.add_argument("--output", default=OUTPUT_NAME)
    return parser.parse_args()


def target_cams_grid(input_root: Path, start: str, end: str, domain: tuple[float, float, float, float]):
    date_dirs = cams_date_dirs(input_root, start, end)
    if not date_dirs:
        raise RuntimeError("No CAMS date directories were found")
    lon_min, lon_max, lat_min, lat_max = domain
    with xr.open_dataset(date_dirs[0] / "sfc.nc") as ds:
        lat = np.asarray(ds["latitude"].to_numpy(), dtype=float)
        lon = np.asarray(ds["longitude"].to_numpy(), dtype=float)
    target_lat = lat[(lat >= lat_min) & (lat <= lat_max)]
    target_lon = lon[(lon >= lon_min) & (lon <= lon_max)]
    if target_lat.size == 0 or target_lon.size == 0:
        raise RuntimeError("The requested domain does not overlap the CAMS grid")
    return target_lat, target_lon, date_dirs


def interpolate_cams_pm25(date_dir: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(date_dir / "sfc.nc") as ds:
        da = (
            ds["pm2p5"]
            .isel(forecast_period=0)
            .sel(latitude=target_lat, longitude=target_lon)
            .to_numpy()
            * 1.0e9
        )
        times = pd.to_datetime(ds["forecast_reference_time"].values).to_numpy(dtype="datetime64[ns]")
    return times, np.asarray(da, dtype=float)


def interpolate_station_field(
    points: pd.DataFrame,
    values: np.ndarray,
    grid_lon_2d: np.ndarray,
    grid_lat_2d: np.ndarray,
) -> np.ndarray:
    lon = points["lon"].to_numpy(dtype=float)
    lat = points["lat"].to_numpy(dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
    if valid.sum() < 3:
        return np.full_like(grid_lon_2d, np.nan, dtype=float)

    station_xy = np.column_stack([lon[valid], lat[valid]])
    grid_xy = np.column_stack([grid_lon_2d.ravel(), grid_lat_2d.ravel()])
    linear = griddata(station_xy, values[valid], grid_xy, method="linear")
    nearest = griddata(station_xy, values[valid], grid_xy, method="nearest")
    field = np.where(np.isfinite(linear), linear, nearest)
    return field.reshape(grid_lon_2d.shape)


def chap_file(chap_root: Path, token: str) -> Path:
    path = chap_root / token[:4] / f"CHAP_PM2.5_D1K_{token}_V4.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def interpolate_chap_pm25(chap_root: Path, token: str, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    path = chap_file(chap_root, token)
    lon_min = float(np.nanmin(target_lon)) - 0.2
    lon_max = float(np.nanmax(target_lon)) + 0.2
    lat_min = float(np.nanmin(target_lat)) - 0.2
    lat_max = float(np.nanmax(target_lat)) + 0.2
    with xr.open_dataset(path) as ds:
        subset = ds["PM2.5"].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        subset = subset.rename({"lat": "latitude", "lon": "longitude"}).sortby("latitude")
        field = subset.interp(latitude=target_lat, longitude=target_lon).to_numpy()
    return np.asarray(field, dtype=float)


def rmse_from_sums(sum_sq: np.ndarray, count: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sum_sq / count)


def compute_rmse_fields(args: argparse.Namespace):
    domain = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    target_lat, target_lon, date_dirs = target_cams_grid(args.input_root, args.start, args.end, domain)
    grid_lon_2d, grid_lat_2d = np.meshgrid(target_lon, target_lat)

    cities = cnemc_city_columns(args.cnemc_root, args.start, args.end)
    points = load_city_points(args.station_list, cities, domain)
    city_names = points["city"].tolist()
    if not city_names:
        raise RuntimeError("No CNEMC city points were found in the requested domain")

    sum_sq_cnemc = np.zeros_like(grid_lon_2d, dtype=float)
    count_cnemc = np.zeros_like(grid_lon_2d, dtype=int)
    sum_sq_chap = np.zeros_like(grid_lon_2d, dtype=float)
    count_chap = np.zeros_like(grid_lon_2d, dtype=int)
    island_mask = taiwan_mask_from_grid(grid_lon_2d, grid_lat_2d)
    pair_mask = ~island_mask

    cnemc_cache: dict[str, pd.DataFrame] = {}
    chap_cache: dict[str, np.ndarray] = {}
    cnemc_pair_chunks: list[np.ndarray] = []
    cnemc_cams_chunks: list[np.ndarray] = []
    chap_pair_chunks: list[np.ndarray] = []
    chap_cams_chunks: list[np.ndarray] = []
    matched_times = 0
    matched_chap_times = 0

    for date_dir in date_dirs:
        token = date_dir.name.replace("-", "")
        times, cams_pm25 = interpolate_cams_pm25(date_dir, target_lat, target_lon)
        if token not in chap_cache:
            chap_cache[token] = interpolate_chap_pm25(args.chap_root, token, target_lat, target_lon)
        chap_field = chap_cache[token]

        for time_index, utc_time64 in enumerate(times):
            utc_time = pd.Timestamp(utc_time64)
            bj_time = utc_time + pd.Timedelta(hours=8)
            bj_date = bj_time.strftime("%Y%m%d")
            bj_hour = int(bj_time.hour)
            if bj_date not in cnemc_cache:
                cnemc_cache[bj_date] = load_cnemc_for_date(args.cnemc_root, bj_date, city_names, [PM25_OBS_TYPE])
            obs_df = cnemc_cache[bj_date]
            row = obs_df[(obs_df["hour"] == bj_hour) & (obs_df["type"] == PM25_OBS_TYPE)]
            if row.empty:
                continue

            obs_values = row.iloc[0][city_names].to_numpy(dtype=float)
            cnemc_field = interpolate_station_field(points, obs_values, grid_lon_2d, grid_lat_2d)
            cams_field = np.asarray(cams_pm25[time_index, :, :], dtype=float)

            valid_cnemc = np.isfinite(cams_field) & np.isfinite(cnemc_field)
            sum_sq_cnemc[valid_cnemc] += (cams_field[valid_cnemc] - cnemc_field[valid_cnemc]) ** 2
            count_cnemc[valid_cnemc] += 1
            matched_times += 1

            valid_cnemc_pairs = valid_cnemc & pair_mask
            cnemc_pair_chunks.append(cnemc_field[valid_cnemc_pairs])
            cnemc_cams_chunks.append(cams_field[valid_cnemc_pairs])

            valid_chap = np.isfinite(cams_field) & np.isfinite(chap_field)
            sum_sq_chap[valid_chap] += (cams_field[valid_chap] - chap_field[valid_chap]) ** 2
            count_chap[valid_chap] += 1
            matched_chap_times += 1

            valid_chap_pairs = valid_chap & pair_mask
            chap_pair_chunks.append(chap_field[valid_chap_pairs])
            chap_cams_chunks.append(cams_field[valid_chap_pairs])

    rmse_cnemc = rmse_from_sums(sum_sq_cnemc, count_cnemc)
    rmse_chap = rmse_from_sums(sum_sq_chap, count_chap)

    common_mask = np.isfinite(rmse_chap)
    rmse_cnemc = np.where(common_mask, rmse_cnemc, np.nan)
    rmse_chap = np.where(common_mask, rmse_chap, np.nan)

    metadata = {
        "cams_dates": len(date_dirs),
        "matched_times": matched_times,
        "matched_chap_times": matched_chap_times,
        "cities": len(points),
        "lat_count": len(target_lat),
        "lon_count": len(target_lon),
    }
    pairs = {
        "cnemc": np.concatenate(cnemc_pair_chunks) if cnemc_pair_chunks else np.array([], dtype=float),
        "cams_cnemc": np.concatenate(cnemc_cams_chunks) if cnemc_cams_chunks else np.array([], dtype=float),
        "chap": np.concatenate(chap_pair_chunks) if chap_pair_chunks else np.array([], dtype=float),
        "cams_chap": np.concatenate(chap_cams_chunks) if chap_cams_chunks else np.array([], dtype=float),
    }
    metadata["cnemc_pairs"] = len(pairs["cnemc"])
    metadata["chap_pairs"] = len(pairs["chap"])
    return target_lat, target_lon, rmse_cnemc, rmse_chap, pairs, metadata


def positive_levels(values: np.ndarray, target_intervals: int = 8) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, target_intervals + 1)
    vmax = float(np.nanmax(finite))
    step = 5.0 if vmax <= 80 else 10.0
    vmax = max(step, np.ceil(vmax / step) * step)
    return np.arange(0.0, vmax + step * 0.5, step)


def diverging_levels(values: np.ndarray) -> np.ndarray:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return np.linspace(-1.0, 1.0, 9)
    vmax = float(np.nanmax(finite))
    step = 2.0 if vmax <= 20 else 5.0
    vmax = max(step, np.ceil(vmax / step) * step)
    return np.arange(-vmax, vmax + step * 0.5, step)


def taiwan_mask_from_grid(grid_lon_2d: np.ndarray, grid_lat_2d: np.ndarray) -> np.ndarray:
    points = zip(grid_lon_2d.ravel(), grid_lat_2d.ravel(), strict=True)
    island = np.fromiter((TAIWAN_POLYGON.contains(Point(lon, lat)) for lon, lat in points), dtype=bool)
    return island.reshape(grid_lon_2d.shape)


def mask_taiwan(values: np.ndarray, grid_lon_2d: np.ndarray, grid_lat_2d: np.ndarray) -> np.ndarray:
    masked = np.array(values, dtype=float, copy=True)
    masked[taiwan_mask_from_grid(grid_lon_2d, grid_lat_2d)] = np.nan
    return masked


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.7,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def style_colorbar(cb) -> None:
    cb.locator = MaxNLocator(nbins=6)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=13.0, pad=3, length=4.5, width=1.15)
    for label in cb.ax.get_yticklabels():
        label.set_fontweight("bold")
    cb.ax.set_title(UNIT_LABEL, fontsize=12.0, fontweight="bold", pad=8)


def style_map_ticks(ax, show_left_labels: bool) -> None:
    pc = ccrs.PlateCarree()
    ax.set_xticks([105, 111, 117, 123], crs=pc)
    ax.set_yticks([20, 25, 30, 35, 40], crs=pc)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="x", labelsize=13.5, width=0, length=0, pad=3)
    ax.tick_params(axis="y", labelsize=13.5, width=0, length=0, pad=5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    if not show_left_labels:
        ax.set_yticklabels([])


def clean_pairs(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]


def scatter_limit(*arrays: np.ndarray) -> float:
    return 400.0


def density_histogram(x_values: np.ndarray, y_values: np.ndarray, limit: float, bins: int = 70):
    return np.histogram2d(x_values, y_values, bins=bins, range=[[0.0, limit], [0.0, limit]])


def fit_stats(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, float]:
    if x_values.size < 2 or float(np.nanstd(x_values)) == 0.0:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x_values, y_values, 1)
    corr = np.corrcoef(x_values, y_values)[0, 1]
    return float(slope), float(intercept), float(corr)


def style_density_axis(ax, limit: float, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal", adjustable="box")
    if limit <= 450.0:
        ticks = np.arange(0.0, limit + 1.0, 100.0)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    ax.grid(True, linestyle="--", linewidth=0.45, color="#9e9e9e", alpha=0.45)
    ax.tick_params(axis="both", labelsize=13.0, width=1.25, length=5.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
    ax.set_xlabel(xlabel, fontsize=13.2, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13.2, fontweight="bold")


def style_count_colorbar(cb) -> None:
    cb.ax.tick_params(labelsize=12.5, pad=3, length=4.5, width=1.15)
    for label in cb.ax.get_yticklabels():
        label.set_fontweight("bold")
    cb.ax.set_title("Count", fontsize=12.0, fontweight="bold", pad=8)


def plot_density_panel(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    limit: float,
    hist_norm: LogNorm,
    label: str,
    xlabel: str,
    ylabel: str,
):
    x_values, y_values = clean_pairs(x_values, y_values)
    in_range = (x_values >= 0.0) & (x_values <= limit) & (y_values >= 0.0) & (y_values <= limit)
    x_plot = x_values[in_range]
    y_plot = y_values[in_range]
    hist, x_edges, y_edges = density_histogram(x_plot, y_plot, limit)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_less_equal(hist.T, 0.0),
        cmap=plt.get_cmap("viridis"),
        norm=hist_norm,
        shading="auto",
    )
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", color="#2e2e2e", linewidth=1.45, label="1:1")
    slope, intercept, corr = fit_stats(x_plot, y_plot)
    if np.isfinite(slope):
        x_line = np.array([0.0, limit])
        ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2.1, label="Fit")
        intercept_text = f"+ {intercept:.1f}" if intercept >= 0.0 else f"- {abs(intercept):.1f}"
        stats_text = f"N = {x_plot.size:,}\nr = {corr:.2f}\ny = {slope:.2f}x {intercept_text}"
    else:
        stats_text = f"N = {x_plot.size:,}"
    style_density_axis(ax, limit, xlabel, ylabel)
    add_panel_label(ax, label)
    ax.text(
        0.04,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=20,
    )
    return mesh


def plot_rmse_maps(
    args: argparse.Namespace,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    rmse_cnemc: np.ndarray,
    rmse_chap: np.ndarray,
    pairs: dict[str, np.ndarray],
) -> Path:
    map_extent = (args.lon_min, args.lon_max + 1.5, args.lat_min - 1.0, args.lat_max + 1.0)
    country_geoms = load_country_geoms(map_extent)
    china_geoms = load_china_geoms()
    province_lines = load_china_province_lines(map_extent)

    grid_lon_2d, grid_lat_2d = np.meshgrid(target_lon, target_lat)
    rmse_cnemc_plot = mask_taiwan(rmse_cnemc, grid_lon_2d, grid_lat_2d)
    rmse_chap_plot = mask_taiwan(rmse_chap, grid_lon_2d, grid_lat_2d)
    rmse_diff = rmse_chap_plot - rmse_cnemc_plot

    rmse_levels = positive_levels(np.concatenate([rmse_cnemc_plot.ravel(), rmse_chap_plot.ravel()]))
    rmse_cmap = plt.get_cmap("YlOrRd")
    rmse_norm = BoundaryNorm(rmse_levels, rmse_cmap.N)
    diff_levels = diverging_levels(rmse_diff)
    diff_cmap = plt.get_cmap("RdBu_r")
    diff_norm = BoundaryNorm(diff_levels, diff_cmap.N)
    pc = ccrs.PlateCarree()

    cnemc_x, cnemc_y = clean_pairs(pairs["cnemc"], pairs["cams_cnemc"])
    chap_x, chap_y = clean_pairs(pairs["chap"], pairs["cams_chap"])
    scatter_max = scatter_limit(cnemc_x, cnemc_y, chap_x, chap_y)
    cnemc_hist, _, _ = density_histogram(cnemc_x, cnemc_y, scatter_max)
    chap_hist, _, _ = density_histogram(chap_x, chap_y, scatter_max)
    cnemc_norm = LogNorm(vmin=1.0, vmax=max(1.0, float(np.nanmax(cnemc_hist))))
    chap_norm = LogNorm(vmin=1.0, vmax=max(1.0, float(np.nanmax(chap_hist))))

    fig = plt.figure(figsize=(15.6, 9.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.82])
    map_axes = [
        fig.add_subplot(gs[0, 0:2], projection=pc),
        fig.add_subplot(gs[0, 2:4], projection=pc),
        fig.add_subplot(gs[0, 4:6], projection=pc),
    ]
    density_axes = [
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]
    panels = [
        (map_axes[0], rmse_cnemc_plot, rmse_levels, rmse_cmap, rmse_norm, r"(a) CAMS vs CNEMC PM$_{2.5}$ RMSE", True),
        (map_axes[1], rmse_chap_plot, rmse_levels, rmse_cmap, rmse_norm, r"(b) CAMS vs CHAP PM$_{2.5}$ RMSE", False),
        (map_axes[2], rmse_diff, diff_levels, diff_cmap, diff_norm, r"(c) CHAP RMSE - CNEMC RMSE", False),
    ]
    for ax, values, levels, cmap, norm, label, show_left in panels:
        add_base_map(
            ax,
            map_extent,
            country_geoms,
            china_geoms,
            province_lines,
            show_bottom_labels=False,
            show_left_labels=False,
        )
        style_map_ticks(ax, show_left)
        ax.contourf(
            grid_lon_2d,
            grid_lat_2d,
            values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=pc,
            antialiased=True,
            zorder=5,
        )
        add_panel_label(ax, label)

    d_mesh = plot_density_panel(
        density_axes[0],
        cnemc_x,
        cnemc_y,
        scatter_max,
        cnemc_norm,
        r"(d) CAMS vs CNEMC PM$_{2.5}$",
        r"CNEMC PM$_{2.5}$ $\mathbf{(\mu g\ m^{-3})}$",
        r"CAMS PM$_{2.5}$ $\mathbf{(\mu g\ m^{-3})}$",
    )
    e_mesh = plot_density_panel(
        density_axes[1],
        chap_x,
        chap_y,
        scatter_max,
        chap_norm,
        r"(e) CAMS vs CHAP PM$_{2.5}$",
        r"CHAP PM$_{2.5}$ $\mathbf{(\mu g\ m^{-3})}$",
        r"CAMS PM$_{2.5}$ $\mathbf{(\mu g\ m^{-3})}$",
    )

    fig.subplots_adjust(left=0.055, right=0.915, top=0.94, bottom=0.075, hspace=0.31, wspace=0.16)
    fig.canvas.draw()

    pos_b_initial = map_axes[1].get_position()
    map_axes[1].set_position(
        [
            pos_b_initial.x0 - 0.044,
            pos_b_initial.y0,
            pos_b_initial.width,
            pos_b_initial.height,
        ]
    )
    fig.canvas.draw()

    rmse_sm = ScalarMappable(norm=rmse_norm, cmap=rmse_cmap)
    rmse_sm.set_array([])
    pos_b = map_axes[1].get_position()
    pos_c = map_axes[2].get_position()
    map_cbar_height = pos_b.height * 0.76
    cax_rmse = fig.add_axes([pos_b.x1 + 0.030, pos_b.y0 + pos_b.height * 0.12, 0.012, map_cbar_height])
    cb_rmse = fig.colorbar(rmse_sm, cax=cax_rmse, orientation="vertical")
    style_colorbar(cb_rmse)

    diff_sm = ScalarMappable(norm=diff_norm, cmap=diff_cmap)
    diff_sm.set_array([])
    cax_diff = fig.add_axes([pos_c.x1 + 0.030, pos_c.y0 + pos_c.height * 0.12, 0.012, map_cbar_height])
    cb_diff = fig.colorbar(diff_sm, cax=cax_diff, orientation="vertical")
    style_colorbar(cb_diff)

    pos_d = density_axes[0].get_position()
    pos_e = density_axes[1].get_position()
    density_cbar_height = pos_d.height * 0.82
    cax_d = fig.add_axes([pos_d.x1 + 0.020, pos_d.y0 + pos_d.height * 0.09, 0.012, density_cbar_height])
    cb_d = fig.colorbar(d_mesh, cax=cax_d, orientation="vertical")
    style_count_colorbar(cb_d)
    cax_e = fig.add_axes([pos_e.x1 + 0.020, pos_e.y0 + pos_e.height * 0.09, 0.012, density_cbar_height])
    cb_e = fig.colorbar(e_mesh, cax=cax_e, orientation="vertical")
    style_count_colorbar(cb_e)

    out_path = args.out_dir / args.output
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_lat, target_lon, rmse_cnemc, rmse_chap, pairs, metadata = compute_rmse_fields(args)
    out_path = plot_rmse_maps(args, target_lat, target_lon, rmse_cnemc, rmse_chap, pairs)

    print(f"CAMS date directories: {metadata['cams_dates']}")
    print(f"Matched CAMS-CNEMC times: {metadata['matched_times']}")
    print(f"Matched CAMS-CHAP times: {metadata['matched_chap_times']}")
    print(f"CNEMC city points: {metadata['cities']}")
    print(f"CAMS target grid: {metadata['lat_count']} x {metadata['lon_count']}")
    print(f"CAMS-CNEMC paired grid samples: {metadata['cnemc_pairs']}")
    print(f"CAMS-CHAP paired grid samples: {metadata['chap_pairs']}")
    print(f"Mean RMSE CAMS vs CNEMC: {float(np.nanmean(rmse_cnemc)):.3f} ug m-3")
    print(f"Mean RMSE CAMS vs CHAP: {float(np.nanmean(rmse_chap)):.3f} ug m-3")
    print(f"Median RMSE CAMS vs CNEMC: {float(np.nanmedian(rmse_cnemc)):.3f} ug m-3")
    print(f"Median RMSE CAMS vs CHAP: {float(np.nanmedian(rmse_chap)):.3f} ug m-3")
    print(f"PDF written: {out_path}")


if __name__ == "__main__":
    main()
