#!/usr/bin/env python3
"""Figure 2: period-mean PM2.5 and PM10 spatial distributions."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from shapely import contains_xy, prepare
from shapely.geometry import box
from shapely.ops import unary_union


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_WRFOUT_DIR = Path("/home/xeon/wrf/cases/eastchina_20240206_20240212_20km/output/wrfchem_latest")
DEFAULT_AURORA_ROOT = PROJECT_DIR / "aurora" / "2024"
DEFAULT_STATION_LIST = PROJECT_DIR / "aurora" / "station-list.csv"
DEFAULT_CNAQ_DIR = PROJECT_DIR / "aurora" / "station_20240101-20241231"
DEFAULT_OUTPUT = SCRIPT_DIR / "F2.pdf"

P0 = 100000.0
RCP = 0.2854


@dataclass(frozen=True)
class Species:
    key: str
    label: str
    wrf_name: str
    aurora_name: str
    obs_type: str
    cmap: str


SPECIES = [
    Species("pm25", "PM$_{2.5}$", "PM2_5_DRY", "pm2p5", "PM2.5", "YlOrRd"),
    Species("pm10", "PM$_{10}$", "PM10", "pm10", "PM10", "YlOrBr"),
]

SOURCE_ORDER = [
    ("CNEMC", "cnemc"),
    ("WRF-Chem", "wrf"),
    ("Aurora", "aurora"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrfout-dir", type=Path, default=DEFAULT_WRFOUT_DIR)
    parser.add_argument("--aurora-root", type=Path, default=DEFAULT_AURORA_ROOT)
    parser.add_argument("--station-list", type=Path, default=DEFAULT_STATION_LIST)
    parser.add_argument("--cnaq-dir", type=Path, default=DEFAULT_CNAQ_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2024-02-06_00:00:00")
    parser.add_argument("--end", default="2024-02-12_23:00:00")
    parser.add_argument("--hours", nargs="+", type=int, default=[0, 12])
    parser.add_argument("--wrf-pressure-hpa", type=float, default=1000.0)
    parser.add_argument("--idw-neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--contour-levels", type=int, default=14)
    parser.add_argument("--limit-percentiles", nargs=2, type=float, default=(2.0, 98.0))
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "font.weight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def parse_time(value: str) -> datetime:
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported time format: {value}")


def parse_wrf_time(path: Path) -> datetime:
    return datetime.strptime(path.name.replace("wrfout_d01_", ""), "%Y-%m-%d_%H:%M:%S")


def day_range(start: datetime, end: datetime) -> list[datetime]:
    out = []
    day = datetime(start.year, start.month, start.day)
    last = datetime(end.year, end.month, end.day)
    while day <= last:
        out.append(day)
        day += timedelta(days=1)
    return out


def aligned_times(start: datetime, end: datetime, hours: list[int]) -> list[datetime]:
    times = []
    for day in day_range(start, end):
        for hour in sorted(hours):
            time = day + timedelta(hours=hour)
            if start <= time <= end:
                times.append(time)
    return times


def as_array(values) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)


def read_wrf_grid(first_wrfout: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    with Dataset(first_wrfout) as ds:
        lats = as_array(ds.variables["XLAT"][0])
        lons = as_array(ds.variables["XLONG"][0])
    extent = [
        float(np.nanmin(lons)) - 0.15,
        float(np.nanmax(lons)) + 0.15,
        float(np.nanmin(lats)) - 0.15,
        float(np.nanmax(lats)) + 0.15,
    ]
    return lons, lats, extent


def wrf_boundary_path(wrf_lons: np.ndarray, wrf_lats: np.ndarray) -> MplPath:
    boundary = np.concatenate(
        [
            np.column_stack([wrf_lons[0, :], wrf_lats[0, :]]),
            np.column_stack([wrf_lons[1:, -1], wrf_lats[1:, -1]]),
            np.column_stack([wrf_lons[-1, -2::-1], wrf_lats[-1, -2::-1]]),
            np.column_stack([wrf_lons[-2:0:-1, 0], wrf_lats[-2:0:-1, 0]]),
        ]
    )
    return MplPath(boundary)


def load_domain_stations(station_list: Path, wrf_lons: np.ndarray, wrf_lats: np.ndarray) -> pd.DataFrame:
    stations = pd.read_csv(station_list)
    stations.columns = [str(col).strip() for col in stations.columns]
    stations = stations.rename(
        columns={
            "监测点编码": "station",
            "监测点名称": "name",
            "城市": "city",
            "经度": "lon",
            "纬度": "lat",
            "对照点": "control",
        }
    )
    stations["station"] = stations["station"].astype(str)
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations = stations.dropna(subset=["station", "lon", "lat"]).copy()
    inside = wrf_boundary_path(wrf_lons, wrf_lats).contains_points(stations[["lon", "lat"]].to_numpy())
    return stations.loc[inside].reset_index(drop=True)


def load_land_mask(wrf_lons: np.ndarray, wrf_lats: np.ndarray, extent: list[float]) -> np.ndarray:
    land_path = shapereader.natural_earth("10m", "physical", "land")
    area = box(extent[0], extent[2], extent[1], extent[3])
    geoms = [geom for geom in shapereader.Reader(land_path).geometries() if geom.intersects(area)]
    land = unary_union(geoms)
    prepare(land)
    return np.asarray(contains_xy(land, wrf_lons, wrf_lats), dtype=bool)


def load_province_boundaries(extent: list[float]) -> list:
    boundary_path = shapereader.natural_earth("10m", "cultural", "admin_1_states_provinces_lines")
    area = box(extent[0], extent[2], extent[1], extent[3])
    geoms = []
    for record in shapereader.Reader(boundary_path).records():
        attrs = record.attributes
        if attrs.get("ADM0_A3") != "CHN" and attrs.get("ADM0_NAME") != "China":
            continue
        geom = record.geometry
        if geom is not None and geom.intersects(area):
            geoms.append(geom)
    return geoms


def vertical_interp_weights(pressure: np.ndarray, target_pa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nlev = pressure.shape[0]
    n_above = np.sum(pressure >= target_pa, axis=0)
    k1 = np.clip(n_above - 1, 0, nlev - 1)
    k2 = np.clip(n_above, 0, nlev - 1)
    p1 = np.take_along_axis(pressure, k1[None, :, :], axis=0)[0]
    p2 = np.take_along_axis(pressure, k2[None, :, :], axis=0)[0]
    denom = np.log(p2) - np.log(p1)
    frac = np.divide(
        np.log(target_pa) - np.log(p1),
        denom,
        out=np.zeros_like(p1, dtype=np.float64),
        where=np.abs(denom) > 0.0,
    )
    return k1, k2, frac, np.isfinite(pressure[0])


def vertical_interp(values: np.ndarray, k1: np.ndarray, k2: np.ndarray, frac: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v1 = np.take_along_axis(values, k1[None, :, :], axis=0)[0]
    v2 = np.take_along_axis(values, k2[None, :, :], axis=0)[0]
    out = v1 + frac * (v2 - v1)
    return np.where(valid, out, np.nan)


def wrf_pm_at_1000hpa(path: Path, target_pa: float) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        pressure = as_array(ds.variables["P"][0]) + as_array(ds.variables["PB"][0])
        k1, k2, frac, valid = vertical_interp_weights(pressure, target_pa)
        return {
            spec.key: vertical_interp(as_array(ds.variables[spec.wrf_name][0]), k1, k2, frac, valid)
            for spec in SPECIES
        }


def mean_wrf_fields(wrfout_dir: Path, times: list[datetime], target_pa: float) -> dict[str, np.ndarray]:
    samples = {spec.key: [] for spec in SPECIES}
    for time in times:
        path = wrfout_dir / f"wrfout_d01_{time:%Y-%m-%d_%H:%M:%S}"
        if not path.exists():
            raise FileNotFoundError(path)
        fields = wrf_pm_at_1000hpa(path, target_pa)
        for spec in SPECIES:
            samples[spec.key].append(fields[spec.key])
    return {key: np.nanmean(np.stack(values), axis=0) for key, values in samples.items()}


def aurora_pm_to_ug_m3(field: xr.DataArray) -> xr.DataArray:
    return field * 1.0e9


def interp_aurora_to_grid(field: xr.DataArray, wrf_lats: np.ndarray, wrf_lons: np.ndarray) -> np.ndarray:
    field = field.sortby("latitude")
    target_lat = xr.DataArray(wrf_lats, dims=("south_north", "west_east"))
    target_lon = xr.DataArray(np.mod(wrf_lons, 360.0), dims=("south_north", "west_east"))
    return np.asarray(field.interp(latitude=target_lat, longitude=target_lon, method="linear"), dtype=np.float64)


def mean_aurora_fields(aurora_root: Path, times: list[datetime], wrf_lats: np.ndarray, wrf_lons: np.ndarray) -> dict[str, np.ndarray]:
    samples = {spec.key: [] for spec in SPECIES}
    by_day: dict[datetime, list[datetime]] = {}
    for time in times:
        by_day.setdefault(datetime(time.year, time.month, time.day), []).append(time)
    for day, day_times in by_day.items():
        path = aurora_root / "sfc" / f"{day:%Y%m%d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            for time in day_times:
                wanted = np.datetime64(f"{time:%Y-%m-%dT%H:%M:%S}")
                if wanted not in ds["time"].values:
                    raise ValueError(f"Aurora time missing: {wanted} in {path}")
                idx = int(np.where(ds["time"].values == wanted)[0][0])
                for spec in SPECIES:
                    field = aurora_pm_to_ug_m3(ds[spec.aurora_name].isel(time=idx))
                    samples[spec.key].append(interp_aurora_to_grid(field, wrf_lats, wrf_lons))
    return {key: np.nanmean(np.stack(values), axis=0) for key, values in samples.items()}


def clean_obs_values(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce")
    values = values.where(values >= 0.0)
    return values.where(values < 10000.0)


def load_cnemc_station_means(
    cnaq_dir: Path,
    stations: pd.DataFrame,
    times_utc: list[datetime],
) -> dict[str, pd.Series]:
    station_ids = stations["station"].astype(str).tolist()
    wanted_index = pd.DatetimeIndex(times_utc)
    local_times = [time + timedelta(hours=8) for time in times_utc]
    days = sorted({datetime(time.year, time.month, time.day) for time in local_times})
    wanted_types = {spec.obs_type for spec in SPECIES}
    parts = {spec.key: [] for spec in SPECIES}

    for day in days:
        path = cnaq_dir / f"china_sites_{day:%Y%m%d}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        present_ids = [sid for sid in station_ids if sid in header]
        df = pd.read_csv(
            path,
            usecols=["date", "hour", "type", *present_ids],
            na_values=["", "NA", "NaN", "nan", "None", "-", "--", "—"],
            low_memory=False,
        )
        df = df.copy()
        local_time = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["hour"].astype(int), unit="h")
        df["time_utc"] = local_time - pd.Timedelta(hours=8)
        df = df[pd.DatetimeIndex(df["time_utc"]).isin(wanted_index) & df["type"].astype(str).isin(wanted_types)]
        for spec in SPECIES:
            sub = df[df["type"].astype(str) == spec.obs_type]
            if sub.empty:
                continue
            values = clean_obs_values(sub[present_ids])
            values.index = pd.to_datetime(sub["time_utc"].to_numpy())
            values = values.reindex(columns=station_ids)
            parts[spec.key].append(values)

    means = {}
    for spec in SPECIES:
        if not parts[spec.key]:
            means[spec.key] = pd.Series(index=station_ids, dtype=float)
            continue
        obs = pd.concat(parts[spec.key]).sort_index()
        obs = obs.groupby(obs.index).mean().reindex(wanted_index)
        means[spec.key] = obs.mean(axis=0, skipna=True)
    return means


def idw_to_grid(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    values: np.ndarray,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    neighbors: int,
    power: float,
) -> np.ndarray:
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.full_like(grid_lons, np.nan, dtype=float)
    points = np.column_stack([station_lons[mask], station_lats[mask]])
    vals = values[mask]
    k = min(max(1, neighbors), len(vals))
    tree = cKDTree(points)
    targets = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])
    dist, idx = tree.query(targets, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    exact = dist[:, 0] < 1.0e-12
    weights = 1.0 / np.maximum(dist, 1.0e-12) ** power
    interp = np.sum(weights * vals[idx], axis=1) / np.sum(weights, axis=1)
    interp[exact] = vals[idx[exact, 0]]
    return interp.reshape(grid_lons.shape)


def mean_cnemc_fields(
    cnaq_dir: Path,
    stations: pd.DataFrame,
    times: list[datetime],
    wrf_lons: np.ndarray,
    wrf_lats: np.ndarray,
    land_mask: np.ndarray,
    neighbors: int,
    power: float,
) -> dict[str, np.ndarray]:
    station_means = load_cnemc_station_means(cnaq_dir, stations, times)
    fields = {}
    for spec in SPECIES:
        field = idw_to_grid(
            stations["lon"].to_numpy(),
            stations["lat"].to_numpy(),
            station_means[spec.key].to_numpy(dtype=float),
            wrf_lons,
            wrf_lats,
            neighbors,
            power,
        )
        fields[spec.key] = np.where(land_mask, field, np.nan)
    return fields


def contour_limits(fields: list[np.ndarray], low_pct: float, high_pct: float) -> tuple[float, float]:
    data = np.concatenate([np.asarray(field, dtype=float).ravel() for field in fields])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(np.nanpercentile(data, low_pct)))
    hi = float(np.nanpercentile(data, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(data))
        lo = 0.0
    lo = np.floor(lo / 10.0) * 10.0
    hi = np.ceil(hi / 10.0) * 10.0
    if hi <= lo:
        hi = lo + 10.0
    return lo, hi


def add_map_features(ax, province_geoms: list) -> None:
    ax.coastlines(resolution="10m", linewidth=1.0, color="0.10", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.75, edgecolor="0.15", zorder=4)
    if province_geoms:
        ax.add_geometries(
            province_geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="0.22",
            linewidth=0.75,
            zorder=4.2,
        )


def add_ticks(ax, extent: list[float], show_x: bool, show_y: bool) -> None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(116, 126, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(28, 36, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", which="major", width=1.3, length=5.2, direction="out", labelsize=12)
    ax.tick_params(labelbottom=show_x, labelleft=show_y)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(116, 126, 2),
        ylocs=np.arange(28, 36, 2),
        linewidth=0.55,
        color="0.58",
        alpha=0.35,
        linestyle="--",
        zorder=2,
    )


def plot_figure(
    output: Path,
    fields: dict[str, dict[str, np.ndarray]],
    wrf_lons: np.ndarray,
    wrf_lats: np.ndarray,
    extent: list[float],
    province_geoms: list,
    contour_count: int,
    percentiles: tuple[float, float],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), subplot_kw={"projection": ccrs.PlateCarree()})
    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    for row_i, spec in enumerate(SPECIES):
        vmin, vmax = contour_limits(
            [fields[source_key][spec.key] for _, source_key in SOURCE_ORDER],
            percentiles[0],
            percentiles[1],
        )
        levels = np.linspace(vmin, vmax, contour_count)
        ticks = np.linspace(vmin, vmax, 5)
        for col_i, (source_label, source_key) in enumerate(SOURCE_ORDER):
            ax = axes[row_i, col_i]
            data = np.ma.masked_invalid(fields[source_key][spec.key])
            contour = ax.contourf(
                wrf_lons,
                wrf_lats,
                data,
                levels=levels,
                cmap=spec.cmap,
                extend="both",
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            add_map_features(ax, province_geoms)
            add_ticks(ax, extent, show_x=True, show_y=True)
            ax.set_title(f"{panel_labels[row_i][col_i]} {spec.label} | {source_label}", loc="left", pad=6)
            cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.070, pad=0.105, aspect=28)
            cbar.ax.tick_params(labelsize=11, width=1.1, length=3.2)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks])
            for label in cbar.ax.get_xticklabels():
                label.set_fontweight("bold")
            cbar.outline.set_linewidth(1.0)
            cbar.set_label("($\\mu$g m$^{-3}$)", fontsize=12, fontweight="bold", labelpad=2)

    fig.subplots_adjust(left=0.055, right=0.988, top=0.965, bottom=0.075, wspace=0.18, hspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start = parse_time(args.start)
    end = parse_time(args.end)
    times = aligned_times(start, end, args.hours)
    if not times:
        raise ValueError("No aligned times selected.")

    first_wrf = args.wrfout_dir / f"wrfout_d01_{times[0]:%Y-%m-%d_%H:%M:%S}"
    wrf_lons, wrf_lats, extent = read_wrf_grid(first_wrf)
    stations = load_domain_stations(args.station_list, wrf_lons, wrf_lats)
    land_mask = load_land_mask(wrf_lons, wrf_lats, extent)
    province_geoms = load_province_boundaries(extent)

    print(f"Using {len(times)} UTC samples: {times[0]} to {times[-1]}")
    print(f"Selected {len(stations)} CNEMC stations inside WRF d01")
    print("Computing WRF-Chem period mean")
    wrf_fields = mean_wrf_fields(args.wrfout_dir, times, args.wrf_pressure_hpa * 100.0)
    print("Computing Aurora period mean")
    aurora_fields = mean_aurora_fields(args.aurora_root, times, wrf_lats, wrf_lons)
    print("Computing CNEMC period mean")
    cnemc_fields = mean_cnemc_fields(
        args.cnaq_dir,
        stations,
        times,
        wrf_lons,
        wrf_lats,
        land_mask,
        args.idw_neighbors,
        args.idw_power,
    )
    plot_figure(
        args.output,
        {"cnemc": cnemc_fields, "wrf": wrf_fields, "aurora": aurora_fields},
        wrf_lons,
        wrf_lats,
        extent,
        province_geoms,
        args.contour_levels,
        tuple(args.limit_percentiles),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
