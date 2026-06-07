#!/usr/bin/env python3
"""Plot mean CAMS-minus-CNEMC biases at CNEMC city points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box


PROJECT_ROOT = Path("/home/xeon/lcy-front")
INPUT_ROOT = PROJECT_ROOT / "input"
CNEMC_ROOT = PROJECT_ROOT / "cnemc"
STATION_LIST = Path("/home/xeon/lcy-east/aurora/station-list.csv")
OUT_DIR = PROJECT_ROOT / "plot"

STUDY_START = "20231001"
STUDY_END = "20240412"
DEFAULT_DOMAIN = (105.0, 124.0, 20.0, 43.0)

R_DRY_AIR = 287.05
MAP_RESOLUTION = "50m"
UG_M3_UNIT = r"$\mathbf{(\mu g\ m^{-3})}$"


@dataclass(frozen=True)
class Pollutant:
    key: str
    panel: str
    obs_type: str
    cams_file: str
    cams_var: str
    unit: str
    conversion: str
    min_vlim: float
    pressure_level: int | None = None
    obs_scale: float = 1.0


POLLUTANTS = [
    Pollutant("pm25", "PM2.5", "PM2.5", "sfc", "pm2p5", UG_M3_UNIT, "kg_m3_to_ug_m3", 5.0),
    Pollutant("pm10", "PM10", "PM10", "sfc", "pm10", UG_M3_UNIT, "kg_m3_to_ug_m3", 10.0),
    Pollutant("no2", "NO2 at 1000 hPa", "NO2", "atmos", "no2", UG_M3_UNIT, "mass_fraction_to_ug_m3", 5.0, 1000),
    Pollutant("so2", "SO2 at 1000 hPa", "SO2", "atmos", "so2", UG_M3_UNIT, "mass_fraction_to_ug_m3", 3.0, 1000),
    Pollutant("o3", "O3 at 1000 hPa", "O3", "atmos", "go3", UG_M3_UNIT, "mass_fraction_to_ug_m3", 10.0, 1000),
    Pollutant("co", "CO at 1000 hPa", "CO", "atmos", "co", UG_M3_UNIT, "mass_fraction_to_ug_m3", 100.0, 1000, 1000.0),
]

PANEL_LABELS = {
    "pm25": r"(a) PM$_{2.5}$",
    "pm10": r"(b) PM$_{10}$",
    "no2": r"(c) NO$_2$",
    "so2": r"(d) SO$_2$",
    "o3": r"(e) O$_3$",
    "co": r"(f) CO",
}

AVERAGE_OBS_TYPES = {
    "pm25": "PM2.5_24h",
    "pm10": "PM10_24h",
    "no2": "NO2_24h",
    "so2": "SO2_24h",
    "o3": "O3_8h",
    "co": "CO_24h",
}

CITY_ALIASES = {
    "黔东南州": "黔东南苗族侗族自治州",
    "黔南州": "黔南布依族苗族自治州",
    "黔西南州": "黔西南布依族苗族自治州",
    "博尔塔拉州": "博尔塔拉蒙古自治州",
    "海西州": "海西蒙古族藏族自治州",
    "延边州": "延边朝鲜族自治州",
}


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
    parser.add_argument(
        "--obs-mode",
        choices=("average", "hourly"),
        default="average",
        help="Use averaged CNEMC rows by default; use hourly for raw hourly concentration rows.",
    )
    parser.add_argument("--output", default="S1.pdf")
    parser.add_argument("--city-stats", default=None)
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def finalize_output_names(args: argparse.Namespace) -> None:
    if args.output is None:
        args.output = f"cams_cnemc_{args.obs_mode}_mean_bias_2x3.pdf"


def obs_type_for(pollutant: Pollutant, obs_mode: str) -> str:
    if obs_mode == "average":
        return AVERAGE_OBS_TYPES[pollutant.key]
    return pollutant.obs_type


def cams_date_dirs(input_root: Path, start: str, end: str) -> list[Path]:
    out = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir():
            continue
        token = path.name.replace("-", "")
        if start <= token <= end and (path / "sfc.nc").exists() and (path / "atmos.nc").exists():
            out.append(path)
    return out


def cnemc_city_columns(cnemc_root: Path, start: str, end: str) -> list[str]:
    cities: set[str] = set()
    for year_dir in sorted(cnemc_root.glob("20??")):
        for path in sorted(year_dir.glob("china_cities_*.csv")):
            token = path.stem.rsplit("_", 1)[-1]
            if start <= token <= end:
                cols = pd.read_csv(path, nrows=0).columns.tolist()
                cities.update(str(col).strip() for col in cols[3:] if str(col).strip())
    return sorted(cities)


def load_city_points(station_list: Path, cities: list[str], domain: tuple[float, float, float, float]) -> pd.DataFrame:
    stations = pd.read_csv(station_list).rename(
        columns={
            "监测点编码": "station",
            "监测点名称": "name",
            "城市": "city",
            "经度": "lon",
            "纬度": "lat",
            "对照点": "control",
        }
    )
    stations["city"] = stations["city"].astype(str).str.strip()
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["control"] = stations["control"].astype(str).str.strip()
    stations = stations.dropna(subset=["city", "lon", "lat"])
    stations = stations[stations["lon"].between(70, 140) & stations["lat"].between(15, 55)]

    ordinary = stations[stations["control"].ne("Y")]
    city_mean = ordinary.groupby("city", as_index=False)[["lon", "lat"]].mean()
    fallback = stations.groupby("city", as_index=False)[["lon", "lat"]].mean()
    city_mean = fallback.merge(city_mean, on="city", how="left", suffixes=("_fallback", ""))
    city_mean["lon"] = city_mean["lon"].fillna(city_mean["lon_fallback"])
    city_mean["lat"] = city_mean["lat"].fillna(city_mean["lat_fallback"])
    by_city = city_mean[["city", "lon", "lat"]].set_index("city")

    rows = []
    missing = []
    for city in cities:
        lookup = city if city in by_city.index else CITY_ALIASES.get(city, city)
        if lookup not in by_city.index:
            missing.append(city)
            continue
        rows.append(
            {
                "city": city,
                "lookup_city": lookup,
                "lon": float(by_city.loc[lookup, "lon"]),
                "lat": float(by_city.loc[lookup, "lat"]),
            }
        )
    if missing:
        print("Missing city coordinates:", ", ".join(missing))

    points = pd.DataFrame(rows)
    lon_min, lon_max, lat_min, lat_max = domain
    points = points[points["lon"].between(lon_min, lon_max) & points["lat"].between(lat_min, lat_max)].copy()
    points = points.sort_values("city").reset_index(drop=True)
    return points


def load_cnemc_for_date(cnemc_root: Path, bj_date: str, cities: list[str], obs_types: list[str]) -> pd.DataFrame:
    path = cnemc_root / bj_date[:4] / f"china_cities_{bj_date}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["date", "hour", "type"] + cities
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    df = df[df["type"].isin(obs_types) & df["hour"].isin([8, 20])].copy()
    for city in cities:
        df[city] = pd.to_numeric(df[city], errors="coerce")
    return df


def convert_cams_values(values: np.ndarray, temperature_k: np.ndarray | None, pollutant: Pollutant) -> np.ndarray:
    if pollutant.conversion == "kg_m3_to_ug_m3":
        return values * 1.0e9
    if temperature_k is None:
        raise ValueError(f"{pollutant.key} requires temperature for mass-fraction conversion")
    if pollutant.pressure_level is None:
        raise ValueError(f"{pollutant.key} requires a pressure level for mass-fraction conversion")
    air_density = (pollutant.pressure_level * 100.0) / (R_DRY_AIR * temperature_k)
    mass_conc = values * air_density
    if pollutant.conversion == "mass_fraction_to_ug_m3":
        return mass_conc * 1.0e9
    if pollutant.conversion == "mass_fraction_to_mg_m3":
        return mass_conc * 1.0e6
    raise ValueError(f"Unknown conversion: {pollutant.conversion}")


def interpolate_cams_for_date(date_dir: Path, points: pd.DataFrame) -> dict[str, np.ndarray]:
    lat_points = xr.DataArray(points["lat"].to_numpy(), dims="city")
    lon_points = xr.DataArray(points["lon"].to_numpy(), dims="city")
    out: dict[str, np.ndarray] = {}

    with xr.open_dataset(date_dir / "sfc.nc") as sfc:
        times = pd.to_datetime(sfc["forecast_reference_time"].values).to_pydatetime()
        out["times"] = np.asarray(times, dtype="datetime64[ns]")
        for pollutant in POLLUTANTS:
            if pollutant.cams_file != "sfc":
                continue
            values = (
                sfc[pollutant.cams_var]
                .isel(forecast_period=0)
                .interp(latitude=lat_points, longitude=lon_points)
                .to_numpy()
            )
            out[pollutant.key] = convert_cams_values(values, None, pollutant)

    with xr.open_dataset(date_dir / "atmos.nc") as atmos:
        for pollutant in POLLUTANTS:
            if pollutant.cams_file != "atmos":
                continue
            if pollutant.pressure_level is None:
                raise ValueError(f"{pollutant.key} needs a pressure level")
            temperature = (
                atmos["t"]
                .isel(forecast_period=0)
                .sel(pressure_level=pollutant.pressure_level)
                .interp(latitude=lat_points, longitude=lon_points)
                .to_numpy()
            )
            values = (
                atmos[pollutant.cams_var]
                .isel(forecast_period=0)
                .sel(pressure_level=pollutant.pressure_level)
                .interp(latitude=lat_points, longitude=lon_points)
                .to_numpy()
            )
            out[pollutant.key] = convert_cams_values(values, temperature, pollutant)
    return out


def build_city_statistics(args: argparse.Namespace, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_dirs = cams_date_dirs(args.input_root, args.start, args.end)
    cities = points["city"].tolist()
    obs_types = [obs_type_for(pollutant, args.obs_mode) for pollutant in POLLUTANTS]

    records = []
    cnemc_cache: dict[str, pd.DataFrame] = {}
    for date_dir in date_dirs:
        cams = interpolate_cams_for_date(date_dir, points)
        for time_index, utc_time64 in enumerate(cams["times"]):
            utc_time = pd.Timestamp(utc_time64).to_pydatetime()
            bj_time = pd.Timestamp(utc_time) + pd.Timedelta(hours=8)
            bj_date = bj_time.strftime("%Y%m%d")
            bj_hour = int(bj_time.hour)
            if bj_date not in cnemc_cache:
                cnemc_cache[bj_date] = load_cnemc_for_date(args.cnemc_root, bj_date, cities, obs_types)
            obs_df = cnemc_cache[bj_date]
            for pollutant in POLLUTANTS:
                obs_type = obs_type_for(pollutant, args.obs_mode)
                row = obs_df[(obs_df["hour"] == bj_hour) & (obs_df["type"] == obs_type)]
                if row.empty:
                    continue
                obs_values = row.iloc[0][cities].to_numpy(dtype=float) * pollutant.obs_scale
                cams_values = np.asarray(cams[pollutant.key][time_index, :], dtype=float)
                bias = cams_values - obs_values
                valid = np.isfinite(cams_values) & np.isfinite(obs_values)
                for idx in np.where(valid)[0]:
                    records.append(
                        {
                            "pollutant": pollutant.key,
                            "panel": pollutant.panel,
                            "obs_mode": args.obs_mode,
                            "obs_type": obs_type,
                            "unit": pollutant.unit,
                            "city": cities[idx],
                            "lon": float(points.loc[idx, "lon"]),
                            "lat": float(points.loc[idx, "lat"]),
                            "utc_time": pd.Timestamp(utc_time).strftime("%Y-%m-%d %H:%M:%S"),
                            "bj_date": bj_date,
                            "bj_hour": bj_hour,
                            "cams": float(cams_values[idx]),
                            "cnemc": float(obs_values[idx]),
                            "bias": float(bias[idx]),
                        }
                    )
    paired = pd.DataFrame(records)
    if paired.empty:
        raise RuntimeError("No paired CAMS-CNEMC samples were found")

    city_stats = (
        paired.groupby(["pollutant", "panel", "obs_mode", "obs_type", "unit", "city", "lon", "lat"], as_index=False)
        .agg(
            n=("bias", "count"),
            cams_mean=("cams", "mean"),
            cnemc_mean=("cnemc", "mean"),
            bias_mean=("bias", "mean"),
            bias_median=("bias", "median"),
        )
        .sort_values(["pollutant", "city"])
    )
    summary = (
        city_stats.groupby(["pollutant", "panel", "obs_mode", "obs_type", "unit"], as_index=False)
        .agg(
            cities=("city", "count"),
            mean_bias=("bias_mean", "mean"),
            median_bias=("bias_mean", "median"),
            negative_fraction=("bias_mean", lambda x: float(np.mean(np.asarray(x) < 0))),
            p05=("bias_mean", lambda x: float(np.nanpercentile(x, 5))),
            p95=("bias_mean", lambda x: float(np.nanpercentile(x, 95))),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.city_stats:
        city_stats.to_csv(args.out_dir / args.city_stats, index=False)
    if args.summary:
        summary.to_csv(args.out_dir / args.summary, index=False)
    return city_stats, summary


def load_country_geoms(map_extent: tuple[float, float, float, float]) -> list:
    shp = shapereader.natural_earth(MAP_RESOLUTION, "cultural", "admin_0_countries")
    area = box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
    return [record.geometry for record in shapereader.Reader(shp).records() if record.geometry.intersects(area)]


def load_china_geoms() -> list:
    shp = shapereader.natural_earth(MAP_RESOLUTION, "cultural", "admin_0_countries")
    geoms = []
    for record in shapereader.Reader(shp).records():
        attrs = record.attributes
        if attrs.get("ADM0_A3") == "CHN" or attrs.get("ADMIN") == "China" or attrs.get("NAME") == "China":
            geoms.append(record.geometry)
    return geoms


def load_china_province_lines(map_extent: tuple[float, float, float, float]) -> list:
    shp = shapereader.natural_earth(MAP_RESOLUTION, "cultural", "admin_1_states_provinces_lines")
    area = box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
    lines = []
    for record in shapereader.Reader(shp).records():
        attrs = record.attributes
        if attrs.get("ADM0_A3") != "CHN" and attrs.get("ADM0_NAME") != "China":
            continue
        geom = record.geometry
        if geom is not None and geom.intersects(area):
            lines.append(geom)
    return lines


def add_base_map(
    ax,
    map_extent: tuple[float, float, float, float],
    country_geoms: list,
    china_geoms: list,
    province_lines: list,
    show_bottom_labels: bool,
    show_left_labels: bool,
) -> None:
    pc = ccrs.PlateCarree()
    ax.set_extent(map_extent, crs=pc)
    ax.add_feature(cfeature.OCEAN.with_scale(MAP_RESOLUTION), facecolor="#dcecf5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(MAP_RESOLUTION), facecolor="#f7f4ee", zorder=0)
    ax.add_geometries(country_geoms, pc, facecolor="none", edgecolor="#9c9c9c", linewidth=0.45, zorder=1)
    ax.add_geometries(china_geoms, pc, facecolor="none", edgecolor="#333333", linewidth=1.05, zorder=2)
    ax.add_geometries(province_lines, pc, facecolor="none", edgecolor="#666666", linewidth=0.42, zorder=3)
    ax.coastlines(MAP_RESOLUTION, linewidth=0.6, color="#555555", zorder=4)
    gl = ax.gridlines(crs=pc, draw_labels=True, linewidth=0.35, color="#9e9e9e", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_bottom_labels
    gl.left_labels = show_left_labels
    gl.xlabel_style = {"size": 14, "weight": "bold"}
    gl.ylabel_style = {"size": 14, "weight": "bold"}


def robust_vlim(values: np.ndarray, minimum: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return minimum
    return float(max(np.nanpercentile(finite, 95), minimum))


def plot_bias_maps(args: argparse.Namespace, city_stats: pd.DataFrame) -> Path:
    domain = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    map_extent = (args.lon_min - 1.5, args.lon_max + 1.5, args.lat_min - 1.0, args.lat_max + 1.0)

    country_geoms = load_country_geoms(map_extent)
    china_geoms = load_china_geoms()
    province_lines = load_china_province_lines(map_extent)

    pc = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.35), subplot_kw={"projection": pc})
    axes = axes.ravel()
    cmap = plt.get_cmap("RdBu_r")

    for index, (ax, pollutant) in enumerate(zip(axes, POLLUTANTS, strict=True)):
        data = city_stats[city_stats["pollutant"] == pollutant.key].copy()
        add_base_map(
            ax,
            map_extent,
            country_geoms,
            china_geoms,
            province_lines,
            show_bottom_labels=index >= 3,
            show_left_labels=index % 3 == 0,
        )
        vlim = robust_vlim(data["bias_mean"].to_numpy(), pollutant.min_vlim)
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        sc = ax.scatter(
            data["lon"],
            data["lat"],
            c=data["bias_mean"],
            s=50,
            marker="o",
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.18,
            alpha=0.95,
            transform=pc,
            zorder=8,
        )
        ax.text(
            0.025,
            0.965,
            PANEL_LABELS[pollutant.key],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
            zorder=12,
        )
        cb = fig.colorbar(
            sc,
            ax=ax,
            orientation="vertical",
            pad=0.045,
            shrink=0.84,
            fraction=0.058,
            aspect=15,
        )
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=14, pad=3, length=4.5, width=1.2)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
        cb.ax.text(
            -0.2,
            1.035,
            pollutant.unit,
            transform=cb.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14.5,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.045, right=0.987, top=0.985, bottom=0.06, hspace=0.075, wspace=0.065)
    out_path = args.out_dir / args.output
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    finalize_output_names(args)
    domain = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    cities = cnemc_city_columns(args.cnemc_root, args.start, args.end)
    points = load_city_points(args.station_list, cities, domain)
    city_stats, summary = build_city_statistics(args, points)
    out_path = plot_bias_maps(args, city_stats)

    print(f"Selected city points: {len(points)}")
    print(f"CAMS date directories: {len(cams_date_dirs(args.input_root, args.start, args.end))}")
    if args.city_stats:
        print(f"City mean stats: {args.out_dir / args.city_stats}")
    if args.summary:
        print(f"Summary stats: {args.out_dir / args.summary}")
    print(summary.to_string(index=False))
    print(f"PDF written: {out_path}")


if __name__ == "__main__":
    main()
