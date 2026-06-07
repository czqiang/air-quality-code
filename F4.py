#!/usr/bin/env python3
"""Figure 4: model-observation scatter density plots."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_WRFOUT_DIR = Path("/home/xeon/wrf/cases/eastchina_20240206_20240212_20km/output/wrfchem_latest")
DEFAULT_AURORA_ROOT = PROJECT_DIR / "aurora" / "2024"
DEFAULT_STATION_LIST = PROJECT_DIR / "aurora" / "station-list.csv"
DEFAULT_CNAQ_DIR = PROJECT_DIR / "aurora" / "station_20240101-20241231"
DEFAULT_OUTPUT = SCRIPT_DIR / "F4.pdf"

R_UNIVERSAL = 8.314462618
R_DRY_AIR = 287.05
P0 = 100000.0
RCP = 0.2854


@dataclass(frozen=True)
class Species:
    key: str
    label: str
    wrf_name: str
    aurora_name: str
    obs_type: str
    kind: str
    molecular_weight_g_mol: float | None = None


SPECIES = [
    Species("pm25", "PM$_{2.5}$", "PM2_5_DRY", "pm2p5", "PM2.5", "pm"),
    Species("pm10", "PM$_{10}$", "PM10", "pm10", "PM10", "pm"),
    Species("o3", "O$_3$", "o3", "go3", "O3", "gas", 47.9982),
    Species("no2", "NO$_2$", "no2", "no2", "NO2", "gas", 46.0055),
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
    parser.add_argument("--aurora-pressure-hpa", type=int, default=1000)
    parser.add_argument("--gridsize", type=int, default=42)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "font.weight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "axes.titlesize": 23,
            "axes.titleweight": "bold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
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


def read_wrf_grid(first_wrfout: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(first_wrfout) as ds:
        lats = as_array(ds.variables["XLAT"][0])
        lons = as_array(ds.variables["XLONG"][0])
    return lons, lats


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


def wrf_pressure_temperature(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    pressure = as_array(ds.variables["P"][0]) + as_array(ds.variables["PB"][0])
    theta = as_array(ds.variables["T"][0]) + 300.0
    temperature = theta * np.power(pressure / P0, RCP)
    return pressure, temperature


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


def ppmv_to_ug_m3(ppmv: np.ndarray, molecular_weight_g_mol: float, pressure_pa: float, temperature_k: np.ndarray) -> np.ndarray:
    return ppmv * molecular_weight_g_mol * pressure_pa / (R_UNIVERSAL * temperature_k)


def grid_to_points(grid_lons: np.ndarray, grid_lats: np.ndarray, field: np.ndarray, point_lons: np.ndarray, point_lats: np.ndarray) -> np.ndarray:
    finite = np.isfinite(field)
    points = np.column_stack([grid_lons[finite], grid_lats[finite]])
    values = field[finite]
    targets = np.column_stack([point_lons, point_lats])
    if len(values) < 3:
        return np.full(len(point_lons), np.nan)
    linear = LinearNDInterpolator(points, values, fill_value=np.nan)(targets)
    if np.isnan(linear).any():
        linear[np.isnan(linear)] = NearestNDInterpolator(points, values)(targets[np.isnan(linear)])
    return np.asarray(linear, dtype=np.float64)


def wrf_fields_at_1000hpa(path: Path, target_pa: float) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        pressure, temperature_profile = wrf_pressure_temperature(ds)
        k1, k2, frac, valid = vertical_interp_weights(pressure, target_pa)
        temperature = vertical_interp(temperature_profile, k1, k2, frac, valid)
        fields = {}
        for spec in SPECIES:
            raw = vertical_interp(as_array(ds.variables[spec.wrf_name][0]), k1, k2, frac, valid)
            if spec.kind == "pm":
                fields[spec.key] = raw
            else:
                fields[spec.key] = ppmv_to_ug_m3(raw, spec.molecular_weight_g_mol, target_pa, temperature)
    return fields


def load_wrf_station_values(
    wrfout_dir: Path,
    times: list[datetime],
    wrf_lons: np.ndarray,
    wrf_lats: np.ndarray,
    stations: pd.DataFrame,
    target_pa: float,
) -> dict[str, pd.DataFrame]:
    station_ids = stations["station"].astype(str).tolist()
    rows = {spec.key: [] for spec in SPECIES}
    point_lons = stations["lon"].to_numpy()
    point_lats = stations["lat"].to_numpy()
    for time in times:
        path = wrfout_dir / f"wrfout_d01_{time:%Y-%m-%d_%H:%M:%S}"
        if not path.exists():
            raise FileNotFoundError(path)
        fields = wrf_fields_at_1000hpa(path, target_pa)
        for spec in SPECIES:
            rows[spec.key].append(grid_to_points(wrf_lons, wrf_lats, fields[spec.key], point_lons, point_lats))
    return {
        spec.key: pd.DataFrame(rows[spec.key], index=pd.DatetimeIndex(times), columns=station_ids)
        for spec in SPECIES
    }


def aurora_pm_to_ug_m3(field: xr.DataArray) -> xr.DataArray:
    return field * 1.0e9


def aurora_gas_to_ug_m3(gas_kg_kg: xr.DataArray, temp_k: xr.DataArray, pressure_hpa: int) -> xr.DataArray:
    pressure_pa = float(pressure_hpa) * 100.0
    rho_air = pressure_pa / (R_DRY_AIR * temp_k)
    return gas_kg_kg * rho_air * 1.0e9


def interp_aurora_to_stations(field: xr.DataArray, stations: pd.DataFrame) -> np.ndarray:
    field = field.sortby("latitude")
    target_lat = xr.DataArray(stations["lat"].to_numpy(), dims=("station",))
    target_lon = xr.DataArray(np.mod(stations["lon"].to_numpy(), 360.0), dims=("station",))
    return np.asarray(field.interp(latitude=target_lat, longitude=target_lon, method="linear"), dtype=np.float64)


def load_aurora_station_values(
    aurora_root: Path,
    times: list[datetime],
    stations: pd.DataFrame,
    pressure_hpa: int,
) -> dict[str, pd.DataFrame]:
    station_ids = stations["station"].astype(str).tolist()
    rows = {spec.key: [] for spec in SPECIES}
    by_day: dict[datetime, list[datetime]] = {}
    for time in times:
        by_day.setdefault(datetime(time.year, time.month, time.day), []).append(time)

    for day, day_times in by_day.items():
        sfc_path = aurora_root / "sfc" / f"{day:%Y%m%d}.nc"
        lev_path = aurora_root / "lev" / f"{day:%Y%m%d}.nc"
        if not sfc_path.exists() or not lev_path.exists():
            raise FileNotFoundError(f"Missing Aurora files: {sfc_path} / {lev_path}")
        with xr.open_dataset(sfc_path) as sfc, xr.open_dataset(lev_path) as lev:
            levp = lev.sel(pressure_level=pressure_hpa)
            for time in day_times:
                wanted = np.datetime64(f"{time:%Y-%m-%dT%H:%M:%S}")
                if wanted not in sfc["time"].values:
                    raise ValueError(f"Aurora time missing: {wanted}")
                sfc_i = int(np.where(sfc["time"].values == wanted)[0][0])
                lev_i = int(np.where(levp["time"].values == wanted)[0][0])
                temp = levp["t"].isel(time=lev_i)
                for spec in SPECIES:
                    if spec.kind == "pm":
                        field = aurora_pm_to_ug_m3(sfc[spec.aurora_name].isel(time=sfc_i))
                    else:
                        field = aurora_gas_to_ug_m3(levp[spec.aurora_name].isel(time=lev_i), temp, pressure_hpa)
                    rows[spec.key].append(interp_aurora_to_stations(field, stations))
    return {
        spec.key: pd.DataFrame(rows[spec.key], index=pd.DatetimeIndex(times), columns=station_ids)
        for spec in SPECIES
    }


def clean_obs_values(frame: pd.DataFrame, spec: Species) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce")
    values = values.where(values >= 0.0)
    if spec.key == "co":
        return values.where(values < 100.0) * 1000.0
    return values.where(values < 10000.0)


def load_cnemc_observations(cnaq_dir: Path, stations: pd.DataFrame, times_utc: list[datetime]) -> dict[str, pd.DataFrame]:
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
        ).copy()
        local_time = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["hour"].astype(int), unit="h")
        df["time_utc"] = local_time - pd.Timedelta(hours=8)
        df = df[pd.DatetimeIndex(df["time_utc"]).isin(wanted_index) & df["type"].astype(str).isin(wanted_types)]
        for spec in SPECIES:
            sub = df[df["type"].astype(str) == spec.obs_type]
            if sub.empty:
                continue
            values = clean_obs_values(sub[present_ids], spec)
            values.index = pd.to_datetime(sub["time_utc"].to_numpy())
            values = values.reindex(columns=station_ids)
            parts[spec.key].append(values)

    out = {}
    for spec in SPECIES:
        if not parts[spec.key]:
            out[spec.key] = pd.DataFrame(index=wanted_index, columns=station_ids, dtype=float)
            continue
        obs = pd.concat(parts[spec.key]).sort_index()
        obs = obs.groupby(obs.index).mean().reindex(wanted_index)
        out[spec.key] = obs.reindex(columns=station_ids)
    return out


def paired_arrays(model: pd.DataFrame, obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model, obs = model.align(obs, join="inner", axis=0)
    model, obs = model.align(obs, join="inner", axis=1)
    x = obs.to_numpy(dtype=float).ravel()
    y = model.to_numpy(dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def nice_limit(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    hi = float(np.nanpercentile(values, 99.0))
    if hi <= 50:
        step = 10.0
    elif hi <= 150:
        step = 25.0
    elif hi <= 400:
        step = 50.0
    elif hi <= 1200:
        step = 200.0
    else:
        step = 500.0
    return max(step, np.ceil(hi / step) * step)


def axis_limit(spec: Species, values: np.ndarray) -> float:
    fixed = {
        "o3": 160.0,
        "no2": 80.0,
    }
    return fixed.get(spec.key, nice_limit(values))


def fit_line(x: np.ndarray, y: np.ndarray, limit: float) -> tuple[float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= limit) & (y >= 0.0) & (y <= limit)
    if mask.sum() < 2:
        return None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def draw_panel(ax, fig, spec: Species, panel: str, wrf_pair, aurora_pair, gridsize: int):
    wrf_x, wrf_y = wrf_pair
    aur_x, aur_y = aurora_pair
    all_x = np.concatenate([wrf_x, aur_x])
    all_y = np.concatenate([wrf_y, aur_y])
    limit = axis_limit(spec, np.concatenate([all_x, all_y]))
    plotted = (aur_x >= 0.0) & (aur_x <= limit) & (aur_y >= 0.0) & (aur_y <= limit)

    hb = ax.hexbin(
        aur_x[plotted],
        aur_y[plotted],
        gridsize=gridsize,
        extent=(0.0, limit, 0.0, limit),
        cmap="magma_r",
        mincnt=1,
        norm=LogNorm(),
        linewidths=0.0,
    )
    xx = np.array([0.0, limit])
    ax.plot(xx, xx, color="0.12", linewidth=2.8, linestyle="--", zorder=4)
    for x, y, color, label in [
        (wrf_x, wrf_y, "#1f77b4", "WRF-Chem"),
        (aur_x, aur_y, "#d62728", "Aurora"),
    ]:
        coeff = fit_line(x, y, limit)
        if coeff is None:
            continue
        slope, intercept = coeff
        ax.plot(xx, slope * xx + intercept, color=color, linewidth=3.8, label=label, zorder=5)

    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{panel} {spec.label}", loc="left", pad=7)
    ax.set_xlabel("CNEMC ($\\mu$g m$^{-3}$)")
    ax.set_ylabel("Aurora ($\\mu$g m$^{-3}$)")
    ax.grid(True, color="0.86", linewidth=0.8, linestyle="-", zorder=0)
    ax.tick_params(axis="both", which="major", width=1.75, length=6.5, direction="out")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.75)

    cbar = fig.colorbar(hb, ax=ax, orientation="vertical", fraction=0.044, pad=0.045)
    cbar.ax.tick_params(labelsize=20, width=1.30, length=4.0)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    cbar.outline.set_linewidth(1.10)


def plot_figure(output: Path, wrf_station, aurora_station, obs_station, gridsize: int) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 13.6))
    axes = axes.ravel()
    panels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, spec, panel in zip(axes, SPECIES, panels):
        wrf_pair = paired_arrays(wrf_station[spec.key], obs_station[spec.key])
        aurora_pair = paired_arrays(aurora_station[spec.key], obs_station[spec.key])
        draw_panel(ax, fig, spec, panel, wrf_pair, aurora_pair, gridsize)

    handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=3.8, label="WRF-Chem"),
        Line2D([0], [0], color="#d62728", linewidth=3.8, label="Aurora"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, handlelength=2.4, borderaxespad=0.2)
    fig.subplots_adjust(left=0.080, right=0.965, top=0.980, bottom=0.070, wspace=0.34, hspace=0.12)
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
    wrf_lons, wrf_lats = read_wrf_grid(first_wrf)
    stations = load_domain_stations(args.station_list, wrf_lons, wrf_lats)
    print(f"Using {len(times)} UTC samples: {times[0]} to {times[-1]}")
    print(f"Selected {len(stations)} CNEMC stations inside WRF d01")

    print("Loading CNEMC observations")
    obs_station = load_cnemc_observations(args.cnaq_dir, stations, times)
    print("Interpolating WRF-Chem fields to stations")
    wrf_station = load_wrf_station_values(
        args.wrfout_dir,
        times,
        wrf_lons,
        wrf_lats,
        stations,
        args.wrf_pressure_hpa * 100.0,
    )
    print("Interpolating Aurora fields to stations")
    aurora_station = load_aurora_station_values(args.aurora_root, times, stations, args.aurora_pressure_hpa)
    plot_figure(args.output, wrf_station, aurora_station, obs_station, args.gridsize)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
