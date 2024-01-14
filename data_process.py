from __future__ import annotations

# main packages
import os
import numpy as np
import pandas as pd
import logging
import xarray as xr

# jrc packages
from pyposeidon.utils.statistics import get_stats
from analysea.utils import cleanup, detect_time_step
from analysea.tide import detide
from searvey.multi import multiprocess
import observer

# global variables
from common import (
    BASE_URL,
    VALID_SENSORS,
    KURTOSIS_THRESHOLD,
    NCPU,
    OPTS,
    STORAGE_AZ,
    CONTAINER,
)
from typing import List


#  DATA - FOLDERS
def ensure_directory(path: str) -> None:
    """Ensure that the directory exists; create it if it doesn't."""
    os.makedirs(
        path, exist_ok=True
    )  # No need to check if it exists, makedirs can do it


def ioc_subset_from_files_in_folder(
    stations: pd.DataFrame, folder: str, ext: str = ".json"
):
    """this function return a subset of the ioc database from all the files (json or parquet)
    present in a folder
    """
    list_files = []
    if ext == ".json":
        sensor_list = []
    for file in os.listdir(folder):
        name = file.split(ext)[0]
        if file.endswith(ext):
            if ext == ".json":
                code, sensor = name.split("_")
                list_files.append(code)
                sensor_list.append(sensor)
            elif ext == ".parquet":
                list_files.append(name)
            elif ext == ".csv":
                list_files.append(name)

    boolist = stations.ioc_code.isin(list_files)
    res = stations[boolist]
    if ext == ".json":
        res["sensor"] = sensor_list
        for i_s, station in enumerate(list_files):
            idx = res.ioc_code.tolist().index(station)
            res["sensor"].iloc[idx] = sensor_list[i_s]

    return res


# DATA - PROCESSING
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "sensor" not in df.columns:
        return pd.DataFrame()
    df = df[df.sensor.isin(VALID_SENSORS)]
    if df.empty or len(df) < 1:
        return pd.DataFrame()
    df = df.assign(stime=pd.to_datetime(df.stime))
    df = df.rename(columns={"stime": "time"})
    df = df.pivot(index="time", columns="sensor", values="slevel")
    df._mgr.items.name = ""
    return df


def generate_dicts(
    stations: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    obs_root: str,
    suffix: str = ".csv",
) -> list[str]:
    dicts = []
    for ioc_code in stations.ioc_code:
        dicts.append(
            dict(
                station=ioc_code,
                start=start,
                end=end,
                obs_folder=obs_root,
                suffix=suffix,
            )
        )
    return dicts


def generate_dicts2(stations: pd.DataFrame, obs_root: str) -> list[str]:
    dicts = []
    for ioc_code in stations.ioc_code:
        dicts.append(dict(station=ioc_code, obs_root=obs_root))
    return dicts


def get_stations(
    ioc_df: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    obs_folder: str,
    suffix=".csv",
) -> pd.DataFrame:
    os.makedirs(obs_folder, exist_ok=True)
    # scrape using observer
    data = observer.scrape_ioc(
        ioc_codes=ioc_df.ioc_code, start_date=start, end_date=end
    )
    # write all non empty dataframe
    for station in data.keys():
        if not data[station].empty:
            df = data[station]
            write_df(df, os.path.join(obs_folder, station + suffix))


def get_one_ioc(
    station: str, start: pd.Timestamp, end: pd.Timestamp, obs_folder: str, suffix=".csv"
):
    # scrape using observer
    no_years = pd.Timestamp.now().year - start.year + 1
    df = observer.get_ioc_df(station, no_years=no_years)
    if len(df) > 0:
        mask = (df.index >= start) & (df.index <= end)
        df = df[mask]
        if len(df) > 0:
            write_df(df, os.path.join(obs_folder, station + suffix))


def get_multi_ioc(
    ioc_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    obs_folder: str,
    suffix=".csv",
):
    logging.info("cleaning selected stations..")
    ioc_dict = generate_dicts(ioc_df, start, end, obs_folder, suffix)
    multiprocess(
        get_one_ioc,  # with the observer package
        ioc_dict,
        n_workers=NCPU,  ##!!CAREFUL!! here adapt the numper of procs to your machine !
        disable_progress_bar=False,
    )
    return ioc_subset_from_files_in_folder(ioc_df, obs_folder, ext=suffix)


def write_df(df, fout):
    if fout.endswith(".csv"):
        df.to_csv(fout)
    elif fout.endswith(".parquet"):
        df.to_parquet(fout)
    else:
        raise ValueError(f"format {format} not supported")


def read_df(fin):
    if fin.endswith(".csv"):
        df = pd.read_csv(fin, index_col=0, parse_dates=True)
    elif fin.endswith(".parquet"):
        df = pd.read_parquet(fin)
    else:
        raise ValueError(f"format {format} not supported")
    return df


def clean_one_ioc(station: str, obs_root: str, suffix=".csv") -> None:
    #  read station data
    obs_raw = os.path.join(obs_root, "raw")
    obs_clean = os.path.join(obs_root, "clean")
    fileClean = f"{obs_clean}/{station}{suffix}"
    ts = read_df(os.path.join(obs_raw, station + suffix))
    df = cleanup(ts, kurtosis=KURTOSIS_THRESHOLD)
    if len(df) > 0:
        write_df(df, fileClean)


def clean_and_select_ioc(stations: pd.DataFrame, obs_root: str) -> None:
    logging.info("cleaning selected stations..")
    ioc_to_clean = generate_dicts2(stations, obs_root)
    multiprocess(
        clean_one_ioc,  # with the observer package
        ioc_to_clean,
        n_workers=NCPU,  ##!!CAREFUL!! here adapt the numper of procs to your machine !
        disable_progress_bar=False,
    )


def convert_zarr_to_netcdf(fn, storage_options, fout):
    with xr.open_dataset(
        fn, storage_options=storage_options, engine="zarr"
    ) as ds_model:
        ds_model.to_netcdf(fout)


def get_model_data(
    ds: xr.Dataset,
    station: str,
    id_column: str = "id",
    elev_column: str = "elev_sim",
    prefix: str = "IOC-",
) -> pd.DataFrame:
    ix = np.where(ds[id_column].values == prefix + station)[0]
    if len(ix) == 1:
        tg = ds[{id_column: ix}]
        tg_df = tg.to_dataframe()
        sim = tg_df.reset_index(level=id_column).drop(columns=id_column)
        return sim[elev_column]
    else:
        return pd.DataFrame()


def compute_stats(obs: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    # compare time steps: because otherwise reindex induces error
    # https://github.com/ec-jrc/pyPoseidon/blob/d32f16bee6968426f143f060f62d4ee37d9f0fca/pyposeidon/utils/statistics.py#L38C22-L38C22
    try:
        ts_sim = detect_time_step(sim)
        ts_obs = detect_time_step(obs)

        obs = obs.dropna()
        sim = sim.dropna()

        if ts_obs < ts_sim:
            stats = get_stats(obs, sim)
            stats["BIAS or mean error"] = -stats["BIAS or mean error"]
        else:
            stats = get_stats(sim, obs)
        return stats
    except Exception as e:
        logging.error(
            f"Failed to compute stats: {e}\n > obs \n{obs.head()} \n > sim \n{sim.head()}"
        )
        return pd.DataFrame()


def compare_one_ioc(
    station: str,
    lat: float,
    obs_root: str,
    opts: dict,
    t_rsp: int = 30,
    interp: bool = False,
) -> pd.DataFrame():
    obs_data = read_df(os.path.join(obs_root, "clean", station + ".csv"))
    sim = read_df(os.path.join(obs_root, "model", station + ".csv"))
    sim = sim.iloc[:, 0]  # take only the first column, whatever its name
    local_opts = opts.copy()
    local_opts["lat"] = lat
    for sensor in obs_data.columns:
        ss = obs_data[sensor]
        # resample
        h_rsmp = ss.resample(f"{t_rsp}min").mean().shift(freq=f"{int(t_rsp/2)}min")
        # detide
        obs = detide(h_rsmp, **local_opts)
        if interp:
            obs = obs.interpolate()
        stats = compute_stats(obs, sim)
        if len(obs) > 0:
            write_df(obs.to_frame(), os.path.join(obs_root, "surge", station + ".csv"))
        # add sensor info
        stats["sensor"] = sensor
        return pd.DataFrame(
            data={key: val for key, val in stats.items()}, index=[station]
        )


def generate_ioc_comparison_inputs(
    stations: pd.DataFrame, obs_folder: str, opts: dict
) -> List[dict]:
    inputs = []
    for i_s, station in enumerate(stations.ioc_code):
        lat = stations.iloc[i_s].latitude
        inputs.append(dict(station=station, lat=lat, obs_root=obs_folder, opts=opts))
    return inputs


def extract_from_ds(
    stations: pd.DataFrame,
    work_folder: str,
    ds_model: xr.Dataset,
    id_column: str,
    elev_column: str,
    prefix: str,
    t_rsp: int = 30,
) -> pd.DataFrame:
    #
    os.makedirs(work_folder, exist_ok=True)
    # #
    # for id in stations[id_column]:
    #     sim = get_model_data(ds_model, id, id_column, elev_column, prefix)
    #     if len(sim) > 0:
    #         sim.resample(f"{t_rsp}min").mean().shift(freq=f"{int(t_rsp/2)}min").to_csv(
    #             os.path.join(work_folder, id + ".csv")
    #         )

    extracted = ioc_subset_from_files_in_folder(stations, work_folder, ext=".csv")
    return extracted


def compute_surge_comparison(
    stations: pd.DataFrame,
    obs_folder: str,
    opts: dict = OPTS,
):
    logging.info("Computing model vs obs surge comparison..")
    inputs = generate_ioc_comparison_inputs(stations, obs_folder, opts)
    # the line equation:
    results = multiprocess(
        compare_one_ioc,
        inputs,
        n_workers=NCPU,  ##!!CAREFUL!! here adapt the numper of procs to your machine !
        disable_progress_bar=False,
    )
    res = pd.DataFrame()
    for result in results:
        res = pd.concat([res, result.result])
    return res


def compute_surge_comparison_serial(
    stations: pd.DataFrame,
    obs_folder: str,
    opts: dict = OPTS,
):
    logging.info("Computing model vs obs surge comparison.. (sequential execution)")
    inputs = generate_ioc_comparison_inputs(
        stations,
        obs_folder,
        opts,
    )
    # the line equation:
    res = pd.DataFrame()
    for i_, inp in enumerate(inputs):
        result = compare_one_ioc(**inp)
        res = pd.concat([res, result])
    return res


def avg_ts(timestamps: list) -> pd.Timestamp:
    """
    Convert the times to seconds, then average, then back convert to pd.Timestamp
    """
    seconds = 0
    sec_len = 0
    for ts in timestamps:
        try:
            seconds += ts.timestamp()
            sec_len += 1
        except Exception as e:
            print(e)
            pass
    average_seconds = seconds / sec_len
    return pd.Timestamp(average_seconds, unit="s")


def get_storm_peak_time(ds, station):
    sim = get_model_data(ds, station)
    return sim.idxmax()
