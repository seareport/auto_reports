from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from analysea.tide import detide
from analysea.utils import cleanup
from analysea.utils import detect_time_step
from analysea.utils import interpolate
from analysea.utils import resample
from pyposeidon.utils.statistics import get_stats
from searvey.coops import fetch_coops_station
from searvey.ioc import fetch_ioc_station
from searvey.multi import multiprocess

from .common import KURTOSIS_THRESHOLD
from .common import NCPU
from .common import OPTS
from .common import VALID_SENSORS


#  DATA - FOLDERS
def ensure_directory(path: str) -> None:
    """Ensure that the directory exists; create it if it doesn't."""
    os.makedirs(
        path, exist_ok=True
    )  # No need to check if it exists, makedirs can do it


def seaset_subset_from_files_in_folder(
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

    stations.loc[:, "nos_id_str"] = (
        stations[~pd.isna(stations["nws_id"])]["nos_id"].astype(int).astype(str)
    )
    boolist1 = stations.ioc_code.isin(list_files)  # IOC
    boolist2 = stations.nos_id_str.isin(list_files)  # CO-OPS
    res = stations[boolist1 | boolist2]

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


def generate_dicts_fetch(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    obs_root: str,
    ext: str = ".csv",
) -> list[str]:
    dicts = []

    for ii in range(len(df)):
        if not pd.isna(df.iloc[ii]["ioc_code"]):
            # ioc takes priority over coops
            provider = "ioc"
            station = df.iloc[ii]["ioc_code"]
        elif not pd.isna(df.iloc[ii]["nos_id"]):
            provider = "coops"
            station = df.iloc[ii]["nos_id"].astype(int).astype(str)
        else:
            # other providers not yet implemented
            continue
        dicts.append(
            dict(
                provider=provider,
                station=station,
                start=start,
                end=end,
                obs_folder=obs_root,
                ext=ext,
            )
        )
    return dicts


def generate_dicts_clean(
    df: pd.DataFrame,
    obs_root: str,
    ext: str = ".csv",
    t_rsp: int = 30,
) -> list[str]:
    dicts = []

    for ii in range(len(df)):
        if not pd.isna(df.iloc[ii]["ioc_code"]):
            # ioc takes priority over coops
            provider = "ioc"
            station = df.iloc[ii]["ioc_code"]
        elif not pd.isna(df.iloc[ii]["nos_id"]):
            provider = "coops"
            station = df.iloc[ii]["nos_id"].astype(int).astype(str)
        else:
            # other providers not yet implemented
            continue
        dicts.append(
            dict(
                provider=provider,
                station=station,
                obs_root=obs_root,
                ext=ext,
                t_rsp=t_rsp,
            )
        )
    return dicts


def get_one_provider(
    provider: str,
    station: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    obs_folder: str,
    ext=".csv",
):
    try:
        if not os.path.exists(os.path.join(obs_folder, str(station) + ext)):
            if provider == "ioc":
                df = fetch_ioc_station(station, start, end)
            elif provider == "coops":
                df = fetch_coops_station(station, start, end)
                start = start.tz_localize("UTC") if start.tzinfo is None else start
                end = end.tz_localize("UTC") if end.tzinfo is None else end
            if len(df) > 0:
                mask = (df.index >= start) & (df.index <= end)
                df = df[mask]
                if len(df) > 0:
                    write_df(df, os.path.join(obs_folder, str(station) + ext))
    except Exception as e:
        logging.error(f"failed to fetch {provider} station {station}")
        logging.error(e)


def get_multi_provider(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    obs_folder: str,
    ext=".csv",
):
    """
    This function fetches data from multiple sources (IOC and CO-OPS) for the stations in the input dataframe.

    Args:
        df (pd.DataFrame): A dataframe containing the stations to fetch data for.
        start (pd.Timestamp): The start date for the data fetch.
        end (pd.Timestamp): The end date for the data fetch.
        obs_folder (str): The folder where the observed data is stored.
        ext (str, optional): The file extension of the observed data. Defaults to ".csv".

    Returns:
        pd.DataFrame: A dataframe containing the stations and their data sources.
    """
    logging.info("fetching IOC/COOPS stations..")
    seaset_fetch_dict = generate_dicts_fetch(df, start, end, obs_folder, ext)
    for inp in seaset_fetch_dict:
        get_one_provider(**inp)
    return seaset_subset_from_files_in_folder(df, obs_folder, ext=ext)


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


def clean_one_seaset(
    provider: str,
    station: str,
    obs_root: str,
    ext: str = ".csv",
    t_rsp: int = 30,
) -> None:
    #  read station data
    obs_raw = os.path.join(obs_root, "raw")
    obs_clean = os.path.join(obs_root, "clean")
    fileClean = f"{obs_clean}/{station}{ext}"
    fileRaw = f"{obs_raw}/{station}{ext}"
    ts = read_df(fileRaw)
    if provider == "coops":
        ts = ts.drop(columns=["quality", "flags"])
    df = cleanup(ts, kurtosis=KURTOSIS_THRESHOLD)
    df = resample(df, t_rsp=t_rsp)
    df = interpolate(df, t_rsp=t_rsp)
    if len(df) > 0:
        write_df(df, fileClean)


def clean_and_select_seaset(
    stations: pd.DataFrame,
    obs_root: str,
    ext: str = ".csv",
    t_rsp: int = 30,
) -> None:
    logging.info("cleaning selected stations..")
    seaset_clean_dict = generate_dicts_clean(stations, obs_root, ext, t_rsp)
    # for inp in seaset_clean_dict:
    #     clean_one_seaset(**inp)
    multiprocess(
        clean_one_seaset,  # with the observer package
        seaset_clean_dict,
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
) -> pd.DataFrame:
    ix = np.where(ds[id_column].values == station)[0]
    if len(ix) == 1:
        tg = ds[{id_column: ix}]
        tg_df = tg.to_dataframe()
        sim = tg_df.reset_index(level=id_column).drop(columns=id_column)
        return sim
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


def compare_one_seaset(
    station: str,
    lat: float,
    obs_root: str,
    opts: dict,
    t_rsp: int = 30,
    ext: str = ".csv",
) -> pd.DataFrame:
    obs_data = read_df(os.path.join(obs_root, "clean", f"{station}.parquet"))
    sim = read_df(os.path.join(obs_root, "model", f"{station}.parquet"))
    sim = sim.iloc[:, 0]  # take only the first column, whatever its name
    local_opts = opts.copy()
    local_opts["lat"] = lat
    for sensor in obs_data.columns:
        ss = obs_data[sensor]
        # resample
        h_rsmp = resample(ss, t_rsp=t_rsp)
        h_rsmp = interpolate(h_rsmp, t_rsp=t_rsp)
        # detide
        obs = detide(h_rsmp, **local_opts)
        stats = compute_stats(obs, sim)
        if len(obs) > 0:
            write_df(
                obs.to_frame(), os.path.join(obs_root, "surge", f"{station}.parquet")
            )
        # add sensor info
        stats["sensor"] = sensor
        return pd.DataFrame(
            data={key: val for key, val in stats.items()}, index=[station]
        )


def generate_ioc_comparison_inputs(
    stations: pd.DataFrame, obs_folder: str, opts: dict, ext: str = ".csv"
) -> List[dict]:
    inputs = []
    for i_s, station in enumerate(stations.ioc_code):
        lat = stations.iloc[i_s].latitude
        inputs.append(
            dict(station=station, lat=lat, obs_root=obs_folder, opts=opts, ext=ext)
        )
    return inputs


def extract_from_ds(
    stations: pd.DataFrame,
    work_folder: str,
    ds_model: xr.Dataset,
    id_column: str,
    ext: str = ".csv",
    t_rsp: int = 30,
) -> pd.DataFrame:
    #
    os.makedirs(work_folder, exist_ok=True)
    #
    stations.loc[:, "nos_id_str"] = (
        stations[~pd.isna(stations["nws_id"])]["nos_id"].astype(int).astype(str)
    )
    for id in stations[id_column]:
        ioc_id = stations[stations.seaset_id == id].ioc_code.values[0]
        coops_id = stations[stations.seaset_id == id].nos_id_str.values[0]
        if not pd.isna(ioc_id):
            name = ioc_id
        elif not pd.isna(coops_id):
            name = coops_id
        sim = get_model_data(ds_model, id, id_column)
        if len(sim) > 0:
            sim = resample(sim, t_rsp=t_rsp)
            sim = interpolate(sim, t_rsp=t_rsp)
            write_df(sim, os.path.join(work_folder, name + ext))

    extracted = seaset_subset_from_files_in_folder(stations, work_folder, ext=ext)
    return extracted


def compute_surge_comparison(
    stations: pd.DataFrame,
    obs_folder: str,
    opts: dict = OPTS,
    ext: str = ".csv",
):
    logging.info("Computing model vs obs surge comparison..")
    inputs = generate_ioc_comparison_inputs(stations, obs_folder, opts, ext)
    # the line equation:
    results = multiprocess(
        compare_one_seaset,
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
        result = compare_one_seaset(**inp)
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
