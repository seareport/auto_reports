from __future__ import annotations

import logging
import os

import pandas as pd
import panel as pn
import seastats.storms
from tqdm import tqdm

from auto_reports._io import assign_oceans
from auto_reports._io import DATA_DIR
from auto_reports._io import get_model_names
from auto_reports._io import get_obs_station_names
from auto_reports._io import get_parquet_attrs
from auto_reports._io import load_data
from auto_reports._io import MODEL_DIR
from auto_reports._io import OBS_DIR

logger = logging.getLogger(name="auto-report")
CLUSTER_DURATION = 72


def sim_on_obs(sim, obs):
    obs = pd.Series(obs, name="obs")
    sim = pd.Series(sim, name="sim")
    df = pd.merge(sim, obs, left_index=True, right_index=True, how="outer")
    df["sim"] = df["sim"].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=["obs"])
    sim_ = df["sim"].drop_duplicates()
    obs_ = df["obs"].drop_duplicates()
    return sim_, obs_


def run_stats(model):
    stats = {}
    for station_sensor in tqdm(get_obs_station_names()):
        station, sensor = station_sensor.split("_")
        try:
            obs = load_data(OBS_DIR / f"{station_sensor}.parquet")
            sim = load_data(f"{model}/{station}.parquet")
            info = get_parquet_attrs(OBS_DIR / f"{station_sensor}.parquet")
            sim_, obs_ = sim_on_obs(sim, obs)
            normal_stats = seastats.get_stats(sim_, obs, seastats.GENERAL_METRICS_ALL)
            storm_stats = seastats.get_stats(
                sim_,
                obs_,
                seastats.STORM_METRICS,
                quantile=0.995,
            )
            stats[station] = {**normal_stats, **storm_stats}
            stats[station]["lon"] = float(info["lon"])
            stats[station]["lat"] = float(info["lat"])
            stats[station]["sim_std"] = sim.std()
            stats[station]["obs_std"] = obs.std()
            stats[station]["station"] = station
        except FileNotFoundError as e:
            logger.warning(e)
    return pd.DataFrame(stats).T


def run_stats_ext(model):
    extreme_events = pd.DataFrame()
    for station_sensor in tqdm(get_obs_station_names()):
        station, sensor = station_sensor.split("_")
        try:
            obs = load_data(OBS_DIR / f"{station_sensor}.parquet")
            sim = load_data(f"{model}/{station}.parquet")
            info = get_parquet_attrs(OBS_DIR / f"{station_sensor}.parquet")
            sim_, obs_ = sim_on_obs(sim, obs)
            ext_ = seastats.storms.match_extremes(sim_, obs_, quantile=0.995)
            ext_["lon"] = float(info["lon"])
            ext_["lat"] = float(info["lat"])
            ext_["station"] = station
            extreme_events = pd.concat([extreme_events, ext_])
        except FileNotFoundError as e:
            logger.warning(e)
    return extreme_events


def get_model_stats(model: str) -> pd.DataFrame:
    def load_or_generate(file_path, stats_func, file_name):
        if os.path.exists(file_path):
            logger.info(f"File {file_path} already exists")
            return pd.read_parquet(file_path)
        else:
            logger.info(f"No file {file_path} found.")
            logger.info(f"Running {file_name} for model {model}")
            path = MODEL_DIR / model
            df = stats_func(path)
            df.to_parquet(file_path)
            return df

    stat_file = DATA_DIR / f"stats/{model}.parquet"
    df_general = load_or_generate(stat_file, run_stats, "stats")
    extreme_stat_file = DATA_DIR / f"stats/{model}_eva.parquet"
    df_extreme = load_or_generate(extreme_stat_file, run_stats_ext, "extreme stats")
    df_general = assign_oceans(df_general)
    df_extreme = assign_oceans(df_extreme)
    return df_general, df_extreme


@pn.cache
def get_stats(model=None) -> dict[pd.DataFrame]:
    if model:
        models = [model]
    else:
        models = get_model_names()
    all_stats = {}
    for m in models:
        all_stats[m] = get_model_stats(m)
    return all_stats
