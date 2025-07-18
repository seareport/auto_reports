from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import panel as pn
import seastats.storms
from tqdm import tqdm

from auto_reports._io import assign_oceans
from auto_reports._io import get_model_names
from auto_reports._io import get_models_dir
from auto_reports._io import get_obs_dir
from auto_reports._io import get_obs_station_names
from auto_reports._io import get_parquet_attrs
from auto_reports._io import load_data
from auto_reports._tide import compute_rss
from auto_reports._tide import compute_score
from auto_reports._tide import concat_tides_constituents
from auto_reports._tide import pytide_get_coefs
from auto_reports._tide import pytides_to_df
from auto_reports._tide import reduce_coef_to_fes
from auto_reports.graphs.tidal_plots import END
from auto_reports.graphs.tidal_plots import START

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


def run_stats(data_dir: Path, model: str):
    stats = {}
    obs_dir = get_obs_dir(data_dir)
    model_dir = get_models_dir(data_dir) / model
    for station_sensor in tqdm(get_obs_station_names(data_dir)):
        station, sensor = station_sensor.split("_")
        try:
            obs = load_data(obs_dir / f"{station_sensor}.parquet")
            sim = load_data(model_dir / f"{station}.parquet")
            info = get_parquet_attrs(obs_dir / f"{station_sensor}.parquet")
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


def run_stats_ext(data_dir: Path, model: str):
    extreme_events = pd.DataFrame()
    obs_dir = get_obs_dir(data_dir)
    model_dir = get_models_dir(data_dir) / model
    for station_sensor in tqdm(get_obs_station_names(data_dir)):
        station, sensor = station_sensor.split("_")
        try:
            obs = load_data(obs_dir / f"{station_sensor}.parquet")
            sim = load_data(model_dir / f"{station}.parquet")
            info = get_parquet_attrs(obs_dir / f"{station_sensor}.parquet")
            sim_, obs_ = sim_on_obs(sim, obs)
            ext_ = seastats.storms.match_extremes(sim_, obs_, quantile=0.995)
            ext_["lon"] = float(info["lon"])
            ext_["lat"] = float(info["lat"])
            ext_["station"] = station
            extreme_events = pd.concat([extreme_events, ext_])
        except FileNotFoundError as e:
            logger.warning(e)
    return extreme_events


def run_stats_tide(data_dir: Path, model: str, const: list):
    tide_all = pd.DataFrame()
    obs_dir = get_obs_dir(data_dir)
    model_dir = get_models_dir(data_dir) / model
    for station_sensor in tqdm(get_obs_station_names(data_dir)):
        station, sensor = os.path.splitext(station_sensor)[0].split("_")
        try:
            obs = load_data(obs_dir / f"{station_sensor}.parquet")
            sim = load_data(model_dir / f"{station}.parquet")
            info = get_parquet_attrs(obs_dir / f"{station_sensor}.parquet")
            out_pytides_sim = pytide_get_coefs(sim, 20)
            out_pytides_obs = pytide_get_coefs(obs, 20)
            pytides_reduced_coef_sim = reduce_coef_to_fes(
                pytides_to_df(out_pytides_sim),
                cnst=const,
            )
            pytides_reduced_coef_obs = reduce_coef_to_fes(
                pytides_to_df(out_pytides_obs),
                cnst=const,
            )
            tide_ = concat_tides_constituents(
                {
                    "sim": pytides_reduced_coef_sim,
                    "obs": pytides_reduced_coef_obs,
                },
            )
            tide_["station"] = station
            tide_["lon"] = float(info["lon"])
            tide_["lat"] = float(info["lat"])
            tide_["station"] = station
            rss_sim = compute_rss(tide_, "amplitude", "sim", "obs")

            # compute stats on TS for the 3 first months of pure tidal signal
            ts_sim_obs, _ = sim_on_obs(sim.loc[START:END], obs.loc[START:END])
            df_sim_obs = pd.concat(
                {
                    "sim": ts_sim_obs,
                    "obs": obs.loc[START:END],
                },
                axis=1,
            )

            corr_matrix = df_sim_obs.corr(method="pearson")
            corr_sim = corr_matrix.loc["sim", "obs"]
            score_sim = compute_score(corr_sim, float(rss_sim))
            tide_["corr"] = corr_sim
            tide_["rss"] = rss_sim
            tide_["score"] = score_sim
            tide_all = pd.concat([tide_all, tide_])
        except Exception as e:
            logger.warning(e)
    return tide_all


class StatsRunner:
    def __init__(self, data_dir: Path, const: list):
        self.data_dir = data_dir
        self.const = const
        os.makedirs(self.data_dir / "stats", exist_ok=True)

    def load_or_generate(self, file_path, stats_func, file_name, model, **kwargs):
        if file_path.exists():
            logger.info(f"File {file_path} already exists")
            return pd.read_parquet(file_path)
        else:
            logger.info(f"No file {file_path} found.")
            logger.info(f"Running {file_name} for model {model}")
            df = stats_func(self.data_dir, model, **kwargs)
            df.to_parquet(file_path)
            return df

    def run_model_stats(self, model: str):
        stat_file = self.data_dir / f"stats/{model}.parquet"
        extreme_stat_file = self.data_dir / f"stats/{model}_eva.parquet"
        tide_stat_file = self.data_dir / f"stats/{model}_tide.parquet"

        df_general = self.load_or_generate(stat_file, run_stats, "stats", model)
        df_extreme = self.load_or_generate(
            extreme_stat_file,
            run_stats_ext,
            "extreme stats",
            model,
        )
        df_tides = self.load_or_generate(
            tide_stat_file,
            run_stats_tide,
            "tidal stats",
            model,
            const=self.const,
        )

        df_general = assign_oceans(df_general)
        df_extreme = assign_oceans(df_extreme)
        return df_general, df_extreme, df_tides


@pn.cache
def get_stats(data_dir, const: list, model=None) -> dict[str, tuple[pd.DataFrame]]:
    runner = StatsRunner(Path(data_dir), const)

    if model:
        models = [model]
    else:
        models = get_model_names(data_dir)

    return {m: runner.run_model_stats(m) for m in models}
