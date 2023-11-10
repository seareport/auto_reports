
from __future__ import annotations
# main packages
import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
import xarray as xr

# jrc packages
from pyposeidon.utils.statistics import get_stats
from analysea.utils import cleanup, detect_time_step,completeness
from analysea.tide import detide
from searvey.multi import multiprocess

# global variables
from common import BASE_URL, VALID_SENSORS, KURTOSIS_THRESHOLD, NCPU, OPTS

#  DATA - FOLDERS
def ensure_directory(path: str) -> None:
    """Ensure that the directory exists; create it if it doesn't."""
    os.makedirs(path, exist_ok=True)  # No need to check if it exists, makedirs can do it


def ioc_subset_from_files_in_folder(stations: pd.DataFrame, folder:str, ext:str=".json"):
    """ this function return a subset of the ioc database from all the files (json or parquet)
        present in a folder
    """
    list_files = []    
    if ext ==".json": 
        sensor_list = []
    for file in os.listdir(folder):
        name = file.split(ext)[0]
        if file.endswith(ext): 
            if ext == '.json':
                code, sensor = name.split("_")
                list_files.append(code)
                sensor_list.append(sensor)
            elif ext == ".parquet": 
                list_files.append(name)
            elif ext == ".csv": 
                list_files.append(name)
    
    boolist = stations.ioc_code.isin(list_files)
    res = stations[boolist]
    if ext ==".json": 
        res["sensor"] = sensor_list
        for i_s,station in enumerate(list_files):
            idx = res.ioc_code.tolist().index(station)
            res["sensor"].iloc[idx] = sensor_list[i_s]
            
    return res

# DATA - PROCESSING
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'sensor' not in df.columns:
        return pd.DataFrame()
    df = df[df.sensor.isin(VALID_SENSORS)]
    if df.empty or len(df)<1:
        return pd.DataFrame()
    df = df.assign(stime=pd.to_datetime(df.stime))
    df = df.rename(columns={"stime": "time"})
    df = df.pivot(index="time", columns="sensor", values="slevel")
    df._mgr.items.name = ""
    return df


def generate_dicts(stations: pd.DataFrame, start: pd.Timestamp,end: pd.Timestamp, obs_root: str) -> list[str]:
    dicts = []
    for ioc_code in stations.ioc_code:
        st_ = start.strftime("%Y-%m-%d")
        en_ = end.strftime("%Y-%m-%d")
        url = BASE_URL.format(ioc_code=ioc_code, start=st_, end=en_)
        dicts.append(dict(url=url, station = ioc_code, min_time = st_, max_time = en_, obs_root=obs_root))
    return dicts


def process_url(url):
    df = pd.read_json(url)
    if df.empty or len(df)<1:
        return pd.DataFrame()
    df = normalize_df(df)
    return df


def write_df(df, fout): 
    if fout.endswith('.csv'):
        df.to_csv(fout)
    elif fout.endswith('.parquet'): 
        df.to_parquet(fout)
    else: 
        raise ValueError(f"format {format} not supported")


def clean_one_ioc(url:str,station: str, min_time:str, max_time:str, obs_root: str, format = 'csv') -> None:
    #  read station data
    obs_raw = os.path.join(obs_root,"raw")
    obs_clean = os.path.join(obs_root,"clean")
    file = f"{obs_raw}/{station}.{format}"
    fileClean = f"{obs_clean}/{station}.{format}"
    if ~os.path.exists(file):
        ts = process_url(url)
        write_df(ts,file)
    if ~os.path.exists(fileClean):
        df = cleanup(ts, kurtosis=KURTOSIS_THRESHOLD)
        if len(df)>0:
            write_df(df,fileClean)

    
def clean_and_select_ioc(stations:pd.DataFrame, tmin : pd.Timestamp, tmax: pd.Timestamp, obs_root:str) -> None:
    logging.info("cleaning selected stations..")
    ioc_to_clean = generate_dicts(stations, tmin, tmax, obs_root)
    multiprocess(
        clean_one_ioc,
        ioc_to_clean,
        n_workers=NCPU, ##!!CAREFUL!! here adapt the numper of procs to your machine !
        disable_progress_bar=False,
    )

def get_obs_data(obs_folder: str, station: str, format: str = 'csv') -> pd.DataFrame:
    filein = os.path.join(obs_folder,station+'.'+format)
    try:
        if not os.path.exists(filein):
            return pd.DataFrame()
        if format == 'csv':
            return pd.read_csv(filein, index_col=0, parse_dates=True)
        elif format == 'parquet': 
            return pd.read_parquet(filein)
    except Exception as e:
        logging.error(f"Failed to read {filein}: {e}")
        return pd.DataFrame()


def convert_zarr_to_netcdf(fn,storage_options,  fout ): 
    with xr.open_dataset(fn, storage_options=storage_options, engine="zarr") as ds_model:
        ds_model.to_netcdf(fout)


def get_model_data(ds_model, station): 
    ix = np.where(ds_model.id.values == 'IOC-' + station)[0]
    if len(ix) == 1:
        tg = ds_model[dict(id=ix)]
        tg2 = tg.to_dataframe()
        sim = tg2.reset_index(level='id').drop(columns='id')
        return sim['elev_sim']
    else: 
        return pd.DataFrame()


def compute_stats(obs,sim): 
    # compare time steps: because otherwise reindex induces error
    # https://github.com/ec-jrc/pyPoseidon/blob/d32f16bee6968426f143f060f62d4ee37d9f0fca/pyposeidon/utils/statistics.py#L38C22-L38C22
    try: 
        ts_sim = detect_time_step(sim)
        ts_obs = detect_time_step(obs)
        if ts_obs<ts_sim:
            stats = get_stats(obs,sim)
        else : 
            stats = get_stats(sim,obs)
        return stats
    except Exception as e:
        logging.error(f"Failed to compute stats: {e}")
        return pd.DataFrame()


def compare_one_ioc(station:str, 
                    lat:float,
                    obs_folder:str, 
                    opts:dict, 
                    ds_model:xr.Dataset, 
                    plot: bool = False,
                    avg: bool = True,
    ) -> pd.DataFrame():
    obs_data = get_obs_data(obs_folder,station)
    local_opts = opts.copy()
    local_opts['lat'] = lat
    for sensor in obs_data.columns: 
        ss = obs_data[sensor]
        # detide
        obs= detide(ss,local_opts)
        sim = get_model_data(ds_model, station) 
        stats = compute_stats(obs, sim)
        if len(obs)>0:
            write_df(obs.to_frame(), os.path.join(obs_folder,station+'.csv') ) 
        # add sensor info
        stats['sensor'] = sensor
        return pd.DataFrame(data = {key:val for key,val in stats.items()} , index=[station])
    

def generate_ioc_comparison_inputs(stations: pd.DataFrame, 
                                   obs_folder:str, 
                                   opts: dict,
                                   ds_model:xr.Dataset):
    inputs = []
    N = len(stations)
    for i_s, station in enumerate(stations.ioc_code):
        lat = stations.iloc[i_s].latitude
        inputs.append(dict(station=station,
                           lat=lat,
                           obs_folder=obs_folder, 
                           opts=opts,
                           ds_model=ds_model))
    return inputs


def compute_surge_comparison(stations: pd.DataFrame, 
                             obs_folder:str, 
                             ds_model:xr.Dataset,
                             opts:dict = OPTS):    
    logging.info("Computing model vs obs surge comparison..")
    inputs = generate_ioc_comparison_inputs(stations, 
                                           obs_folder, 
                                           opts, 
                                           ds_model)
    # the line equation:
    results = multiprocess(
        compare_one_ioc,
        inputs,
        n_workers=NCPU, ##!!CAREFUL!! here adapt the numper of procs to your machine !
        disable_progress_bar=False,
    )
    res = pd.DataFrame()
    for result in results:
        res = pd.concat([res, result.result])
    return res


def compute_surge_comparison_serial(stations: pd.DataFrame, 
                             obs_folder:str, 
                             ds_model:xr.Dataset,
                             opts:dict = OPTS):    
    logging.info("Computing model vs obs surge comparison.. (sequential execution)")
    inputs = generate_ioc_comparison_inputs(stations, 
                                           obs_folder, 
                                           opts, 
                                           ds_model)
    # the line equation:
    res = pd.DataFrame()
    for inp in inputs:
        result = compare_one_ioc(**inp)
        res = pd.concat([res, result])
    return res


def avg_ts(timestamps:list) ->pd.Timestamp:
    """
    Convert the times to seconds, then average, then back convert to pd.Timestamp
    """
    seconds = [ts.timestamp() for ts in timestamps]
    average_seconds = sum(seconds) / len(seconds)
    return pd.Timestamp(average_seconds, unit='s')


def get_storm_peak_time(ds, station): 
    sim = get_model_data(ds, station)
    return sim.idxmax()