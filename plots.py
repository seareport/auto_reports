import os 
import numpy as np
import pandas as pd 
import xarray as xr
import geopandas as gpd

import logging
import shapely 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from searvey import ioc
import thalassa

from data_process import (ensure_directory, 
                          get_model_data, 
                          get_obs_data, clean_and_select_ioc, 
                          compute_surge_comparison_serial,
                          ioc_subset_from_files_in_folder, 
                          get_storm_peak_time, avg_ts)
from common import CONTAINER, COASTLINES, SEASET_CATALOG, STORAGE_AZ, NOW, START

# PLOTS
def normalize(values, vmin, vmax):
    normalized = np.clip(values, vmin, vmax)
    return normalized


def normalize_bar(values, vmin, vmax):
    values = (values - vmin) / (vmax - vmin)
    normalized = np.clip(values, vmin, vmax)
    return normalized


def plot_stations_map(skill, 
                      skill_param='RMSE', 
                      vmin = 0, 
                      vmax=1,
                      ax = None, 
                      time = None, 
                      ds = None, 
                      bbox = None, 
                      colors = None,):
    """
    Plot a geographical distribution of station RMSE differences with side histograms.
    
    Parameters:
    - res: Dataframe for statistics of stations
    - stations: DataFrame, contains station information including ioc_code, longitude, and latitude.
    - skill_param: str, the name of the column in 'res' that contains the values to plot.
    - vmax: numeric, the maximum value for the colorbar.
    
    The function will plot a main map with station RMSEs, a horizontal histogram of station counts along longitude,
    and a vertical histogram of station counts along latitude.
    """
    try:
        if ax is None: fig, ax = plt.subplots(figsize=(29.7/2,21))
        plt.tight_layout()
        divider = make_axes_locatable(ax)
        ax1 = divider.append_axes('bottom', size='3%', pad=0.3)
        ax2 = divider.append_axes('top', size='14%', pad=0.3)
        ax3 = divider.append_axes('right', size='10%', pad=0.3)        
        ax4 = divider.append_axes('right', size='2%', pad=0.3)
        ax2.sharex(ax)
        ax3.sharey(ax)
        
        countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        _ = countries.plot(color='lightgrey', ax=ax, zorder=-1)
        if time is not None: 
            if ds is not None: 
                levels = np.arange(-2, 2, 0.1)
                cmap = plt.colormaps['coolwarm']
                im = plot_simple_azure(time,ds, ax=ax, bbox=bbox, levels=levels, cmap=cmap)
                plt.colorbar(im, cax=ax1, orientation='horizontal').set_label('Elevation [m]')
                ax.set_xlim([bbox.bounds[0]-0.5, bbox.bounds[2]+0.5])
                ax.set_ylim([bbox.bounds[1]-0.5, bbox.bounds[3]+0.5])
        
        normalized_values = normalize(skill[skill_param],  vmin,  vmax)
        im = ax.scatter(skill.longitude, skill.latitude, c=normalized_values, cmap='jet', vmin=vmin, vmax=vmax, marker='.', s=300)
        if colors is not None: im = ax.scatter(skill.longitude, skill.latitude, c=colors, marker='.', s=50)
        plt.colorbar(im, cax=ax4).set_label(skill_param)

        plt.suptitle(f"Geographical distribution of {skill_param} differences for {len(skill)} tide gauges")
        plot_side_histograms(skill, ax2, ax3, skill_param, vmin, vmax)
        plt.savefig('stations_map_'+skill_param+'.png')
        
    except Exception as e:
        logging.ERROR(f"An error occurred: {e}")


def plot_side_histograms(df, ax_horizontal, ax_vertical, value_col, vmin, vmax):
    """
    Plot side histograms for the given DataFrame.
    
    Parameters:
    - df: DataFrame, the data source for the histograms.
    - ax_horizontal: AxesSubplot, the horizontal axis for longitude histogram.
    - ax_vertical: AxesSubplot, the vertical axis for latitude histogram.
    - value_col: str, the name of the column with values to calculate the mean for coloring.
    - vmax: numeric, the maximum value for normalization of the color map.
    """
    binsLon = np.arange(-180, 180, 0.5)
    binsLat = np.arange(-90, 90, 0.5)
    cmap = matplotlib.colormaps['jet']

    # Histogram for longitude
    for ibi, _ in enumerate(binsLon[:-1]):
        minlo = binsLon[ibi]
        maxlo = binsLon[ibi+1]
        tmp = df.loc[(df['longitude'] > minlo) & (df['longitude'] <= maxlo)]
        val = tmp[value_col].mean()
        norm = normalize_bar(val,  vmin,  vmax)
        ax_horizontal.bar((minlo + maxlo) / 2, tmp.size, color=cmap(norm), width=maxlo - minlo)
    ax_horizontal.set_ylabel('Number of stations')

    # Histogram for latitude
    for ibi, _ in enumerate(binsLat[:-1]):
        minla = binsLat[ibi]
        maxla = binsLat[ibi+1]
        tmp = df.loc[(df['latitude'] > minla) & (df['latitude'] <= maxla)]
        val = tmp[value_col].mean()
        norm = normalize_bar(val,  vmin,  vmax)
        ax_vertical.barh((minla + maxla) / 2, tmp.size, color=cmap(norm ), height=maxla - minla)
    ax_vertical.set_xlabel('Number of stations')


def plot_time_series(df:pd.DataFrame, obs_d: str, ds: xr.Dataset, ax = None, avg = False):
    station = df.ioc_code
    obs = get_obs_data(obs_d,station)
    sim = get_model_data(ds, station)
    if ax is None: fig, ax = plt.subplots(figsize=(10,5))
    label = df.Station_Name
    for sensor in obs.columns: 
        obs[sensor].interpolate().plot(ax=ax, label=str(label) + ' measured', color='k', linestyle='dashed')
    sim.plot(ax=ax, label=label+' model')
    ax.legend(loc='lower left')
    ax.set_xlim([obs.index.min(), obs.index.max()])
    # Add a light opacity text boxabs(sim_max - obs_max) / obs_max
    for col in obs.columns:
        textstr = f'Max error on the surge {np.round(abs(sim.max() - obs[col].max()/obs[col].max())*100,1)}%'
        if avg: 
            fit =  - sim.mean() + obs[col].mean()
            sim = sim + fit
            textstr += f'\nAttention: sim was fitted to obs by {np.round(fit,2)}m'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.5, 0.07, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

def plot_map_availability(ioc_detided, ax = None, bbox: shapely.box = None): 
    if ax is None: fig, ax = plt.subplots()
    plt.tight_layout()
    countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    _ = countries.plot(color='lightgrey', ax=ax, zorder=-1)
    if bbox is not None:
        # ! bbox is a shapely box or region 
        ax.set_xlim([bbox.bounds[0]-0.5, bbox.bounds[2]+0.5])
        ax.set_ylim([bbox.bounds[1]-0.5, bbox.bounds[3]+0.5])
    SEASET_CATALOG.plot.scatter(ax=ax, x='longitude', y='latitude' , s=2, c='k', label= 'All Seaset Stations')
    colors = np.array([plt.colormaps['tab10'](np.random.rand()) for i in range(len(ioc_detided))])
    ioc_detided.plot.scatter(ax=ax, x='longitude', y='latitude', s = 50,  label= 'IOC with successly detided data', c=colors)
    return colors


def plot_simple_azure(ts: pd.Timestamp, ds:xr.Dataset, ax = None, bbox: shapely.box = None,**kwargs): 
    if ax is None: 
        fig, ax = plt.subplots()
    x = ds.lon.values
    y = ds.lat.values
    t = ds.triface_nodes.values
    z = ds.elev.sel(time=ts,method='nearest').values
    im = ax.tricontourf(x,y,t,z,**kwargs)
    ax.triplot(x,y,t, linewidth = 0.25, color = 'k', alpha = 0.3)
    if bbox is not None:
        # ! bbox is a shapely box or region 
        ax.set_xlim([bbox.bounds[0]-0.5, bbox.bounds[2]+0.5])
        ax.set_ylim([bbox.bounds[1]-0.5, bbox.bounds[3]+0.5])
    ax.axis('scaled')
    return im

def plot_table_stats(df, ax=None, colors = None): 
    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 3))  # Adjust the figure size as necessary
    plt.tight_layout()
    ax.axis('tight')
    ax.axis('off')
    df_ = df[['Station_Name','BIAS or mean error',
                "RMSE","Standard deviation of residuals",
                "Correlation Coefficient", "R^2"]]
    df_ = df_.rename(columns={
            'Station_Name': 'Station', 
            'BIAS or mean error': 'Bias', 
            "RMSE": "RMS", 
            "Standard deviation of residuals": "Std Dev", 
            "Correlation Coefficient": "Corr", })
    column_labels = ['Station', 'Bias', 'RMS', 'Std Dev', 'Corr', 'R^2']
    hex = ['#%02x%02x%02x' % (int(RBG[0]), int(RBG[1]), int(RBG[2])) for RBG in colors*255 ] 
    c_=  np.repeat([hex], len(column_labels), axis = 0).T
    # Creating the table
    table = ax.table(cellText=df_.round(2).values, colLabels=column_labels, loc='center', cellColours=c_)
    return df_

def select_azure_file(t_: pd.Timestamp):
    t_rounded = t_.round(freq='12H')
    if t_rounded > t_:
        t_azure = t_rounded
    else : 
        t_azure = t_rounded - pd.Timedelta(hours=12)
    file = t_azure.strftime("%Y%m%d.%H.zarr") 
    fn = f"az://{CONTAINER}/{file}"
    return fn

def open_azure_file(fn: str, so: dict): 
    return thalassa.open_dataset(fn, engine="zarr", storage_options=so, chunks={})

def convert_and_open_zarr(fn, so, nc_file):
    logging.info("Converting stations.zarr into stations.nc, if not already there ")
    if os.path.exists(nc_file):
        ds = xr.open_dataset(nc_file)
    else : # convert to netcdf for multiprocessing
        ds_zarr = open_azure_file(fn, so)
        ds_zarr.to_netcdf(nc_file)
        ds_zarr.close()
        ds = xr.open_dataset(nc_file)
    return ds


def create_first_page(ioc_detided, gauges, area):
    """
    Create the first page of the surge report with the main map, a secondary map, and statistics.
    """
    fig = plt.figure(figsize=(21, 29.7), constrained_layout = True)  # A4 format
    outgs = GridSpec(3, 2, figure=fig)
    plt.axis('off')
    map_ax = fig.add_subplot(outgs[:2, :])  # Main map
    map_ax2 = fig.add_subplot(outgs[2, 0])  # Secondary map
    stats_ax = fig.add_subplot(outgs[2, 1])  # Statistics text

    # Plot main map
    tmean = avg_ts([get_storm_peak_time(gauges, station) for station in ioc_detided.ioc_code.values])
    fn_map = select_azure_file(tmean)
    ds = open_azure_file(fn_map, STORAGE_AZ) 
    plot_stations_map(ioc_detided, 'RMSE', 0, 1, map_ax, tmean, ds, area)

    # Plot secondary map
    c_ = plot_map_availability(ioc_detided, map_ax2, bbox = area)

    # Add statistics text
    plot_table_stats(ioc_detided, stats_ax, colors=c_)
    return fig


def create_subsequent_pages(df, obs_root, gauges, start_index):
    """
    Create subsequent pages of the surge report with time series plots.
    """
    fig = plt.figure(figsize=(21, 29.7), constrained_layout = True)  # A4 format
    outgs = GridSpec(4, 2, figure=fig)  # Adjust grid dimensions as needed
    plt.axis('off')
    if start_index + 8 < len(df): 
        df_i = df.index[start_index:start_index+8]
    else : 
        df_i = df.index[start_index:]
    for i, i_s in enumerate(df_i):
        station = df.iloc[i_s]
        ts_ax = fig.add_subplot(outgs[int(i / 2), i % 2])
        plot_time_series(station, os.path.join(obs_root,'surge'), gauges, ts_ax)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.1)
    return fig


def create_storm_surge_report(start, end, regions, storm_name, wdir):
    tmin = start.strftime("%Y-%m-%d")
    tmax = end.strftime("%Y-%m-%d")
    obs_root = os.path.join(wdir, f'obs/{tmin}_{tmax}')
    dirs = ['raw', 'clean', 'surge']

    for d in dirs:
        ensure_directory(os.path.join(obs_root, d))

    skill_file = os.path.join(wdir, f'skill_{tmin}-{tmax}_{storm_name}.csv')
    skill_results = pd.DataFrame()

    for region in regions:
        try:
            report_path = os.path.join(wdir, 'reports', f'surge_report_{storm_name}_{tmin}_{tmax}_{region}.pdf')
            with mpdf.PdfPages(report_path) as pdf:
                xmin, xmax, ymin, ymax = COASTLINES[region]
                area = shapely.box(xmin, ymin, xmax, ymax)
                ioc_raw = ioc.get_ioc_stations(region=area).rename(columns={'lon':'longitude', 'lat':'latitude'})
                clean_and_select_ioc(ioc_raw, start, end, obs_root)
                ioc_clean = ioc_subset_from_files_in_folder(ioc_raw, os.path.join(obs_root, 'clean'), ext='.csv')

                if not ioc_clean.empty:
                    gauges = xr.open_dataset(os.path.join(wdir, 'stations.nc'))
                    skill_regional = compute_surge_comparison_serial(ioc_clean, os.path.join(obs_root, 'surge'), gauges)
                    skill_results = pd.concat([skill_results, skill_regional])

                    if not skill_regional.empty:
                        ioc_detided = skill_regional.merge(SEASET_CATALOG[['ioc_code', 'longitude', 'latitude', 'Station_Name']], how='left', left_on=skill_regional.index, right_on='ioc_code')
                        Npages = int(len(ioc_detided) / 8) + 2

                        # First page
                        fig = create_first_page(ioc_detided, gauges, area)
                        pdf.savefig(fig)
                        plt.close(fig)

                        # Subsequent pages
                        for ip in range(0, Npages-1):
                            fig = create_subsequent_pages(ioc_detided, obs_root, gauges, ip * 8)
                            pdf.savefig(fig)
                            plt.close(fig)        

            skill_results.to_csv(skill_file)

        except Exception as e:
            logging.error(f"Error in processing region {region}: {e}")


def create_skill_report(wdir):
    tmin = START.strftime("%Y-%m-%d")
    tmax = NOW.strftime("%Y-%m-%d")
    obs_root = os.path.join(wdir, f'obs/{tmin}_{tmax}')
    dirs = ['raw', 'clean', 'surge']

    for d in dirs:
        ensure_directory(os.path.join(obs_root, d))

    report_path = os.path.join(wdir, 'reports', f'skill_report_{tmin}_{tmax}.pdf')
    with mpdf.PdfPages(report_path) as pdf:
        ioc_raw = SEASET_CATALOG[~SEASET_CATALOG.ioc_code.isna()]
        clean_and_select_ioc(ioc_raw, START, NOW, obs_root)
        ioc_clean = ioc_subset_from_files_in_folder(ioc_raw, os.path.join(obs_root, 'clean'), ext='.csv')

        if not ioc_clean.empty:
            gauges = xr.open_dataset(os.path.join(wdir, 'stations.nc'))
            skill_regional = compute_surge_comparison_serial(ioc_clean, os.path.join(obs_root, 'surge'), gauges)

            if not skill_regional.empty:
                ioc_detided = skill_regional.merge(SEASET_CATALOG[['ioc_code', 'longitude', 'latitude', 'Station_Name']], how='left', left_on=skill_regional.index, right_on='ioc_code')

                # Creating the figure in landscape format
                fig, map_ax = plt.subplots(figsize=(29.7, 21))  # A4 landscape
                plot_stations_map(ioc_detided, 'RMSE', 0, 1, map_ax, tmin, open_azure_file(select_azure_file(tmin), STORAGE_AZ))

                pdf.savefig(fig, orientation='portrait')
                plt.close(fig)

# 
if __name__=="__main__": 
    WORKDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    create_skill_report(WORKDIR)