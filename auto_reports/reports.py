import logging
import os

import geopandas as gpd
import matplotlib
import matplotlib.backends.backend_pdf as mpdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import thalassa
import xarray as xr
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks
from searvey import ioc

from .common import COASTLINES
from .common import PROPS
from .common import SEASET_CATALOG
from .common import WHITE_STROKE
from .data_process import avg_ts
from .data_process import clean_and_select_seaset
from .data_process import compute_surge_comparison
from .data_process import compute_surge_comparison_serial
from .data_process import ensure_directory
from .data_process import extract_from_ds
from .data_process import get_multi_provider
from .data_process import get_storm_peak_time
from .data_process import read_df
from .data_process import seaset_subset_from_files_in_folder

CM = 1 / 1.5


# HELPERS
def normalize(values, vmin, vmax):
    normalized = np.clip(values, vmin, vmax)
    return normalized


def normalize_bar(values, vmin, vmax):
    values = (values - vmin) / (vmax - vmin)
    normalized = np.clip(values, vmin, vmax)
    return normalized


def get_storm_error(
    obs: pd.DataFrame, sim: pd.DataFrame, storm_peak_time: pd.Timestamp
):
    time_window = pd.Timedelta(hours=3)
    time_filter = (obs.index >= storm_peak_time - time_window) & (
        obs.index <= storm_peak_time + time_window
    )

    # Apply time filter
    obs_window = obs[time_filter]
    sim_window = sim.loc[obs_window.index.intersection(sim.index)]

    # Find local peaks
    obs_peaks, _ = find_peaks(obs_window)
    sim_peaks, _ = find_peaks(sim_window)
    if len(obs_peaks) == 0 or len(sim_peaks) == 0:
        return np.nan  # or handle error differently
    # Find the peak closest to the storm surge time in both datasets
    obs_peak_index = obs_peaks[
        np.argmin(abs(obs_window.index[obs_peaks] - storm_peak_time))
    ]
    sim_peak_index = sim_peaks[
        np.argmin(abs(sim_window.index[sim_peaks] - storm_peak_time))
    ]

    obs_peak_value = obs_window.iloc[obs_peak_index]
    sim_peak_value = sim_window.iloc[sim_peak_index]
    error = np.round(abs((sim_peak_value - obs_peak_value) / obs_peak_value) * 100, 1)
    return error


def get_stations_in_bbox(df, bbox):
    def in_bbox(row):
        point = shapely.Point(row["longitude"], row["latitude"])
        return bbox.contains(point)

    df["in_bbox"] = df.apply(in_bbox, axis=1)
    return df[df["in_bbox"]]


# PLOTS
def plot_stations_map(
    skill,
    skill_param="RMSE",
    vmin=0,
    vmax=1,
    ax=None,
    time=None,
    ds=None,
    bbox=None,
    colors=None,
):
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(29.7 / 2, 21))
        plt.tight_layout()
        divider = make_axes_locatable(ax)
        ax1 = divider.append_axes("bottom", size="3%", pad=0.3)
        ax2 = divider.append_axes("top", size="14%", pad=0.3)
        ax3 = divider.append_axes("right", size="10%", pad=0.3)

        ax2.sharex(ax)
        ax3.sharey(ax)

        plot_wl = False
        countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        _ = countries.plot(color="lightgrey", ax=ax, zorder=-1)
        if time is not None:
            if ds is not None:
                levels = np.arange(-2, 2, 0.1)
                cmap = plt.colormaps["coolwarm"]
                im = plot_simple_azure(
                    time, ds, ax=ax, bbox=bbox, levels=levels, cmap=cmap
                )
                plt.colorbar(im, cax=ax1, orientation="horizontal").set_label(
                    "Elevation [m]"
                )
                plot_wl = True
        if bbox is not None:
            ax.set_xlim([bbox.bounds[0] - 0.5, bbox.bounds[2] + 0.5])
            ax.set_ylim([bbox.bounds[1] - 0.5, bbox.bounds[3] + 0.5])
            skill = get_stations_in_bbox(skill, bbox)

        normalized_values = normalize(skill[skill_param], vmin, vmax)
        im = ax.scatter(
            skill.longitude,
            skill.latitude,
            c=normalized_values,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            marker=".",
            s=50,
        )
        if colors is not None:
            im = ax.scatter(skill.longitude, skill.latitude, c=colors, marker=".", s=50)

        if plot_wl:
            ax4 = divider.append_axes("right", size="2%", pad=0.3)
            plt.colorbar(im, cax=ax4).set_label(skill_param)
        else:
            plt.colorbar(im, cax=ax1, orientation="horizontal").set_label(skill_param)

        plt.suptitle(
            f"Geographical distribution of {skill_param} differences for {len(skill)} tide gauges"
        )
        plot_side_histograms(skill, ax2, ax3, skill_param, vmin, vmax)
        plt.savefig("stations_map_" + skill_param + ".png")

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
    cmap = matplotlib.colormaps["jet"]

    # Histogram for longitude
    for ibi, _ in enumerate(binsLon[:-1]):
        minlo = binsLon[ibi]
        maxlo = binsLon[ibi + 1]
        tmp = df.loc[(df["longitude"] > minlo) & (df["longitude"] <= maxlo)]
        val = tmp[value_col].mean()
        norm = normalize_bar(val, vmin, vmax)
        ax_horizontal.bar(
            (minlo + maxlo) / 2, tmp.size, color=cmap(norm), width=maxlo - minlo
        )
    ax_horizontal.set_ylabel("Number of stations")

    # Histogram for latitude
    for ibi, _ in enumerate(binsLat[:-1]):
        minla = binsLat[ibi]
        maxla = binsLat[ibi + 1]
        tmp = df.loc[(df["latitude"] > minla) & (df["latitude"] <= maxla)]
        val = tmp[value_col].mean()
        norm = normalize_bar(val, vmin, vmax)
        ax_vertical.barh(
            (minla + maxla) / 2, tmp.size, color=cmap(norm), height=maxla - minla
        )
    ax_vertical.set_xlabel("Number of stations")


def plot_time_series(df: pd.DataFrame, obs_root: str, ax=None, avg=False):
    station = df.ioc_code
    obs = read_df(os.path.join(obs_root, "surge", station + ".csv"))
    sim = read_df(os.path.join(obs_root, "model", station + ".csv"))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    label = df.Station_Name
    for sensor in obs.columns:
        obs[sensor].reindex(
            index=pd.date_range(obs.index.min(), obs.index.max(), freq="10min")
        ).plot(ax=ax, label=str(label) + " measured", color="k", style=".")
    ax.set_xlim([obs.index.min(), obs.index.max()])
    if not sim.empty and len(sim) > 0:
        sim.reindex(
            index=pd.date_range(obs.index.min(), obs.index.max(), freq="10min")
        ).interpolate().plot(ax=ax, label=str(label) + " model")
        for col in obs.columns:
            peak = sim.idxmax()  #
            try:
                error = get_storm_error(obs[col], sim, peak)
            except Exception as e:
                print(
                    "failed computing storm surge error",
                    e,
                    station,
                    peak,
                    obs[col].index,
                    sim.index,
                )
            error = np.nan
            textstr = f"error at peak surge {error}%"
            if avg:
                fit = -sim.mean() + obs[col].mean()
                sim = sim + fit
                textstr += f"\nAttention: sim was fitted to obs by {np.round(fit,2)}m"
        ax.text(
            0.5,
            0.07,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=PROPS,
        )
    else:
        textstr = "No simulation data"
        ax.text(
            0.5,
            0.07,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=PROPS,
        )
    ax.legend(loc="lower left")


def plot_map_availability(seaset_detided, ax=None, bbox: shapely.box = None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.tight_layout()
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    _ = countries.plot(color="lightgrey", ax=ax, zorder=-1)
    if bbox is not None:
        # ! bbox is a shapely box or region
        ax.set_xlim([bbox.bounds[0] - 0.5, bbox.bounds[2] + 0.5])
        ax.set_ylim([bbox.bounds[1] - 0.5, bbox.bounds[3] + 0.5])
    SEASET_CATALOG.plot.scatter(
        ax=ax, x="longitude", y="latitude", s=2, c="k", label="All Seaset Stations"
    )
    colors = np.array(
        [plt.colormaps["tab10"](np.random.rand()) for i in range(len(seaset_detided))]
    )
    seaset_detided.plot.scatter(
        ax=ax,
        x="longitude",
        y="latitude",
        s=50,
        label="IOC with detided data",
        c=colors,
    )
    for idx, row in seaset_detided.iterrows():
        ax.text(
            row.longitude,
            row.latitude,
            str(row["ioc_code"]),
            fontsize=8,
            ha="right",
            va="bottom",
            path_effects=WHITE_STROKE,
        )

    return colors


def plot_simple_azure(
    ts: pd.Timestamp, ds: xr.Dataset, ax=None, bbox: shapely.box = None, **kwargs
):
    if ax is None:
        fig, ax = plt.subplots()
    x = ds.lon.values
    y = ds.lat.values
    t = ds.triface_nodes.values
    z = ds.elev.sel(time=ts, method="nearest").values
    im = ax.tricontourf(x, y, t, z, **kwargs)
    ax.triplot(x, y, t, linewidth=0.25, color="k", alpha=0.3)
    if bbox is not None:
        # ! bbox is a shapely box or region
        ax.set_xlim([bbox.bounds[0] - 0.5, bbox.bounds[2] + 0.5])
        ax.set_ylim([bbox.bounds[1] - 0.5, bbox.bounds[3] + 0.5])
    ax.axis("scaled")
    return im


def wrap_text(text, width):
    """Wrap text to a specified width."""
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def plot_table_stats(df, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))  # Adjust the figure size as necessary
    plt.tight_layout()
    ax.axis("tight")
    ax.axis("off")
    df_ = df[
        [
            "ioc_code",
            "Station_Name",
            "BIAS or mean error",
            "RMSE",
            "Standard deviation of residuals",
            "Correlation Coefficient",
            "R^2",
        ]
    ]
    df_ = df_.rename(
        columns={
            "Station_Name": "Station",
            "BIAS or mean error": "Bias",
            "RMSE": "RMS",
            "Standard deviation of residuals": "Std Dev",
            "Correlation Coefficient": "Corr",
        }
    )
    column_labels = ["ioc_code", "Station", "Bias", "RMS", "Std Dev", "Corr", "R^2"]
    df_[["Bias", "RMS", "Std Dev"]] = df_[["Bias", "RMS", "Std Dev"]].round(2)

    # Wrap text for each cell
    df_ = df_.applymap(lambda x: wrap_text(str(x), width=8))  # Adjust 'width' as needed

    hex = [
        "#%02x%02x%02x" % (int(RBG[0]), int(RBG[1]), int(RBG[2]))
        for RBG in colors * 255
    ]
    c_ = np.repeat([hex], len(column_labels), axis=0).T
    # Creating the table
    table = ax.table(
        cellText=df_.values, colLabels=column_labels, loc="center", cellColours=c_
    )
    table.set_fontsize(12)  # Set a larger font size
    table.auto_set_column_width(
        col=list(range(len(column_labels)))
    )  # Adjust the column widths
    # Manually adjust the width of the 'Station' column (assuming it's the first column)
    for key, cell in table.get_celld().items():
        if key[1] == 1:  # Column 1 is 'Station'
            cell.set_width(40)  # Set the width as needed (e.g., 0.2)

    return df_


def open_azure_file(fn: str, so: dict):
    return thalassa.open_dataset(fn, engine="zarr", storage_options=so, chunks={})


def convert_and_open_zarr(fn, so, nc_file):
    logging.info("Converting stations.zarr into stations.nc, if not already there ")
    if os.path.exists(nc_file):
        ds = xr.open_dataset(nc_file)
    else:  # convert to netcdf for multiprocessing
        ds_zarr = open_azure_file(fn, so)
        ds_zarr.to_netcdf(nc_file)
        ds_zarr.close()
        ds = xr.open_dataset(nc_file)
    return ds


def create_first_page(seaset_detided, ds, area):
    """
    Create the first page of the surge report with the main map, a secondary map, and statistics.
    """
    fig = plt.figure(figsize=(21 * CM, 29.7 * CM), constrained_layout=True)  # A4 format
    outgs = GridSpec(3, 2, figure=fig)
    plt.axis("off")
    map_ax = fig.add_subplot(outgs[:2, :])  # Main map
    map_ax2 = fig.add_subplot(outgs[2, 0])  # Secondary map
    stats_ax = fig.add_subplot(outgs[2, 1])  # Statistics text

    # Plot main map
    tmean = avg_ts(
        [
            get_storm_peak_time(ds, station)
            for station in seaset_detided.seaset_id.values
        ]
    )
    plot_stations_map(seaset_detided, "RMSE", 0, 1, map_ax, tmean, ds, area)

    # Plot secondary map
    c_ = plot_map_availability(seaset_detided, map_ax2, bbox=area)

    # Add statistics text
    plot_table_stats(seaset_detided, stats_ax, colors=c_)
    return fig


def create_subsequent_pages(
    df: pd.DataFrame, obs_root: str, gauges: xr.Dataset, start_index: int
):
    """
    Create subsequent pages of the surge report with time series plots.
    """
    fig = plt.figure(figsize=(21 * CM, 29.7 * CM), constrained_layout=True)  # A4 format
    outgs = GridSpec(4, 2, figure=fig)  # Adjust grid dimensions as needed
    plt.axis("off")
    if start_index + 8 < len(df):
        df_i = df.index[start_index : start_index + 8]
    else:
        df_i = df.index[start_index:]
    for i, i_s in enumerate(df_i):
        station = df.iloc[i]
        ts_ax = fig.add_subplot(outgs[int(i / 2), i % 2])
        plot_time_series(station, obs_root, ts_ax)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.1)
    return fig


def create_storm_surge_report(start, end, regions, storm_name, wdir):
    tmin = start.strftime("%Y-%m-%d")
    tmax = end.strftime("%Y-%m-%d")
    obs_root = os.path.join(wdir, f"obs/{tmin}_{tmax}")
    dirs = ["raw", "clean", "surge"]

    for d in dirs:
        ensure_directory(os.path.join(obs_root, d))

    skill_file = os.path.join(wdir, f"skill_{tmin}-{tmax}_{storm_name}.csv")
    skill_results = pd.DataFrame()

    for region in regions:
        try:
            report_path = os.path.join(
                wdir, "reports", f"surge_report_{storm_name}_{tmin}_{tmax}_{region}.pdf"
            )
            with mpdf.PdfPages(report_path) as pdf:
                xmin, xmax, ymin, ymax = COASTLINES[region]
                area = shapely.box(xmin, ymin, xmax, ymax)
                ioc_raw = ioc.get_ioc_stations(region=area).rename(
                    columns={"lon": "longitude", "lat": "latitude"}
                )
                clean_and_select_seaset(ioc_raw, start, end, obs_root)
                ioc_clean = seaset_subset_from_files_in_folder(
                    ioc_raw, os.path.join(obs_root, "clean"), ext=".csv"
                )

                if not ioc_clean.empty:
                    gauges = xr.open_dataset(os.path.join(wdir, "stations.nc"))
                    skill_regional = compute_surge_comparison_serial(
                        ioc_clean, obs_root, gauges, "id", "elev_sim", "IOC-"
                    )
                    skill_results = pd.concat([skill_results, skill_regional])

                    if not skill_regional.empty:
                        seaset_detided = skill_regional.merge(
                            SEASET_CATALOG[
                                ["ioc_code", "longitude", "latitude", "Station_Name"]
                            ],
                            how="left",
                            left_on=skill_regional.index,
                            right_on="ioc_code",
                        )
                        Npages = int(len(seaset_detided) / 8) + 2

                        # # First page
                        fig = create_first_page(seaset_detided, gauges, area)
                        pdf.savefig(fig)
                        plt.close(fig)

                        # Subsequent pages
                        for ip in range(0, Npages - 1):
                            fig = create_subsequent_pages(
                                seaset_detided, obs_root, gauges, ip * 8
                            )
                            pdf.savefig(fig)
                            plt.close(fig)

            skill_results.to_csv(skill_file)

        except Exception as e:
            logging.error(f"Error in processing region {region}: {e}")


def create_skill_report(ds: xr.Dataset, wdir: str = None):
    if wdir is None:
        wdir = os.path.join(os.getcwd())
    start = pd.Timestamp(ds.time.min().values)
    end = pd.Timestamp(ds.time.max().values)
    tmin = start.strftime("%Y-%m-%d")
    tmax = end.strftime("%Y-%m-%d")

    obs_root = os.path.join(wdir, f"obs/{tmin}_{tmax}")
    dirs = ["raw", "clean", "model", "surge"]

    for d in dirs:
        ensure_directory(os.path.join(obs_root, d))

    report_path = os.path.join(wdir, "reports", f"skill_report_{tmin}_{tmax}.pdf")
    with mpdf.PdfPages(report_path) as pdf:
        # 0 download the stations
        seaset_avail = get_multi_provider(
            SEASET_CATALOG, start, end, obs_root + "/raw", ext=".parquet"
        )
        # # 1 clean the stations
        clean_and_select_seaset(seaset_avail, obs_root, ext=".parquet", t_rsp=60)
        seaset_clean = seaset_subset_from_files_in_folder(
            seaset_avail, os.path.join(obs_root, "clean"), ext=".parquet"
        )
        if len(seaset_clean) > 0:
            seaset_model = extract_from_ds(
                seaset_clean,
                obs_root + "/model",
                ds,
                "seaset_id",
                ext=".parquet",
                t_rsp=60,
            )
            skill_regional = compute_surge_comparison(
                seaset_model, obs_root, ext="parquet"
            )

            skill_file = os.path.join(wdir, f"skill_{tmin}-{tmax}.csv")
            # skill_regional.to_csv(skill_file)
            skill_regional = pd.read_csv(skill_file, index_col=0)
            if not skill_regional.empty:
                seaset_detided = skill_regional.merge(
                    SEASET_CATALOG[
                        ["ioc_code", "longitude", "latitude", "Station_Name"]
                    ],
                    how="left",
                    left_on=skill_regional.index,
                    right_on="ioc_code",
                )

                # Creating the figure in landscape format
                for region in [
                    None,
                    "Europe",
                    "East Coast US + Gulf of Mexico",
                    "Oceania",
                    "West Coast US",
                ]:
                    if region is None:
                        xmin, xmax, ymin, ymax = -180, 180, -90, 90
                    else:
                        xmin, xmax, ymin, ymax = COASTLINES[region]
                    area = shapely.box(xmin, ymin, xmax, ymax)
                    for param in [
                        "RMSE",
                        "Correlation Coefficient",
                        "R^2",
                        "BIAS or mean error",
                    ]:
                        fig, map_ax = plt.subplots(
                            figsize=(29.7 * CM, 21 * CM)
                        )  # A4 landscape
                        vmax = 1
                        if param == "Correlation Coefficient":
                            vmin = -1
                        else:
                            vmin = 0
                        plot_stations_map(
                            seaset_detided, param, vmin, vmax, map_ax, tmin, bbox=area
                        )
                        pdf.savefig(fig, orientation="portrait")

                # ADD 20 worst / 20 best
                best_rmse = seaset_detided.loc[
                    seaset_detided["RMSE"].abs().nsmallest(8).index
                ]
                worst_rmse = seaset_detided.loc[
                    seaset_detided["RMSE"].abs().nlargest(8).index
                ]
                best_bias = seaset_detided.loc[
                    seaset_detided["BIAS or mean error"].abs().nsmallest(8).index
                ]
                worst_bias = seaset_detided.loc[
                    seaset_detided["BIAS or mean error"].abs().nlargest(8).index
                ]
                best_r2 = seaset_detided.nlargest(8, "R^2")
                worst_r2 = seaset_detided.nsmallest(8, "R^2")

                for df in [
                    best_rmse,
                    worst_rmse,
                    best_bias,
                    worst_bias,
                    best_r2,
                    worst_r2,
                ]:
                    fig = create_subsequent_pages(df, obs_root, ds, 0)
                    pdf.savefig(fig)
                plt.close(fig)
