from __future__ import annotations

import glob

import colorcet as cc
import geoviews as gv
import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
from pyproj import Transformer

import auto_reports._render as rr
from auto_reports._io import assign_storms
from auto_reports._io import DATA_DIR
from auto_reports._io import load_data
from auto_reports._io import OBS_DIR
from auto_reports._storms import STORMS

OPTS = dict(
    show_grid=True,
    active_tools=["box_zoom"],
    line_width=1,
    line_alpha=0.8,
    width=800,
    height=300,
    # legend_position = "right"
)


transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)


def plot_ts(models, all_stats, region, cmap):
    extremes = {model: assign_storms(all_stats[model][1], region) for model in models}
    storms = extremes[models[0]].storm.unique()
    storm_selector = pn.widgets.Select(
        name="Storms",
        value=storms[-1] if len(storms) > 0 else None,
        options=sorted(storms.tolist()),
    )

    @pn.depends(storm_selector.param.value)
    def ts_panel(storm):
        df = extremes[models[0]]
        if storm:
            min_time = pd.Timestamp(min(STORMS[region][storm])) - pd.Timedelta(days=10)
            max_time = pd.Timestamp(max(STORMS[region][storm])) + pd.Timedelta(days=10)
            stations_impacted = df[df.storm == storm].station.values
            timeseries = []
            for station in stations_impacted:
                obs_file = glob.glob(f"{OBS_DIR}/{station}*.parquet")[0]
                obs = load_data(obs_file)
                obs = obs.loc[min_time:max_time]
                sims = {
                    model: load_data(
                        DATA_DIR / f"models/{model}/{station}.parquet",
                    ).loc[min_time:max_time]
                    for model in models
                }
                # plots
                curve = hv.Curve(obs, label="obs").opts(
                    color="lightgrey",
                    **OPTS,
                    title=station,
                )
                for i, (model, ts) in enumerate(sims.items()):
                    curve *= hv.Curve(ts, label=f"{model}").opts(
                        color=cc.glasbey_dark[i],
                        **OPTS,
                    )
                    storm_peaks = extremes[model]
                    storm_peak = storm_peaks[
                        (storm_peaks.station == station) & (storm_peaks.storm == storm)
                    ]
                    curve *= storm_peak.hvplot.scatter(
                        x="time observed",
                        y="observed",
                        s=100,
                        c="lightgrey",
                    )
                    curve *= storm_peak.hvplot.scatter(
                        x="time model",
                        y="model",
                        s=100,
                        c=cc.glasbey_dark[i],
                    )
                timeseries.append(curve)
            layout = (
                hv.Layout(timeseries)
                .cols(1)
                .opts(
                    title=storm,
                )
            )
            return layout
        else:
            return (
                hv.Layout([hv.Curve((0, 0))])
                .cols(1)
                .opts(
                    title="No storm selected",
                )
            )

    @pn.depends(storm_selector.param.value)
    def ts_map_hv(storm):
        df = extremes[models[0]]
        stats = all_stats[models[0]][0]
        if df.empty:
            return gv.Points((0, 0)) * gv.Points((0, 0))
        points = stats.hvplot.points(
            x="lon",
            y="lat",
            geo=True,
            tiles="OSM",
            hover_cols=["station"],
            c="r",
            line_color="k",
            size=20,
        )
        map_hv = df[df.storm == storm].hvplot.points(
            x="lon",
            y="lat",
            geo=True,
            tiles="OSM",
            c=cmap[region],
            hover_cols=["station"],
            s=100,
            line_color="k",
        )
        xmin, xmax = (
            df[df.storm == storm].lon.min() - 5,
            df[df.storm == storm].lon.max() + 5,
        )
        ymin, ymax = (
            df[df.storm == storm].lat.min() - 5,
            df[df.storm == storm].lat.max() + 5,
        )
        x0, y0 = transformer.transform(xmin, ymin)
        x1, y1 = transformer.transform(xmax, ymax)
        return (points * map_hv).opts(xlim=(x0, x1), ylim=(y0, y1), **rr.map_region)

    return pn.Row(storm_selector, ts_panel, ts_map_hv)
