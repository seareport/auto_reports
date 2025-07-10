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
from auto_reports._io import get_obs_dir
from auto_reports._io import load_data
from auto_reports._storms import STORMS

OPTS = dict(
    show_grid=True,
    active_tools=["box_zoom"],
    line_width=1.5,
    line_alpha=0.8,
    width=800,
    height=300,
    # legend_position = "right"
)


transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)


def plot_ts(models_all, all_stats, region, cmap, data_dir):
    obs_dir = get_obs_dir(data_dir)
    extremes = {
        model: assign_storms(all_stats[model][1], region) for model in models_all
    }
    storms = extremes[models_all[0]].storm.unique()

    region_storms = {s: STORMS[region][s] for s in storms}
    options = {
        f"{dates[0]} - {storm_name}": storm_name
        for storm_name, dates in region_storms.items()
    }
    sorted_options = {k: options[k] for k in sorted(options.keys())}

    model_selector = pn.widgets.CrossSelector(
        name="Models",
        value=models_all,
        options=models_all,
        **rr.model_cross_selector,
    )

    storm_selector = pn.widgets.Select(
        name="Storms",
        value=storms[-1] if len(sorted_options) > 0 else None,
        options=sorted_options,
    )

    demean_checkbox = pn.widgets.Checkbox(name="Demean Obs", value=True)

    selector_panel = pn.Column(
        pn.pane.Markdown(f"### Select from {len(models_all)} models"),
        model_selector,
        pn.pane.Markdown(f"### Select from {len(sorted_options)} storms"),
        storm_selector,
    )

    toggle_btn = pn.widgets.Button(name="Hide", button_type="primary")

    def toggle_visibility(event):
        selector_panel.visible = not selector_panel.visible
        toggle_btn.name = "Show" if not selector_panel.visible else "Hide"

    toggle_btn.on_click(toggle_visibility)

    @pn.depends(
        storm_selector.param.value,
        model_selector.param.value,
        demean_checkbox.param.value,
    )
    def ts_panel(storm, models, demean):
        if (storm) and (len(models) > 0):
            df = extremes[models[0]]
            min_time = pd.Timestamp(min(STORMS[region][storm])) - pd.Timedelta(days=4)
            max_time = pd.Timestamp(max(STORMS[region][storm])) + pd.Timedelta(days=4)
            min_time = max(pd.Timestamp(2022, 1, 1), min_time)
            max_time = min(pd.Timestamp(2024, 12, 31, 23), max_time)
            stations_impacted = df[df.storm == storm].station.values
            if len(stations_impacted) > 15:
                storm_peaks = extremes[models[0]]
                subset = storm_peaks[
                    storm_peaks.station.isin(stations_impacted)
                    & (storm_peaks.storm == storm)
                ].iloc[:15]
                stations_impacted = subset.station.unique()
            timeseries = []
            for station in stations_impacted:
                obs_file = glob.glob(f"{obs_dir}/{station}*.parquet")[0]
                obs = load_data(obs_file)
                obs = obs.loc[min_time:max_time]
                if demean:
                    mean = obs.mean()
                    obs = obs - mean
                sims = {
                    model: load_data(
                        data_dir / f"models/{model}/{station}.parquet",
                    ).loc[min_time:max_time]
                    for model in models
                }
                # plots
                storm_peaks = extremes[models[0]]
                storm_peak = storm_peaks[
                    (storm_peaks.station == station) & (storm_peaks.storm == storm)
                ]
                if demean:
                    storm_peak.loc[:, "observed"] -= mean

                curve = hv.Curve(obs, label="obs").opts(
                    color="lightgrey",
                    **OPTS,
                    title=station,
                ) * storm_peak.hvplot.scatter(
                    x="time observed",
                    y="observed",
                    s=50,
                    c="lightgrey",
                    label="obs",
                )
                for i, (model, ts) in enumerate(sims.items()):
                    storm_peaks = extremes[model]
                    storm_peak = storm_peaks[
                        (storm_peaks.station == station) & (storm_peaks.storm == storm)
                    ]
                    curve *= hv.Curve(ts, label=f"{model}").opts(
                        color=cc.glasbey_dark[i],
                        **OPTS,
                    ) * storm_peak.hvplot.scatter(
                        x="time model",
                        y="model",
                        s=50,
                        c=cc.glasbey_dark[i],
                        label=f"{model}",
                    )

                timeseries.append(curve)
            layout = (
                hv.Layout(timeseries)
                .cols(1)
                .opts(
                    title=storm,
                    shared_axes=False,
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
        df = extremes[models_all[0]]
        stats = all_stats[models_all[0]][0]
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

    return pn.Row(
        pn.Column(
            pn.Row(toggle_btn, demean_checkbox),
            selector_panel,
        ),
        pn.panel(ts_panel),
        pn.panel(ts_map_hv),
    )
