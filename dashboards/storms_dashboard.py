from __future__ import annotations

import glob
import logging

import colorcet as cc
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

import auto_reports._render as rr
from auto_reports._io import assign_storms
from auto_reports._io import get_data_dir
from auto_reports._io import get_obs_dir
from auto_reports._io import load_data
from auto_reports._io import load_world_oceans
from auto_reports._io import OVERRIDE_CSS
from auto_reports._io import update_color_map
from auto_reports._stats import get_stats
from auto_reports._storms import STORMS
from auto_reports.graphs.maps import storm_map
from auto_reports.graphs.ts import generate_storm_ts

# Force visibility everywhere
pn.extension("mathjax", "markdown", raw_css=[OVERRIDE_CSS])

logger = logging.getLogger(name="auto-report")
QUANTILE = 0.98
colouring = "ocean"  # can be "name" for Maritime sectors


class StormDashboard(param.Parameterized):
    current_region = param.String(default="Atlantic NE")

    def __init__(self, data_dir="data", **params):
        super().__init__(**params)
        self.data_dir = get_data_dir(data_dir)
        self.obs_dir = get_obs_dir(data_dir)
        self.all_stats = get_stats(self.data_dir)
        self.models = sorted(self.all_stats.keys())
        self.extremes = {}
        self.storms = {}
        self.sorted_storms = {}
        self.regional_stats = {}
        self.model = self.models[0]
        self.cmap = update_color_map(load_world_oceans(), colouring)
        self.region_select_options = load_world_oceans()["ocean"].unique().tolist()
        self.dashboard = pn.template.MaterialTemplate()
        self.region_dropdown = pn.widgets.Select(
            name="Select Region",
            options=sorted(self.region_select_options),
            value="Atlantic NE",
        )
        self.region_dropdown.param.watch(self.update_region_tab, "value")

    def update_region_tab(self, event):
        self.current_region = event.new

    def get_extremes(self):
        self.extremes = {
            model: assign_storms(self.all_stats[model][1], self.current_region)
            for model in self.models
        }

    def sort_storms(self):
        self.storms = self.extremes[self.models[0]].storm.unique()
        region_storms = {s: STORMS[self.current_region][s] for s in self.storms}
        options = {
            f"{dates[0]} - {storm_name}": storm_name
            for storm_name, dates in region_storms.items()
        }
        self.sorted_storms = {k: options[k] for k in sorted(options.keys())}

    def get_regional_stats(self):
        self.regional_stats = {
            model: (
                self.all_stats[model][0][
                    self.all_stats[model][0].ocean == self.current_region
                ],
                self.all_stats[model][1][
                    self.all_stats[model][1].ocean == self.current_region
                ],
            )
            for model in self.models
        }

    @param.depends("current_region", watch=False)
    def plot_regional_storm(self):
        self.get_extremes()
        self.sort_storms()
        self.get_regional_stats()
        model_selector = pn.widgets.CrossSelector(
            name="Models",
            value=self.models,
            options=self.models,
            **rr.model_cross_selector,
        )

        storm_selector = pn.widgets.Select(
            name="Storms",
            value=list(self.sorted_storms.keys())[0]
            if len(self.sorted_storms) > 0
            else None,
            options=self.sorted_storms,
        )

        demean_checkbox = pn.widgets.Checkbox(name="Correct local bias", value=True)

        selector_panel = pn.Column(
            pn.pane.Markdown(f"### Select from {len(self.models)} models"),
            model_selector,
            pn.pane.Markdown(f"### Select from {len(self.sorted_storms)} storms"),
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
        def storm_map_ts(storm, models, demean):
            def update_station(df, index):
                if index:
                    station = df.iloc[index].station.values[0]
                    min_time = pd.Timestamp(
                        min(STORMS[self.current_region][storm]),
                    ) - pd.Timedelta(days=4)
                    max_time = pd.Timestamp(
                        max(STORMS[self.current_region][storm]),
                    ) + pd.Timedelta(days=4)
                    min_time = max(pd.Timestamp(2022, 1, 1), min_time)
                    max_time = min(pd.Timestamp(2024, 12, 31, 23), max_time)
                    sims = {
                        model: load_data(
                            self.data_dir / f"models/{model}/{station}.parquet",
                        ).loc[min_time:max_time]
                        for model in models
                        if (
                            self.data_dir / f"models/{model}/{station}.parquet"
                        ).exists()
                    }
                    obs_file = glob.glob(f"{self.obs_dir}/{station}*.parquet")[0]
                    obs = load_data(obs_file)
                    obs = obs.loc[min_time:max_time]
                    ts_plot = generate_storm_ts(
                        sims,
                        obs,
                        self.extremes,
                        station,
                        storm,
                        demean,
                    )
                else:
                    ts_plot = hv.Curve((0, 0)).opts(title="No station selected")
                ts_panel[:] = [ts_plot]

            ts_panel = pn.Column()
            df = self.extremes[self.models[0]]
            df = df[df.ocean == self.current_region]
            df = df[df.storm == storm]
            storm_map_ = storm_map(df, self.cmap, self.current_region)
            selection = hv.streams.Selection1D(source=storm_map_)
            selection.param.watch(lambda e: update_station(df, e.new), "index")
            return pn.Row(storm_map_, ts_panel)

        @pn.depends(
            storm_selector.param.value,
            model_selector.param.value,
        )
        def scatter_storm_ext(storm, models):
            if (storm) and (len(models) > 0):
                df = self.extremes[models[0]]
                df = df[df.ocean == self.current_region]
                df["is_active"] = df["storm"] == storm
                df["alpha"] = df["is_active"].map({True: 1.0, False: 0.4})
                df["size"] = df["is_active"].map({True: 50, False: 30})
                main_plot = df.hvplot.scatter(
                    x="time observed",
                    y="observed",
                    c="storm",
                    size="size",
                    alpha="alpha",
                    hover_cols=["station"],
                )
                return main_plot.opts(**rr.scatter_ext)
            else:
                return hv.Scatter((0, 0)).opts(
                    title="No storm selected",
                    **rr.scatter_ext,
                )

        @pn.depends(
            storm_selector.param.value,
            model_selector.param.value,
        )
        def scatter_model_ext(storm, models):
            xy_axis = hv.Curve([(0, 0), (10, 10)]).opts(
                color="k",
                line_dash="dashed",
            )
            scatters = [xy_axis]
            xmax = []
            ymax = []
            if (storm) and (len(models) > 0):
                for i, model in enumerate(models):
                    df = self.extremes[model]
                    df = df[df.ocean == self.current_region]
                    df = df[df["storm"] == storm]
                    xmax.append(df["observed"].max())
                    ymax.append(df["model"].max())
                    scatters.append(
                        df.hvplot.scatter(
                            x="observed",
                            y="model",
                            s=100,
                            hover_cols="station",
                            c=cc.glasbey_dark[i],
                            label=model,
                        ),
                    )
                xymax = np.nanmax([*xmax, *ymax])
                xymax *= 1.1
                title = f"Storm - {storm}"
            else:
                scatters.append(hv.Scatter((0, 0)))
                title = "No storm selected"
                xymax = 1
            return hv.Overlay(scatters).opts(
                title=title,
                **rr.scatter_ext,
                xlim=(0, xymax),
                ylim=(0, xymax),
            )

        return pn.Row(
            pn.Column(
                pn.Row(toggle_btn, demean_checkbox),
                selector_panel,
            ),
            pn.Column(
                storm_map_ts,
                pn.Row(scatter_storm_ext, scatter_model_ext),
            ),
        )

    def create_dashboard(self):
        self.dashboard = pn.template.MaterialTemplate(
            title="Storm Events",
            header_background="#007BFF",
            main=pn.Column(
                pn.Row(
                    self.region_dropdown,
                ),
                self.plot_regional_storm,
            ),
        )
