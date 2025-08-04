from __future__ import annotations

import glob
import logging
from datetime import date
from datetime import timedelta

import holoviews as hv
import pandas as pd
import panel as pn
import param

import auto_reports._markdown as md
import auto_reports._render as rr
from auto_reports._io import get_data_dir
from auto_reports._io import get_models_dir
from auto_reports._io import get_obs_dir
from auto_reports._io import load_data
from auto_reports._stats import get_stats
from auto_reports._tide import compute_rss
from auto_reports._tide import compute_score
from auto_reports.graphs.maps import tide_map
from auto_reports.graphs.progress import progress_wheel
from auto_reports.graphs.tidal_plots import empty_plot_comparative_amplitudes
from auto_reports.graphs.tidal_plots import empty_plot_relative_amplitudes
from auto_reports.graphs.tidal_plots import FULL
from auto_reports.graphs.tidal_plots import plot_comparative_amplitudes
from auto_reports.graphs.tidal_plots import plot_relative_amplitudes
from auto_reports.graphs.tidal_plots import SHORT
from auto_reports.graphs.ts import empty_time_series_plot
from auto_reports.graphs.ts import generate_tide_ts

# Force visibility everywhere
pn.extension("mathjax", "markdown")

logger = logging.getLogger(name="auto-report")

CMAP = {
    "sim": "#0900bb",
    "obs": "#5D5D5DFF",
}

SHORT_LIST = SHORT
FULL_LIST = FULL

DEFAULT_START = date(2024, 6, 1)
DEFAULT_END = DEFAULT_START + timedelta(days=180)

date_picker = pn.widgets.DatePicker(
    name="Date Start",
    start=date(2022, 1, 1),
    end=date(2024, 6, 1),
    value=DEFAULT_START,
)

TIDE_ANALYSIS = "utide"  # or "pytides"


class TidalDashboard(param.Parameterized):
    def __init__(self, data_dir="data", **params):
        super().__init__(**params)
        self.data_dir = get_data_dir(data_dir)
        self.all_stats = get_stats(self.data_dir, FULL_LIST, TIDE_ANALYSIS)
        self.models = sorted(self.all_stats.keys())
        self.tide_stats = {model: self.all_stats[model][2] for model in self.models}
        self.model = self.models[0]
        self.dashboard = pn.template.MaterialTemplate()

    def update_tidal_df(self, model):
        self.model = model
        self.tidal_df = self.tide_stats[model]
        self.tidal_reduced = (
            self.tide_stats[model].reset_index().drop_duplicates("station")
        )

    def plot_tidal_graphs(self):
        model_selector = pn.widgets.Select(
            name="Models",
            value=self.model,
            options=self.models,
        )
        metric_selector = pn.widgets.Select(
            name="Metric",
            value="rss",
            options=["rss", "corr", "score"],
        )

        @pn.depends(model_selector.param.value, metric_selector.param.value)
        def tide_panel(model, metric):
            models_dir = get_models_dir(self.data_dir) / model
            obs_dir = get_obs_dir(self.data_dir)
            self.update_tidal_df(model)
            tide_map_ = tide_map(self.tidal_reduced, metric).opts(**rr.tide_map)
            station_panel = pn.Column()
            # Define the progress wheel placeholder and dynamic content
            score_wheel_pane = pn.pane.Bokeh(
                sizing_mode="fixed",
                **rr.progress,
                styles=rr.histo_offset,
            )
            score_md_pane = pn.pane.Markdown(md.SCORE_MD)
            ts_panel = pn.Column()
            selected_station = {"value": None}
            corr_sim = {"value": 0}

            def update_ts_plot(station, start):
                end = start + pd.Timedelta(days=180)
                if station is None:
                    ts_plot = empty_time_series_plot(start, end)
                else:
                    sim = load_data(models_dir / f"{station}.parquet")
                    obs_file = glob.glob(f"{str(obs_dir)}/{station}_*.parquet")[0]
                    obs = load_data(obs_file)
                    ts_plot, corr = generate_tide_ts(obs, sim, start, end, cmap=CMAP)
                    corr_sim["value"] = corr
                ts_panel[:] = [ts_plot]

            def update_station(index):
                if index:
                    # histogram part
                    station = self.tidal_reduced.iloc[index].station.values[0]
                    selected_station["value"] = station  # track current station
                    df = self.tidal_df[self.tidal_df.station == station]
                    rss_sim = compute_rss(df, "amplitude", "sim", "obs")
                    plot_relative = plot_relative_amplitudes(
                        df,
                        "amplitude",
                        SHORT_LIST,
                    )
                    plot_comparison_amp = plot_comparative_amplitudes(
                        df,
                        "amplitude",
                        rss_sim,
                        CMAP,
                        SHORT_LIST,
                    )
                    plot_comparison_phase = plot_comparative_amplitudes(
                        df,
                        "phase",
                        0,
                        CMAP,
                        SHORT_LIST,
                    )

                    update_ts_plot(station, date_picker.value)

                    score_sim = compute_score(corr_sim["value"], float(rss_sim))
                else:
                    score_sim = 0
                    plot_relative = empty_plot_relative_amplitudes("amplitude")
                    plot_comparison_amp = empty_plot_comparative_amplitudes(
                        "amplitude",
                        CMAP,
                    )
                    plot_comparison_phase = empty_plot_comparative_amplitudes(
                        "phase",
                        CMAP,
                    )
                    update_ts_plot(None, date_picker.value)

                score_wheel_pane.object = progress_wheel(score_sim, CMAP)
                station_panel[:] = [
                    pn.Row(plot_relative, plot_comparison_amp, plot_comparison_phase),
                    ts_panel,
                ]

            selection = hv.streams.Selection1D(source=tide_map_)
            selection.param.watch(lambda e: update_station(e.new), "index")

            def on_date_change(event):
                update_ts_plot(selected_station["value"], event.new)

            date_picker.param.watch(on_date_change, "value")

            update_station([])

            return pn.Row(
                pn.Column(
                    tide_map_,
                    pn.Row(score_wheel_pane, score_md_pane),
                ),
                station_panel,
            )

        return pn.Column(
            pn.Row(model_selector, metric_selector, date_picker),
            pn.panel(tide_panel),
        )

    def create_dashboard(self):
        plot_constituents = self.plot_tidal_graphs()
        self.dashboard = pn.template.MaterialTemplate(
            title="Tidal Dashboard",
            header_background="#007BFF",
            main=pn.Row(
                plot_constituents,
            ),
        )
