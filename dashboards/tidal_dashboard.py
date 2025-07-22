from __future__ import annotations

import glob
import logging

import holoviews as hv
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
from auto_reports.graphs.tidal_plots import END
from auto_reports.graphs.tidal_plots import FULL
from auto_reports.graphs.tidal_plots import plot_comparative_amplitudes
from auto_reports.graphs.tidal_plots import plot_relative_amplitudes
from auto_reports.graphs.tidal_plots import SHORT
from auto_reports.graphs.tidal_plots import START
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


class TidalDashboard(param.Parameterized):
    def __init__(self, data_dir="data", **params):
        super().__init__(**params)
        self.data_dir = get_data_dir(data_dir)
        self.all_stats = get_stats(self.data_dir, FULL_LIST)
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
        tidal_param_selector = pn.widgets.Select(
            name="Parameter",
            value="amplitude",
            options=["amplitude", "phase"],
        )

        @pn.depends(
            tidal_param_selector.param.value,
            model_selector.param.value,
        )
        def tide_panel(param, model):
            models_dir = get_models_dir(self.data_dir) / model
            obs_dir = get_obs_dir(self.data_dir)
            self.update_tidal_df(model)
            tide_map_ = tide_map(self.tidal_reduced).opts(**rr.tide_map)

            def update_station_from_map(index):
                if index:
                    # histogram part
                    station = self.tidal_reduced.iloc[index].station.values[0]
                    df = self.tidal_df[self.tidal_df.station == station]
                    rss_sim = compute_rss(df, param, "sim", "obs")
                    plot_relative = plot_relative_amplitudes(df, param, SHORT_LIST)
                    plot_comparison = plot_comparative_amplitudes(
                        df,
                        param,
                        rss_sim,
                        CMAP,
                        SHORT_LIST,
                    )

                    # ts plot part
                    sim = load_data(models_dir / f"{station}.parquet")
                    obs_file = glob.glob(f"{str(obs_dir)}/{station}_*.parquet")[0]
                    obs = load_data(obs_file)
                    ts_plot, corr_sim = generate_tide_ts(
                        obs,
                        sim,
                        START,
                        END,
                        CMAP,
                    )

                    score_sim = compute_score(corr_sim, float(rss_sim))
                else:
                    score_sim = 0
                    plot_relative = empty_plot_relative_amplitudes(param)
                    plot_comparison = empty_plot_comparative_amplitudes(param, CMAP)
                    ts_plot = empty_time_series_plot(START, END)

                return pn.Column(
                    pn.Row(plot_comparison, plot_relative),
                    pn.Row(
                        ts_plot,
                        pn.Column(
                            pn.pane.Bokeh(
                                progress_wheel(score_sim, CMAP),
                                **rr.progress,
                                styles=rr.histo_offset,
                                sizing_mode="fixed",
                            ),
                            pn.pane.Markdown(md.SCORE_MD),
                        ),
                    ),
                )

            selection = hv.streams.Selection1D(source=tide_map_)
            return pn.Row(
                tide_map_,
                pn.bind(update_station_from_map, index=selection.param.index),
            )

        return pn.Column(
            pn.Row(model_selector, tidal_param_selector),
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
