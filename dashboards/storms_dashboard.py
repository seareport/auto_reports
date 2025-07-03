from __future__ import annotations

import logging

import panel as pn
import param

from auto_reports._io import get_data_dir
from auto_reports._io import load_world_oceans
from auto_reports._io import OVERRIDE_CSS
from auto_reports._io import update_color_map
from auto_reports._stats import get_stats
from auto_reports.graphs.ts import plot_ts

# Force visibility everywhere
pn.extension("mathjax", "markdown", raw_css=[OVERRIDE_CSS])

logger = logging.getLogger(name="auto-report")
THRESHOLD = 0.7
QUANTILE = 0.98
colouring = "ocean"  # can be "name" for Maritime sectors


class StormDashboard(param.Parameterized):
    current_region = param.String(default="World")
    export_region = param.String(default="World")
    export_status = param.String(default="")

    def __init__(self, data_dir="data", **params):
        super().__init__(**params)
        self.data_dir = get_data_dir(data_dir)
        self.all_stats = get_stats(self.data_dir)
        self.models = sorted(self.all_stats.keys())
        self.model = self.models[0]
        self.tabs = pn.Tabs()
        self.cmap = update_color_map(load_world_oceans(), colouring)
        self.region_select_options = load_world_oceans()["ocean"].unique().tolist()
        self.dashboard = pn.template.MaterialTemplate()
        self.region_dropdown = pn.widgets.Select(
            name="Select Region",
            options=sorted(self.region_select_options),
            value="World",
        )
        self.region_dropdown.param.watch(self.update_region_tab, "value")

    def update_region_tab(self, event):
        self.current_region = event.new

        regional_stats = {
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

        ts_pane = plot_ts(
            self.models,
            regional_stats,
            self.current_region,
            self.cmap,
            self.data_dir,
        )

        if self.tabs:
            self.tabs[0] = (self.current_region, ts_pane)
        else:
            self.tabs.append((self.current_region, ts_pane))

    def create_dashboard(self):
        self.dashboard = pn.template.MaterialTemplate(
            title="Regional Statistics",
            header_background="#007BFF",
            main=pn.Column(
                pn.Row(
                    self.region_dropdown,
                ),
                self.tabs,
            ),
        )
