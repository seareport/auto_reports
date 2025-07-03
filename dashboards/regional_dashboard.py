from __future__ import annotations

import logging

import pandas as pd
import panel as pn
import param

import auto_reports._markdown as md
import auto_reports._render as rr
from auto_reports._io import assign_storms
from auto_reports._io import get_data_dir
from auto_reports._io import load_world_oceans
from auto_reports._io import OVERRIDE_CSS
from auto_reports._io import update_color_map
from auto_reports._stats import get_stats
from auto_reports.graphs.histograms import create_spider_chart
from auto_reports.graphs.histograms import hist_
from auto_reports.graphs.histograms import METRICS_HISTO
from auto_reports.graphs.maps import bathy_maps
from auto_reports.graphs.maps import map_gv
from auto_reports.graphs.scatter import scatter_table
from auto_reports.graphs.taylor import taylor_panel

pn.extension("mathjax", "markdown", raw_css=[OVERRIDE_CSS])

logger = logging.getLogger(name="auto-report")
THRESHOLD = 0.7
QUANTILE = 0.98

colouring = "ocean"  # can be "name" for Maritime sectors
excluded_stations = ["kala"]

METRICS_DOC = md.generate_metrics_doc(METRICS_HISTO)
REPORT_INFO = md.generate_report_info(METRICS_HISTO)


class RegionalDashboard(param.Parameterized):
    current_region = param.String(default="World")
    export_region = param.String(default="World")
    export_status = param.String(default="")

    def __init__(self, data_dir="data", **params):
        super().__init__(**params)
        self.data_dir = get_data_dir(data_dir)
        self.all_stats = get_stats(self.data_dir)
        self.models = sorted(self.all_stats.keys())
        self.model = self.models[0]
        self.pull_stats()
        self.regions = ["Info", "World"] + sorted(list(self.stats_full.ocean.unique()))
        self.region_select_options = ["World"] + sorted(
            list(self.stats_full.ocean.unique()),
        )
        self.cmap = update_color_map(load_world_oceans(), colouring)
        self.dashboard = pn.template.MaterialTemplate()
        self.tabs = pn.Tabs()
        self.model_dropdown = pn.widgets.Select(
            name="Select Model",
            options=self.models,
            value="Model",
        )
        self.model_dropdown.param.watch(self.update_model, "value")
        self.region_dropdown = pn.widgets.Select(
            name="Select Region",
            options=self.region_select_options,
            value="World",
        )
        self.region_dropdown.param.watch(self.update_region_tab, "value")

    def pull_stats(self):
        stats_full, stats_extreme = self.all_stats[self.model]
        self.stats_full = stats_full
        self.stats_extreme = stats_extreme

    def update_model(self, event):
        """Update the current region and switch to the appropriate tab"""
        self.model = event.new
        self.pull_stats()
        base_index = self.regions.index("World")
        self.tabs[base_index] = (
            self.current_region,
            self.create_region_view(self.current_region),
        )

    def update_region_tab(self, event):
        """Update the current region and switch to the appropriate tab"""
        self.current_region = event.new
        base_index = self.regions.index("World")
        self.tabs[base_index] = (event.new, self.create_region_view(event.new))

    def create_info_tab(self):
        return pn.Column(
            pn.Row(
                pn.pane.Markdown(f"# Model: {self.model} - Information"),
                visible=True,
            ),
            pn.layout.Divider(),
            pn.Row(
                pn.pane.Markdown(REPORT_INFO, visible=True),
                pn.pane.Markdown(METRICS_DOC, visible=True),
            ),
        )

    def create_mesh_tab(self):
        w, eu, flo, cb, info = bathy_maps(self.model)
        return pn.Column(
            info,
            cb.clone(**rr.regional_bathy),
            w.clone(**rr.regional_bathy),
            eu.clone(**rr.regional_bathy),
            flo.clone(**rr.regional_bathy),
            sizing_mode="stretch_width",
        )

    def create_region_view(self, region):
        if region == "World":
            stats = self.stats_full.copy()
            scat_pn = scatter_table(pd.DataFrame(), "#89CFF0")
        else:
            stats = self.stats_full[self.stats_full.ocean == region].copy()
            stats_ext = self.stats_extreme[self.stats_extreme.ocean == region].copy()
            stats_ext = assign_storms(stats_ext, region)
            stats_ext = stats_ext.sort_values(by=["time observed"], ascending=[False])
            scat_pn = scatter_table(stats_ext, self.cmap[region])
        stats["cr"] = stats["cr"].astype(float)
        taylor_pn = taylor_panel(stats, self.cmap, colouring)

        histo_graphs = []
        for m in zip(METRICS_HISTO.keys(), METRICS_HISTO.values()):
            if region == "World":
                h_ = hist_(stats, *m, map=self.cmap, **rr.histogram, yaxis=None, g=None)
            else:
                h_ = hist_(stats, *m, map=self.cmap, **rr.histogram, yaxis=None)
            h_.styles = rr.histo_offset
            histo_graphs.append(h_)

        spider_chart = create_spider_chart(stats, region, METRICS_HISTO, self.cmap)
        map_pn = map_gv(stats, self.cmap, colouring, region).opts(
            shared_axes=False,
            xaxis=None,
            yaxis=None,
        )
        return pn.Column(
            pn.Row(taylor_pn, pn.Column(*histo_graphs), spider_chart, map_pn),
            scat_pn,
        )

    def export_dashboard_to_html(self, event):
        """Export dashboard to HTML with a tab for each region"""
        self.export_status = "Exporting..."
        base_index = self.regions.index("World")
        self.tabs[base_index] = ("World", self.create_region_view("World"))
        for region in self.regions[base_index + 1 :]:
            self.export_status = f"Exporting... {region}"
            self.tabs.append((region, self.create_region_view(region)))
        output_html_path = f"{self.model}_dashboard.html"
        self.export_status = "Saving HTML..."
        self.dashboard.save(output_html_path)
        self.export_status = f"Export complete! Dashboard saved as {output_html_path}"
        # missing step to reinstate the original tabs

    def create_dashboard(self):
        # Info tab first
        self.tabs.append(("Info", self.create_info_tab()))
        self.tabs.append(("World", self.create_region_view("World")))

        export_button_html = pn.widgets.Button(
            name="Export dashboard to HTML",
            button_type="primary",
        )
        export_button_html.on_click(self.export_dashboard_to_html)

        self.dashboard = pn.template.MaterialTemplate(
            title="Regional Statistics",
            header_background="#007BFF",
            main=pn.Column(
                pn.Row(
                    self.model_dropdown,
                    self.region_dropdown,
                    export_button_html,
                    pn.pane.Markdown(self.param.export_status),
                ),
                self.tabs,
            ),
        )
