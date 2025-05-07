from auto_reports.graphs.maps import map_panel
from auto_reports.graphs.maps import bathy_maps
from auto_reports.graphs.taylor import taylor_panel
from auto_reports.graphs.histograms import hist_
from auto_reports.graphs.histograms import create_spider_chart
from auto_reports.graphs.scatter import scatter_table
from auto_reports._io import *
from auto_reports._stats import get_stats
import auto_reports._markdown as md
import auto_reports._render as rr
import panel as pn
import logging
import param
import asyncio
from PyPDF2 import PdfMerger
import numpy as np

# Force visibility everywhere
pn.extension("mathjax", "markdown",raw_css=[OVERRIDE_CSS])

logger = logging.getLogger(name="auto-report")
THRESHOLD = 0.7
QUANTILE = 0.98

# Load data
model = get_model_names()[-4]
print(model)
all_stats = get_stats()
colouring = "ocean"  # can be "name" for Maritime sectors
excluded_stations = ["kala"]

# Process data
stats_full, stats_extreme = all_stats[model]
stats_full = stats_full[~stats_full.index.isin(excluded_stations)]
stats_full = assign_oceans(stats_full)
mask = np.logical_or(
    stats_extreme.observed>stats_extreme.observed.quantile(QUANTILE), 
    stats_extreme.model>stats_extreme.model.quantile(QUANTILE))
stats_extreme = assign_oceans(stats_extreme)

metrics_histo = {
    "rms": "Root Mean Square Error",
    "cr": "Correlation Coefficient",
    "bias": "Systematic Error (Bias)",
    "kge": "Kling Gupta Efficiency",
    "lambda": "Lambda Index",
    "R1": "Error on Highest Peak",
    "R3": "Error on 3 Highest Peaks",
}

METRICS_DOC = md.generate_metrics_doc(metrics_histo)
REPORT_INFO = md.generate_report_info(metrics_histo)

# Create a class for the dashboard to manage state
class RegionalDashboard(param.Parameterized):
    current_region = param.String(default="World")
    export_region = param.String(default="World")
    export_status = param.String(default="")

    def __init__(self, **params):
        super(RegionalDashboard, self).__init__(**params)
        self.stats_full = stats_full
        self.stats_extreme = stats_extreme
        self.model = model
        self.regions = ["Info", "Mesh", "World"] + sorted(list(stats_full.ocean.unique()))
        self.region_select_options = ["World"] + sorted(list(stats_full.ocean.unique()))
        self.cmap = update_color_map(load_world_oceans(), colouring)
        self.dashboard = pn.template.MaterialTemplate()
        self.tabs = pn.Tabs()
        self.region_dropdown = pn.widgets.Select(
            name='Select Region',
            options=self.region_select_options,
            value='World'
        )
        self.region_dropdown.param.watch(self.update_region_tab, 'value')

    def update_region_tab(self, event):
        """Update the current region and switch to the appropriate tab"""
        self.current_region = event.new
        base_index = self.regions.index("World")
        self.tabs[base_index] = (event.new, self.create_region_view(event.new))


    def create_info_tab(self):
        # dashboard_info_html = markdown.markdown(md.DASHBOARD_INFO)
        # metrics_doc_html = markdown.markdown(md.METRICS_DOC)
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"# Model: {self.model} - Information"), visible = True),
            pn.layout.Divider(),
            pn.Row(
                pn.pane.Markdown(REPORT_INFO, visible = True),
                pn.pane.Markdown(METRICS_DOC, visible = True)
            )
        )

    def create_mesh_tab(self):
        w,eu,flo,cb,info = bathy_maps(model)
        return pn.Column(
            info,
            cb.clone(**rr.regional_bathy), 
            w.clone(**rr.regional_bathy), 
            eu.clone(**rr.regional_bathy), 
            flo.clone(**rr.regional_bathy), 
            sizing_mode='stretch_width')

    def create_region_view(self, region):
        if region == "World": 
            stats = self.stats_full.copy()
            stats_ext = self.stats_extreme.copy()
            scat_pn = scatter_table(stats_ext, "#89CFF0")
        else: 
            stats = self.stats_full[self.stats_full.ocean == region].copy()
            stats_ext = self.stats_extreme[self.stats_extreme.ocean == region].copy()
            scat_pn = scatter_table(stats_ext, self.cmap[region])

        taylor_pn = taylor_panel(stats, self.cmap, colouring)
        
        histo_graphs = []
        for m in zip(metrics_histo.keys(), metrics_histo.values()):
            if region == "World":
                h_ = hist_(stats, *m, map=self.cmap, **rr.histogram, yaxis=None, g = None)
            else: 
                h_ = hist_(stats, *m, map=self.cmap, **rr.histogram, yaxis=None)
            h_.styles = rr.histo_offset
            histo_graphs.append(h_)

        spider_chart = create_spider_chart(stats, region, metrics_histo, self.cmap)
        map_pn = map_panel(stats, self.cmap, colouring, region)
        return pn.Column(
            pn.Row(taylor_pn, pn.Column(*histo_graphs), spider_chart, map_pn),
            scat_pn
        )

    async def export_to_pdf(self):
        """Export all tabs from the existing dashboard to PDF (async version)"""
        self.export_status = "Exporting... Please wait."
        
        tabs = self.dashboard.main[1]
        tmp_dir = "data/pdf"        
        html_dir = "data/html"        
        pdf_paths = []
        for i, r in zip(range(len(tabs)), self.regions):
            tabs.active = i
            html_path = f"{html_dir}/{r}.html"
            pdf_path = f"{tmp_dir}/{r}.pdf"
            pdf_paths.append((pdf_path, r))
            tabs[i].save(html_path)
            await html_to_pdf_async(html_path, pdf_path)  
        
        merger = PdfMerger()
        for path, bookmark in pdf_paths:
            with open(path, 'rb') as f:
                merger.append(f, bookmark)
        output_pdf_path = f"Model_{self.model}_Regional_Dashboard.pdf"
        with open(output_pdf_path, 'wb') as f:
            merger.write(f)
        self.export_status = f"Export complete! PDF saved as {output_pdf_path}"
        return output_pdf_path

    def export_dashboard_to_pdf(self, event):
        self.export_status = "Exporting PDF..."
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # async context (i.e. Panel server)
            asyncio.ensure_future(self.export_to_pdf())
        else:
            # sync context (i.e. notebook)
            asyncio.run(self.export_to_pdf())

    def export_dashboard_to_html(self, event):
        """Export dashboard to HTML with a tab for each region"""
        self.export_status = "Exporting..."
        base_index = self.regions.index("World")
        self.tabs[base_index] = ("World", self.create_region_view("World"))
        for region in self.regions[base_index+1:]:
            self.export_status = f"Exporting... {region}"
            self.tabs.append((region, self.create_region_view(region)))
        output_html_path = f"{self.model}_dashboard.html"
        self.export_status = f"Saving HTML..."
        self.dashboard.save(output_html_path)
        self.export_status = f"Export complete! Dashboard saved as {output_html_path}"
        # missing step to reinstate the original tabs

    def create_dashboard(self):
        # Info tab first
        self.tabs.append(("Info", self.create_info_tab()))
        self.tabs.append(("Mesh", self.create_mesh_tab()))
        self.tabs.append(("World", self.create_region_view("World")))

        export_button = pn.widgets.Button(name="Export All Tabs to PDF", button_type="primary")
        export_button.on_click(self.export_dashboard_to_pdf)
        export_button_html = pn.widgets.Button(name="Export dashboard to HTML", button_type="primary")
        export_button_html.on_click(self.export_dashboard_to_html)
   
        self.dashboard = pn.template.MaterialTemplate(
            title=f"Regional Statistics - Model {self.model}",
            header_background="#007BFF",
            main=pn.Column(
                pn.Row(
                    self.region_dropdown,
                    export_button,
                    export_button_html,
                    pn.pane.Markdown(self.param.export_status),
                ),
                self.tabs
            ),
        )
