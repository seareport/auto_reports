# Model skill performance dashboards

This package purpose is to generate **static** HTML pages used for the dissemination of model skill assessment reports of global surge models.

## Features

### Regional Statistics Dashboard
![regional_dashboard](/thumbnails/global_metrics_twl.png)

- **Multi-tab Dashboard**: Visual exploration by ocean region including info, mesh, and metric views.
- **Key Metrics**:
  - Root Mean Square Error (RMSE)
  - Correlation Coefficient (CR)
  - Bias
  - Kling Gupta Efficiency (KGE)
  - Lambda Index
  - Peak Error metrics (R1, R3)
  - More can be added easily
- **Visual Components**:
  - Histograms
  - Taylor diagrams
  - Scatter plots
  - Spider charts
  - Bathymetric and region maps
- **Export Capabilities**:
  - Export all tabs to a single PDF file.
  - Save interactive dashboard as standalone HTML.

### Historical Storm Events comparison
![storm_events](/thumbnails/storm_events_twl.png)
**Visual Components**:
  - Scatter plots for storm events over 2022-2024
  - model vs obs scatter
  - time series

### Tidal Dashboard
![tide_dashboard](/thumbnails/dashboard_tide.png)
**Visual Components**:
  - Comparitive histograms for main tide constituents
  - Interactive map with callback to switch stations
  - Model vs obs tidal time series
  - Score indicator


## Getting Started

### 1. Install dependencies
We use [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

### 2. Launch the Dashboard
```bash
python -mpanel serve app.py
```

 * TS data is fetched  automatically in `data/models/` and `data/obs/` and processed (if not already done) using the package.
 * Once processed, stats files  are saved in `data/stats/*.parquet`
 * Dashboard will fetch meshes in `data/meshes/*.gr3` and generate graphics in `data/images/*.png`
 * Regional report will be ready in the browser shortly after

### Reproduce

 * Create the `data` folder and subfolders. By default, the application uses data from the `data/` folder. To test with a different folder (e.g., `data_demo/`), set the `data_dir` as initial varible in the dashboard:

```python
instance = RegionalDashboard(data_dir="data_swl")
```


follow the directory structure below:

### Directory Structure

```bash
├── app.py                      # Example app for launching the dashboard
├── auto_reports                # Core functions
│   ├── graphs/                 # All visual components
│   ├── _io.py                  # Data loading and preprocessing
│   ├── _markdown.py            # Markdown content for documentation
│   ├── _proj.py                # [Not used] spilhaus projected mesh maps
│   ├── _render.py              # Plot style and layout settings
│   ├── _stats.py               # Statistical computations
│   ├── _storms.py              # Storms definition
│   └── _tide.py                # (De)Tide functions
├── dashboards                  # Thematic dashboards
│   ├── regional_dashboard.py
│   ├── storms_dashboard.py
│   └── tidal_dashboard.py
├── data_demo
│   ├── models                  # Folder containing "{model}/{station}.parquet" files
│   │   ├── ...
│   │   └── stofs2d
│   │       ├── ...
│   │       └── naha.parquet
│   ├── obs                     # Folder containing "{station}_{sensor}.parquet" observation files
│   │   ├── ...
│   │   └── naha_rad.parquet
│   └── stats                   # Folder containing stats files used in dashboards
│       ├── ...
│       ├── stofs2d_eva.parquet
│       ├── stofs2d.parquet
│       └── stofs2d_tide.parquet
├── LICENSE
├── poetry.lock                 # Project dependencies
├── pyproject.toml              # Project dependencies
├── README.md                   # You're here
└── thumbnails                  # Images for README and landing page
    ├── dashboard_tide.png
    ├── global_metrics_twl.png
    └── storm_events_twl.png
```
