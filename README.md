# Regional Model Performance Report

![auto-report_20250616](https://github.com/user-attachments/assets/fe479755-25c7-4b9d-a9fd-56f1965e0549)

This package purpose is to generate **static** HTML pages used for the dissemination of model skill assessment reports of global surge models.

## Features

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
.
├── app.py            # app launching the dashboard
├── regional_dashboard.py # Main dashboard example
├── auto_reports/     # Core functions
│ ├── _io.py          # Data loading and preprocessing
│ ├── _stats.py       # Statistical computations
│ ├── _render.py      # Plot style and layout settings
│ ├── _markdown.py    # Markdown content for documentation
│ ├── graphs/         # All visual components
│ │ ├── histograms.py # Histogram visualizations
│ │ ├── maps.py       # Bathymetry and region maps
│ │ ├── scatter.py    # Scatter plots for extreme values
│ │ ├── taylor.py     # Taylor diagrams
│ │ └── ts.py         # (Placeholder) TS visualizations
├── data/
│ ├── html/, pdf/     # Temp folders for PDF exports
│ ├── images/         # Folder for images contined in reports
│ │   └── *.png
│ ├── meshes/         # Folder containing "{model}.gr3" meshes
│ │   ├── seareport-v3.2.gr3
│ │   └── stofs2d.gr3
│ ├── models/         # Folder containing "{model}/{station}.parquet" files
│ │   ├── seareport-v3.2
│ │   │   ├── benc.parquet
│ │   │   ├── ...
│ │   │   └── naha.parquet
│ │   └── stofs2d
│ │   │   ├── benc.parquet
│ │   │   ├── ...
│ │   │   └── naha.parquet
│ ├── obs/            # Folder containing "{station}_{sensor}.parquet" observation files
│ │   ├── benc.parquet
│ │   ├── ...
│ │   └── naha.parquet
│ └── stats/          # Folder containing "{model}.parquet", "{model}_eva.parquet" stats computed in auto_report/_stats.py routine
│     ├── seareport-v3.2_eva.parquet
│     ├── seareport-v3.2.parquet
│     ├── stofs2d_eva.parquet
│     └── stofs2d.parquet│
├── *.html, *.pdf # Exported reports
├── pyproject.toml, poetry.lock # Project dependencies
└── README.md # You're here!
```
