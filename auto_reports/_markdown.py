DIMENSIONAL_METRICS = {"bias", "rms", "rmse", "mse"}
DIMENSIONLESS_METRICS = {"cr", "kge", "lambda", "nse", "slope"}
STORM_METRICS = {"R1", "R3", "error"}

METRIC_SNIPPETS = {
    "bias": r"""
### Mean Error (or Bias)
$$\langle x_o - x_m \rangle = \langle x_o \rangle - \langle x_m \rangle$$
""",

    "rms": r"""
### RMS (Root Mean Squared)
$$\sqrt{(\langle (x_{o} - \langle x_o \rangle)- (x_m - \langle x_m \rangle))^2 \rangle}$$
""",

    "cr": r"""
### Correlation Coefficient (R)
$$\frac {\langle x_{m}x_{o}\rangle -\langle x_{o}\rangle \langle x_{o}\rangle }{{\sqrt {\langle x_{m}^{2}\rangle -\langle x_{m}\rangle ^{2}}}{\sqrt {\langle x_{o}^{2}\rangle -\langle x_{o}\rangle ^{2}}}}$$
""",

    "kge": r"""
### Kling–Gupta Efficiency (KGE)
$$1 - \sqrt{(r-1)^2 + b^2 + (g-1)^2}$$

Where:  
* `r`: correlation  
* `b`: modified bias $$\frac{\langle x_o \rangle - \langle x_m \rangle}{\sigma_m}$$  
* `g`: std dev ratio $$\frac{\sigma_o}{\sigma_m}$$
""",

    "lambda": r"""
### Lambda Index ($$\lambda$$) [ref](https://eoscience.esa.int/landtraining2017/files/materials/D5P3a_I.pdf)
$$\lambda = 1 - \frac{\sum{(x_c - x_m)^2}}{\sum{(x_m - \overline{x}_m)^2} + \sum{(x_c - \overline{x}_c)^2} + n(\overline{x}_m - \overline{x}_c)^2 + \kappa}$$

Where:  
* $$\kappa = 2 \cdot \left| \sum{((x_m - \overline{x}_m) \cdot (x_c - \overline{x}_c))} \right|$$
""",

    "R1": r"""
### R1 - Error on Highest Peak
Absolute error between the highest modelled and observed peak during storm events.
""",

    "R3": r"""
### R3 - Mean Error on 3 Highest Peaks
Average absolute error between the three largest observed and modelled peaks.
""",

    "error": r"""
### Storm Mean Error
Mean error between modelled and observed storm peaks above a defined threshold.
"""
}

def generate_metrics_doc(metrics_dict):
    header = r"""
# Metrics information
We compare 2 time series:
 * `sim`: modelled surge time series
 * `obs`: observed surge time series

We need metrics to assess the quality of the model.
"""

    dimensional_docs = []
    dimensionless_docs = []
    storm_docs = []

    present_metrics = metrics_dict.keys()

    for metric in present_metrics:
        doc = METRIC_SNIPPETS.get(metric)
        if not doc:
            continue
        if metric in DIMENSIONAL_METRICS:
            dimensional_docs.append(doc)
        elif metric in DIMENSIONLESS_METRICS:
            dimensionless_docs.append(doc)
        elif metric in STORM_METRICS:
            storm_docs.append(doc)

    doc_sections = [header]

    if dimensional_docs:
        doc_sections.append("## A. Dimensional Statistics:\n" + "\n".join(dimensional_docs))

    if dimensionless_docs:
        doc_sections.append("## B. Dimensionless Statistics (best closer to 1):\n" + "\n".join(dimensionless_docs))

    if storm_docs:
        intro = r"""
## C. Storm metrics:
We select the biggest observed storms above a certain quantile, and calculate  the error (absolute difference) between model and observed peaks:
"""
        bullets = []
        if "R1" in present_metrics:
            bullets.append("* `R1` is the error for the biggest storm")
        if "R3" in present_metrics:
            bullets.append("* `R3` is the mean error for the 3 biggest storms")
        if "error" in present_metrics:
            bullets.append('* `"error"` is the mean error for the peaks above that threshold')
        doc_sections.append(intro + "\n" + "\n".join(bullets) + "\n" + "\n".join(storm_docs))

    return "\n".join(doc_sections)


def generate_report_info(metrics_dict):
    header = """# Regional Statistics Report

This report presents performance statistics for ocean surge models across different regions. 

Use the tabs to navigate between the global view and region-specific analyses.
"""

    metrics_section = "## Available Metrics\n\n"
    for key in metrics_dict.keys():
      name = key.upper()
      desc = metrics_dict[key]
      metrics_section += f"- **{name}**: {desc}\n"

    features_section = """
## Features

- **Taylor Diagrams**: Visualize correlation, standard deviation, and RMSE
- **Histograms**: Distribution of key metrics across stations
- **Spider Charts**: Multi-dimensional performance comparison
- **Scatter plot**: Representation of the storm peaks
- **Maps**: Geographical representation of station metrics
"""

    return header + "\n" + metrics_section + features_section

