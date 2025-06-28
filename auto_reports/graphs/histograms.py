from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

import auto_reports._render as rr

METRICS_HISTO = {
    "rms": "Root Mean Square Error",
    "cr": "Correlation Coefficient",
    "bias": "Systematic Error (Bias)",
    "kge": "Kling Gupta Efficiency",
    "lambda": "Lambda Index",
    "R1": "Error on Highest Peak",
    "R3": "Error on 3 Highest Peaks",
}


# Define normalization ranges for each metric type
def get_normalization_range(metric_name):
    if metric_name in ["rmse", "rms", "rms_95", "mad", "madp"]:
        return (0, 0.5)
    elif metric_name in ["bias"]:
        return (-0.2, 0.2)
    elif metric_name in ["slope"]:
        return (0, 2)
    else:
        return (0, 1)


def stacked_hist(plot, element):
    """found here https://discourse.holoviz.org/t/stacked-histogram/6205/2"""
    offset = 0
    for r in plot.handles["plot"].renderers:
        r.glyph.bottom = "bottom"

        data = r.data_source.data
        new_offset = data["top"] + offset
        data["top"] = new_offset
        data["bottom"] = offset * np.ones_like(data["top"])
        offset = new_offset

    plot.handles["plot"].y_range.end = max(offset) * 1.1
    plot.handles["plot"].y_range.reset_end = max(offset) * 1.1


def hist_(src, z, z_name, g="ocean", map=None, type="box", **kwargs):
    range_ = get_normalization_range(z)
    if g:
        df = src[[z, g]].reset_index()

        unique_oceans = df[g].unique()
        # Create a new DataFrame with one-hot encoded structure
        rows = []
        for index, row in df.iterrows():
            new_row = {group: np.nan for group in unique_oceans}
            new_row[row[g]] = row[z]
            rows.append(new_row)
        color_key = hv.Cycle("Category20").values
        # only way to get the colors to match the ocean mapping
        if map is None:
            map = {
                ocean: color_key[i % len(color_key)]
                for i, ocean in enumerate(unique_oceans)
            }
        colors = [map[ocean] for ocean in unique_oceans]
    else:
        df = src[[z]]
    mean = src[z].mean()
    if type == "violin":
        histo_ = hv.Violin(df, g, z).opts(
            violin_fill_color=g,
            cmap=map,
            invert_axes=True,
            ylim=range_,
            title=f"{z_name}, mean value: {mean:.2f}",
            ylabel=z_name,
        )
    elif type == "box":
        histo_ = hv.BoxWhisker(df, g, z).opts(
            box_color=g,
            cmap=map,
            invert_axes=True,
            outlier_radius=0.0005,
            ylim=range_,
            title=f"{z_name}, mean value: {mean:.2f}",
            ylabel="",
        )
    else:
        one_hot_df = pd.DataFrame(rows, columns=unique_oceans)

        histo_ = one_hot_df.hvplot.hist(
            bins=20,
            bin_range=range_,
            # cmap = ocean_mapping,
            color=colors,
        ).opts(
            hooks=[stacked_hist],
            title=f"{z_name}, mean value: {mean:.2f}",
            ylabel=z_name,
        )
    return pn.pane.HoloViews(
        (
            histo_.opts(
                **kwargs,
            )
        ).opts(shared_axes=False),
    )


def create_spider_chart(df, region, metrics_histo, cmap):
    """Create radar chart with metric-specific value ticks on grid lines"""
    # Define which metrics should be inverted (lower is better)
    invert_metrics = [
        "rmse",
        "rms",
        "rms_95",
        "mad",
        "madp",
        "R1",
        "R3",
        "error",
        "bias",
    ]

    # Calculate metrics and prepare grid tick values
    metrics_data = []
    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]  # 5 grid levels

    for m_name, m_label in zip(metrics_histo.keys(), metrics_histo.values()):
        val = df[m_name].mean()
        min_val, max_val = get_normalization_range(m_name)

        # Special handling for bias - use absolute value for normalization
        if m_name == "bias":
            val = abs(val)  # Use absolute value of bias
            if val > 0.2:
                val = 0.2
            min_val = 0  # Min absolute bias is 0
            max_val = max(
                abs(get_normalization_range(m_name)[0]),
                abs(get_normalization_range(m_name)[1]),
            )  # Max absolute bias
        elif m_name == "kge":
            if val < 0:
                val = 0

        # Calculate actual values for each grid level
        grid_values = []
        for level in grid_levels:
            if m_name in invert_metrics:
                actual_val = max_val - (level * (max_val - min_val))
            else:
                actual_val = min_val + (level * (max_val - min_val))

            # If this is bias, display the actual value with sign for grid labels
            if m_name == "bias" and level < 1.0:  # Only modify non-zero values
                if (
                    level <= 0.5
                ):  # For the first half of grid levels, show negative values
                    actual_val = -actual_val

            grid_values.append(actual_val)

        # Normalize current value
        if m_name in invert_metrics:
            norm_val = 1 - ((val - min_val) / (max_val - min_val))
        else:
            norm_val = (val - min_val) / (max_val - min_val)

        metrics_data.append(
            (m_name, m_label, val, norm_val, min_val, max_val, grid_values),
        )

    # Create DataFrame
    metrics_df = pd.DataFrame(
        metrics_data,
        columns=["name", "label", "value", "norm_value", "min", "max", "grid_values"],
    )

    # Prepare angles (close the loop)
    angles = np.linspace(0, 2 * np.pi, len(metrics_df), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    norm_values = metrics_df["norm_value"].tolist() + [metrics_df["norm_value"].iloc[0]]

    # Create radar polygon
    x = np.array(norm_values) * np.cos(angles)
    y = np.array(norm_values) * np.sin(angles)

    if region == "World":
        radar = hv.Polygons([{"x": x, "y": y}], label="World").opts(
            fill_alpha=0.3,
            line_width=2,
        )
    else:
        radar = hv.Polygons([{"x": x, "y": y}]).opts(
            color=cmap[region],
            fill_alpha=0.3,
            line_color=cmap[region],
            line_width=2,
        )

    # Create spider web grid with metric-specific value ticks
    grid_elements = []
    for level_idx, level in enumerate(grid_levels):
        # Create grid ring
        grid_x = np.cos(angles) * level
        grid_y = np.sin(angles) * level
        grid_elements.append(
            hv.Curve((grid_x, grid_y)).opts(
                color="gray",
                line_width=0.5,
                line_dash="dashed",
                alpha=0.3,
            ),
        )

        # Add value ticks for each metric at this level
        for metric_idx, (angle, row) in enumerate(
            zip(angles[:-1], metrics_df.itertuples()),
        ):
            # Position the tick label slightly inside the grid line
            label_x = 0.95 * level * np.cos(angle)
            label_y = 0.95 * level * np.sin(angle)

            # Get the actual value for this metric at this grid level
            actual_value = row.grid_values[level_idx]
            fmt_value = f"{actual_value:.2f}"

            grid_elements.append(
                hv.Text(
                    label_x,
                    label_y,
                    fmt_value,
                ).opts(
                    text_color="gray",
                    text_font_size="6pt",
                    text_align="center",
                    text_baseline="middle",
                ),
            )

    # Create radial spokes with metric names
    label_points = []
    for i, (angle, row) in enumerate(zip(angles[:-1], metrics_df.itertuples())):
        # Main metric label at outer edge - ADJUSTED POSITION
        # Reduced from 1.15 to 1.05 to keep labels closer to the chart
        x = 0.7 * np.cos(angle)
        y = 0.7 * np.sin(angle)

        label_points.append((x, y, f"{row.label}"))
        # Add radial spoke
        grid_elements.append(
            hv.Curve(([0, np.cos(angle)], [0, np.sin(angle)])).opts(
                color="gray",
                line_width=0.5,
                line_dash="dotted",
                alpha=0.9,
            ),
        )

    # Create metric labels
    labels = hv.Labels(
        pd.DataFrame(label_points, columns=["x", "y", "label"]),
        ["x", "y"],
        "label",
    ).opts(
        text_color="black",
        text_font_size="10pt",
        text_align="center",
    )

    # Combine all elements
    spider = (hv.Overlay(grid_elements) * radar * labels).opts(
        **rr.radar,
        title=f"Metrics Radar - {region}",
        xaxis=None,
        yaxis=None,
        show_grid=False,
    )

    return spider
