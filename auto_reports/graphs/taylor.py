from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import HoverTool

import auto_reports._render as rr

hv.extension("bokeh")


def create_std_dev_circles(std_dev_range: np.ndarray) -> hv.Overlay:
    std_dev_circles = []
    for std in std_dev_range:
        angle = np.linspace(0, np.pi / 2, 100)
        radius = np.full(100, std)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        std_dev_circles.append(
            hv.Curve((x, y)).opts(color="gray", line_dash="dotted", line_width=1),
        )
    return hv.Overlay(std_dev_circles)


def create_std_ref(radius: float) -> hv.Overlay:
    angle = np.linspace(0, np.pi / 2, 100)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return hv.Curve((x, y)).opts(
        color="gray",
        line_dash="dashed",
        line_width=2,
    ) * hv.Text(radius, 0.0, "REF", halign="right", valign="bottom").opts(
        text_font_size="10pt",
        text_color="gray",
    )


def create_corr_lines(corr_range: np.ndarray, std_dev_max: float) -> hv.Overlay:
    corr_lines = []
    for corr in corr_range:
        theta = np.arccos(corr)
        radius = np.linspace(0, std_dev_max, 2)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        corr_lines.append(
            hv.Curve((x, y)).opts(color="blue", line_dash="dashed", line_width=1)
            * hv.Text(x[-1], y[-1], f"{corr:.2f}", halign="left", valign="bottom").opts(
                text_font_size="10pt",
                text_color="blue",
            ),
        )
    corr_label = hv.Text(
        0.75 * std_dev_max,
        0.75 * std_dev_max,
        "Correlation Coefficient",
    ).opts(text_font_size="12pt", text_color="blue", angle=-45)
    return hv.Overlay(corr_lines) * corr_label


def create_rms_contours(
    standard_ref: float,
    std_dev_max: float,
    rms_range: np.ndarray,
    norm: bool,
) -> hv.Overlay:
    rms_contours = []
    for rms in rms_range:
        angle = np.linspace(0, np.pi, 100)
        x = standard_ref + rms * np.cos(angle)
        y = rms * np.sin(angle)
        inside_max_std = np.sqrt(x**2 + y**2) < std_dev_max
        x[~inside_max_std] = np.nan
        y[~inside_max_std] = np.nan
        rms_contours.append(
            hv.Curve((x, y)).opts(color="green", line_dash="dashed", line_width=1)
            * hv.Text(
                standard_ref + rms * np.cos(2 * np.pi / 3),
                rms * np.sin(2 * np.pi / 3),
                f"{rms*100:.0f}%",
                halign="left",
                valign="bottom",
            ).opts(text_font_size="10pt", text_color="green"),
        )
    label = "Centered RMS" if norm else "RMS"
    rms_label = hv.Text(
        standard_ref,
        rms_range[1] * np.sin(np.pi / 2),
        label,
        halign="left",
        valign="bottom",
    ).opts(text_font_size="11pt", text_color="green")
    return hv.Overlay(rms_contours) * rms_label


def taylor_diagram(
    df: pd.DataFrame,
    norm: bool = True,
    marker: str = "circle",
    color: str = "black",
    cmap=None,
    label: str = "Taylor Diagram",
) -> hv.Overlay:
    if df.empty:
        std_range = np.arange(0, 1.5, np.round(1 / 5, 2))
        corr_range = np.arange(0, 1, 0.1)
        rms_range = np.arange(0, 1.5, np.round(1 / 5, 2))
        std_dev_overlay = create_std_dev_circles(std_range) * create_std_ref(1)
        corr_lines_overlay = create_corr_lines(corr_range, std_range.max())
        rms_contours_overlay = create_rms_contours(
            1,
            std_range.max(),
            rms_range,
            norm=norm,
        )
        return std_dev_overlay * corr_lines_overlay * rms_contours_overlay
    theta = np.arccos(df["cr"])  # Convert Cr to radians for polar plot
    if norm:
        std_ref = 1
        std_mod = df["sim_std"] / df["obs_std"]
    else:
        if len(df) > 1:
            raise ValueError(
                "for not normalised Taylor diagrams, you need only 1 data point",
            )
        std_ref = df["sim_std"].mean()
        std_mod = df["obs_std"].mean()
    #
    std_range = np.arange(0, 1.5 * std_ref, np.round(std_ref / 5, 2))
    corr_range = np.arange(0, 1, 0.1)
    rms_range = np.arange(0, 1.5 * std_ref, np.round(std_ref / 5, 2))

    std_dev_overlay = create_std_dev_circles(std_range) * create_std_ref(std_ref)
    corr_lines_overlay = create_corr_lines(corr_range, std_range.max())
    rms_contours_overlay = create_rms_contours(
        std_ref,
        std_range.max(),
        rms_range,
        norm=norm,
    )

    x = std_mod * np.cos(theta)
    y = std_mod * np.sin(theta)
    df = df.assign(x=x, y=y, rms_perc=df["rms"] / df["obs_std"])
    # hover parameters
    tooltips = [
        ("Corr Coef (%)", "@cr"),
        ("Centered RMS (m)", "@rms"),
        ("Station (m)", "@id"),
        ("Ocean", "@ocean"),
    ]
    hover = HoverTool(tooltips=tooltips)

    # Scatter plot for models with hover tool
    scatter_plot = hv.Points(
        df,
        ["x", "y"],
        ["cr", "rms", "name", "ocean", "index"],
        label=label,
    ).opts(
        color=color,
        cmap=cmap,
        line_color="k",
        line_width=0.7,
        marker=marker,
        size=7,
        tools=[hover],
        default_tools=[],
        show_legend=True,
        hover_fill_color="firebrick",
        xlim=(0, std_range.max() * 1.05),
        ylim=(0, std_range.max() * 1.05),
        fontscale=0.9,
    )
    # Combine all the elements
    taylor_diagram = scatter_plot
    return taylor_diagram


def taylor_panel(stats, cmap, colouring):
    T_VOID = taylor_diagram(pd.DataFrame())
    taylor = taylor_diagram(
        stats,
        norm=True,
        color=colouring,
        cmap=cmap,
    ).opts(
        show_grid=True,
        show_legend=False,
        default_tools=["pan"],
        tools=["hover", "box_zoom", "reset", "save"],
    )
    taylor_ = (T_VOID * taylor).opts(
        **rr.taylor,
        title=f"Taylor Diagram - {len(stats)} stations",
    )
    return pn.pane.HoloViews(
        (taylor_).opts(shared_axes=False),
    )
