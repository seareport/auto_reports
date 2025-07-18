from __future__ import annotations

from math import pi

from bokeh.plotting import figure

import auto_reports._render as rr


def progress_wheel(actual, cmap):
    p = figure(
        **rr.progress,
        x_range=(-1.6, 1.6),
        y_range=(-1.6, 1.6),
        toolbar_location=None,
        background_fill_color=None,
    )

    start_angle = pi / 2
    # end_ref = start_angle - 2 * pi * reference
    # p.annular_wedge(
    #     x=0, y=0,
    #     inner_radius=1.02, outer_radius=1.15,
    #     start_angle=end_ref, end_angle=start_angle,
    #     color=cmap['fes'], alpha=0.8,
    #     line_color=None
    # )

    end_actual = start_angle - 2 * pi * actual
    p.annular_wedge(
        x=0,
        y=0,
        inner_radius=0.55,
        outer_radius=0.99,
        start_angle=end_actual,
        end_angle=start_angle,
        color=cmap["sim"],
        alpha=0.4,
        line_color=None,
    )

    p.text(
        x=0,
        y=0,
        text=[f"Score\n{int(actual * 100)}%"],
        text_align="center",
        text_baseline="middle",
        text_font_size="20pt",
    )

    # p.text(
    #     x=0, y=1.20,
    #     text=[f"FES Score: {int(reference * 100)}%"],
    #     text_align='center',
    #     text_font_size='15pt'
    # )

    # Clean axes
    p.axis.visible = False
    p.grid.visible = False

    return p
