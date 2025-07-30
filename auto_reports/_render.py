from __future__ import annotations

histogram = {
    "width": 500,
    "height": 70,
}

time_series = {
    "width": 800,
    "height": 500,
}

time_series_tide = {
    "width": 1200,
    "height": 500,
}
time_series_storm = {
    "width": 800,
    "height": 300,
}

progress = {
    "width": 300,
    "height": 300,
}

tidal_barchart_relative = {
    "height": 600,
    "width": 380,
}

tidal_barchart_amplitude = {
    "height": 600,
    "width": 450,
}
tidal_barchart_phase = {
    "height": 600,
    "width": 340,
}

histo_offset = {"margin": "0 0 0 0"}  # top right bottom left

taylor = {
    "width": 470,
    "height": 470,
}
taylor_offset = {"margin": "0 0 0 0"}  # top right bottom left

map_region = {
    "width": 500,
    "height": 470,
}

map_storm = {
    "width": 700,
    "height": 470,
}

tide_map = {
    "width": 800,
    "height": 700,
}

radar = {
    "width": 400,
    "height": 470,
}

table = {
    "width": 400,
    "height": 550,
}

cross_selector = {
    "width": 250,
    "height": 550,
}

model_cross_selector = {
    "width": 400,
    "height": 550,
}

scatter = {
    "width": 500,
    "height": 600,
}

scatter_ext = {
    "width": 700,
    "height": 500,
}

scale = 2
spilhaus_bathy = {
    "width": 1440,
}

regional_bathy = {
    "width": 1440,
}


def points_opts(cmap, region, y_range):
    color = cmap[region] if region else "b"
    return dict(
        color=color,
        size=10,
        alpha=1.0,
        tools=["hover"],
        line_color="black",
        line_width=2,
        ylim=y_range,
    )


def scatter_compo(y_range):
    return {
        **scatter,
        "ylim": y_range,
        "shared_axes": True,
    }
