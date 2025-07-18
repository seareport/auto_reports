from __future__ import annotations

import os

import geoviews as gv
import hvplot.pandas  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap

import auto_reports._render as rr
from auto_reports._io import load_countries
from auto_reports._io import load_world_oceans
from auto_reports._io import parse_hgrid
from auto_reports._proj import wgs84_to_spilhaus


def custom_div_cmap(
    numcolors=11,
    name="custom_div_cmap",
    mincol="blue",
    midcol="white",
    maxcol="red",
):
    return LinearSegmentedColormap.from_list(
        name=name,
        colors=[mincol, midcol, maxcol],
        N=numcolors,
    )


BLEVELS = [-8000, -6000, -4000, -3000, -2000, -1500, -1000, -500, -200, -100, 0]
N = len(BLEVELS) - 1
EUROPE = [-15, 5, 40, 55]

FLORIDA = [-85, -70, 10, 30]
BNORM = BoundaryNorm(BLEVELS, ncolors=N, clip=False)
CMAP = LinearSegmentedColormap.from_list(
    name="custom",
    colors=["DarkBlue", "CornflowerBlue", "w"],
    N=N,
)
MOPTS = dict(vmin=-8000, vmax=0, levels=BLEVELS, norm=BNORM, cmap=CMAP, extend="both")


# Matplotlib map
def is_overlapping(tris, meshx, PIR=180):
    x1, x2, x3 = meshx[tris].T
    return np.logical_or(abs(x2 - x1) > PIR, abs(x3 - x1) > PIR, abs(x3 - x3) > PIR)


def bbox_to_corners(bbox):
    xmin, xmax, ymin, ymax = bbox
    x_coords = [xmin, xmax, xmax, xmin, xmin]
    y_coords = [ymin, ymin, ymax, ymax, ymin]
    return np.array((x_coords, y_coords))


def bathy_maps(model):
    print("reading mesh..")
    file = f"data/meshes/{model}.gr3"
    mesh_dic = parse_hgrid(file)
    x, y, depth = mesh_dic["nodes"].T
    tris = mesh_dic["elements"]
    x1, y1 = wgs84_to_spilhaus(x, y)

    print("printing world map..")
    file = f"data/images/{model}_world_spilhaus.png"
    if not os.path.exists(file):
        fig, ax = plt.subplots(figsize=(40, 40))
        m = is_overlapping(tris, x1, PIR=1e6)
        ax.tricontourf(x1, y1, tris[~m], -depth, **MOPTS)
        ax.triplot(x1, y1, tris[~m], lw=0.2, c="k")
        for bbox in [FLORIDA, EUROPE]:
            wgs84_coords = bbox_to_corners(bbox)
            spilhaus_coords = wgs84_to_spilhaus(wgs84_coords[0, :], wgs84_coords[1, :])
            ax.plot(*spilhaus_coords, "r-", linewidth=2)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(file)
    world_map_pn = pn.pane.PNG(file)

    print("printing europe map..")
    file = f"data/images/{model}_europe.png"
    if not os.path.exists(file):
        fig, ax = plt.subplots(figsize=(20 / 20 * 15, 15))
        m = is_overlapping(tris, x)
        ax.tricontourf(x, y, tris[~m], -depth, **MOPTS)
        ax.triplot(x, y, tris[~m], lw=0.25, c="k")
        ax.set_ylim(EUROPE[2], EUROPE[3])
        ax.set_xlim(EUROPE[0], EUROPE[1])
        ax.set_axis_off()
        plt.tight_layout()
        ax.set_aspect("equal")
        fig.savefig(file)
    europe_map_pn = pn.pane.PNG(file)

    print("printing florida map..")
    file = f"data/images/{model}_florida.png"
    if not os.path.exists(file):
        fig, ax = plt.subplots(figsize=(10, 11))
        ax.triplot(x, y, tris[~m], lw=0.25, c="k")
        ax.set_ylim(FLORIDA[2], FLORIDA[3])
        ax.set_xlim(FLORIDA[0], FLORIDA[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(file)
    florida_map_pn = pn.pane.PNG(file)

    print("printing colorbar..")
    file = f"data/images/{model}_colorbar.png"
    if not os.path.exists(file):
        fig, cax = plt.subplots(figsize=(12, 1.2))
        cb = ColorbarBase(
            cax,
            cmap=CMAP,
            norm=BNORM,
            boundaries=BLEVELS,
            ticks=BLEVELS[:-2] + [BLEVELS[-1]],
            spacing="proportional",
            orientation="horizontal",
            extend="both",
        )
        cb.set_label("Bathymetry (m)")
        cb.ax.tick_params(labelrotation=45, labelsize=8)
        plt.tight_layout()
        fig.savefig(file)
    colorbar_pn = pn.pane.PNG(file)

    # info
    text = f"## Number of nodes: {len(x)}\n"
    text += f"## Number of elements: {len(tris)}\n"
    text += f"## Minimum depth: {min(depth)}m\n"
    text += f"## Maximum depth: {max(depth)}m\n"
    info_pn = pn.pane.Markdown(text)

    return world_map_pn, europe_map_pn, florida_map_pn, colorbar_pn, info_pn


# Interactive map
def map_gv(stats, cmap, ocean_or_sector, region):
    countries = load_countries()
    gdf = load_world_oceans()
    points = gv.Points(stats, kdims=["lon", "lat"], vdims=["index"])
    map_ = countries.hvplot(geo=True).opts(color="lightgrey", line_alpha=0.9)

    if region != "World":
        gdf = gdf[gdf[ocean_or_sector] == region]
    return (
        gdf.hvplot(color=ocean_or_sector, geo=True).opts(
            cmap=cmap,
            **rr.map_region,
            show_legend=False,
        )
        * map_
        * points.opts(color="r", line_color="k", size=7, tools=["hover"])
    )


# tide map
def tide_map(df):
    station_points = df.hvplot.points(
        x="lon",
        y="lat",
        geo=True,
        c="score",
        line_color="k",
        cmap="rainbow4_r",
        size=200,
        tiles="CartoLight",
        hover_cols=["station", "score", "corr", "rss"],
        title="Simulation Score by Station",
        tools=["tap"],
        nonselection_alpha=0.6,
    )
    return station_points
