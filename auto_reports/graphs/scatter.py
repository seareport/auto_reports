import holoviews as hv
import auto_reports._render as rr
from holoviews.streams import Selection1D
import panel as pn
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import colorsys
import seastats

KDIMS = ["tdiff", "diff"]
KDIMS2 = ["observed", "model"]
WINDOW = 12 # hours
lag_range = (-WINDOW, WINDOW)
BASE_OPTIONS = dict(
    shared_axes=True,
    tools=["box_select", "tap", "hover"],
    size=10,
    legend_position= "bottom",
    legend_cols=6,
    muted=True
)

def make_brightness_gradient_hex(base_color, steps=10, lightness_range=(0.3, 0.85)):
    r, g, b = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    colors_hex = []
    lightness_values = np.linspace(*lightness_range, steps)
    
    for lightness in lightness_values:
        r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
        hex_color = '#{0:02x}{1:02x}{2:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))
        colors_hex.append(hex_color)
    
    return colors_hex

def scatter_table(df, color):
    if df.empty: 
        return pn.pane.Markdown("## No extreme to show")
    else:
        
        unique_stations = sorted(df['station'].unique().tolist())
        unique_storms = sorted(df['storm'].unique().tolist())

        station_selector = pn.widgets.CrossSelector(
            name='Stations',
            value=[],
            options=unique_stations,
            **rr.cross_selector
        )

        storm_selector = pn.widgets.CrossSelector(
            name='Storms',
            value=[],
            options=unique_storms,
            **rr.cross_selector
        )

        selector_panel = pn.Row(
            pn.Column(
                pn.pane.Markdown(f"### Select from {len(unique_stations)} stations"),
                station_selector
            ),
            pn.Column(
                pn.pane.Markdown(f"### Select from {len(unique_storms)} storms"),
                storm_selector
            )
        )

        toggle_btn = pn.widgets.Button(name='Hide', button_type='primary')

        def toggle_visibility(event):
            selector_panel.visible = not selector_panel.visible
            toggle_btn.name = 'Show' if not selector_panel.visible else 'Hide'

        toggle_btn.on_click(toggle_visibility)

        @pn.depends(station_selector.param.value, storm_selector.param.value)
        def create_plots(selected_stations, selected_storms):
            filtered_df = df.copy()

            if selected_stations:
                filtered_df = filtered_df[filtered_df['station'].isin(selected_stations)]
            if selected_storms:
                filtered_df = filtered_df[filtered_df['storm'].isin(selected_storms)]

            if filtered_df.empty:
                return pn.pane.Markdown("## No data for the selected stations")
            
            y_range = (filtered_df["diff"].min()-0.1, filtered_df["diff"].max()+0.1)
            options = {
                    "ylim":(min(0, y_range[0]), y_range[1]),
                    "xlabel":"time lag [hours]",
                    "ylabel": "model - observed [m]",
                    "title": "model vs. observed error",
                    **BASE_OPTIONS
                }
            colors = make_brightness_gradient_hex(color, len(filtered_df.storm.unique()))
            overlays = {
                storm: filtered_df[filtered_df.storm==storm].hvplot.scatter(
                    x=KDIMS[0],
                    y=KDIMS[1],
                    c = colors[istorm],
                    label=str(storm),  # Important for the legend
                ).opts(**options)
                for istorm, storm in enumerate(filtered_df.storm.unique())
            }
            scat_ = hv.NdOverlay(overlays, kdims='storm')
            columns = ["station", "name", "storm", "observed", "model", "diff", "tdiff"]
            table = hv.Table(filtered_df[columns].round(2)).opts(**rr.table)
            selection = Selection1D(source=table)
            def highlight_points(index):
                if not index or len(index) == 0:
                    return hv.Scatter([]).opts(**options)
                selected_df = filtered_df.iloc[index]
                options["size"] = 20
                options["line_color"] = "k"
                options["color"] = color
                return selected_df.hvplot.scatter(
                    x = KDIMS[0], 
                    y = KDIMS[1], 
                    hover_cols=["time observed", "station", "name"]
                ).opts(**options)

            highlighted_scatter = hv.DynamicMap(highlight_points, streams=[selection])
            composition = ((scat_ * highlighted_scatter).opts(
                **rr.scatter_compo(y_range))
                )

            data_min = min(filtered_df["observed"].min(), filtered_df["model"].min()) - 0.1
            data_max = max(filtered_df["observed"].max(), filtered_df["model"].max()) + 0.1
            x_range = y_range = (min(0, data_min), data_max)

            x = filtered_df[KDIMS2[1]]
            y = filtered_df[KDIMS2[0]]
            slope, intercept = seastats.get_slope_intercept(x,y)
            xx = np.arange(-10, 10)
            yy = xx * slope + intercept
            line = hv.Curve((xx, yy)).opts(line_dash="dashed", color="grey")

            options2 = {
                "xlabel": "observed",
                "ylabel": "model",
                "xlim": (min(0, x_range[0]), x_range[1]),
                "ylim":(min(0, y_range[0]), y_range[1]),
                "title": f"model vs. observed peaks.\nSlope: {slope:.3f}, Offset: {intercept:.2f}",
                **BASE_OPTIONS
            }

            overlays = {
                storm: filtered_df[filtered_df.storm==storm].hvplot.scatter(
                    x=KDIMS2[0],
                    y=KDIMS2[1],
                    c = colors[istorm],
                    label=str(storm),  # Important for the legend
                ).opts(**options2)
                for istorm, storm in enumerate(filtered_df.storm.unique())
            }
            scat2_ = hv.NdOverlay(overlays, kdims='storm')
            scat2_ *= line
            def highlight_points2(index):
                if not index or len(index) == 0:
                    return hv.Scatter([]).opts(**options2)
                selected_df = filtered_df.iloc[index]
                options2["size"] = 20
                options2["line_color"] = "k"
                options2["color"] = color
                return selected_df.hvplot.scatter(
                    x = KDIMS2[0], 
                    y = KDIMS2[1], 
                    hover_cols=["time observed", "station", "name"]
                ).opts(**options2)
            xy_axis = hv.Curve([(0,0), (10,10)]).opts(color="k")
            highlighted_scatter = hv.DynamicMap(highlight_points2, streams=[selection])
            composition2 = (scat2_ * highlighted_scatter * xy_axis).opts(**rr.scatter)

            return pn.Row(
                pn.pane.HoloViews(table.opts(**rr.table)),
                pn.pane.HoloViews(composition),
                pn.pane.HoloViews(composition2)
                )
        
      
        # Layout with the selector and plots
        return pn.Row(
            pn.Column(
                toggle_btn,
                selector_panel
            ),
            create_plots
        )