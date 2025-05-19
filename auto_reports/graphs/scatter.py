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

def make_brightness_gradient(base_color, steps=256):
    r, g, b = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    colors = []
    for lightness in np.linspace(0.2, 0.9, steps):  # darker to lighter
        r2, g2, b2 = colorsys.hls_to_rgb(h, 1 - lightness, s)
        colors.append((r2, g2, b2))
    return LinearSegmentedColormap.from_list(f"gradient_{base_color}", colors)

def return_gauss(df, KDIMS):
    x0 = df[KDIMS[0]].values
    y0 = df[KDIMS[1]].values
    xy = np.vstack([x0,y0])
    z = gaussian_kde(xy)(xy)
    df["density"] = z
    return df

def scatter_table(df, color):
    if df.empty: 
        return pn.pane.Markdown("## No extreme to show")
    else:
        unique_stations = sorted(df['station'].unique().tolist())
        cross_selector = pn.widgets.CrossSelector(
            name='Stations',
            value=[],
            options=unique_stations,
            **rr.cross_selector)
        
        @pn.depends(cross_selector.param.value)
        def create_plots(selected_stations):
            filtered_df = df if not selected_stations else df[df['station'].isin(selected_stations)]
            if filtered_df.empty:
                return pn.pane.Markdown("## No data for the selected stations")
            
            y_range = (filtered_df["diff"].min()-0.1, filtered_df["diff"].max()+0.1)
            cmap = make_brightness_gradient(color)
            options = dict(
                    tools=["box_select", "tap"],
                    alpha=0.7,
                    size=3,
                    ylim=y_range,
                    xlabel="time lag [hours]",
                    ylabel = "model - observed [m]",
                    shared_axes=True, 
                    title = "model vs. observed error",
                    cmap = cmap
                )
            filtered_df = return_gauss(filtered_df, KDIMS)
            scat_ = filtered_df.sort_values("density", ascending=False).hvplot.scatter(
                x=KDIMS[0],
                y=KDIMS[1],
                c="density").opts(**options)
            
            filtered_df["time observed"] = filtered_df.index
            columns = ["station", "name", "lon", "lat", "observed", "model", "diff", "time observed"]
            table = hv.Table(filtered_df[columns]).opts(**rr.table)
            selection = Selection1D(source=table)
            def highlight_points(index):
                if not index or len(index) == 0:
                    return hv.Scatter([]).opts(**options)
                selected_df = filtered_df.iloc[index]
                options["size"] = 8
                options["line_color"] = "k"
                options["alpha"] = 0.9
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

            x_range = (filtered_df["observed"].min()-0.1, filtered_df["observed"].max()+0.1)
            y_range = (filtered_df["model"].min()-0.1, filtered_df["model"].max()+0.1)

            filtered_df = return_gauss(filtered_df, KDIMS2)

            x = filtered_df[KDIMS2[1]]
            y = filtered_df[KDIMS2[0]]
            slope, intercept = seastats.get_slope_intercept(x,y)
            pc1, pc2 = seastats.get_percentiles(x, y, True)
            xx = np.arange(-10, 10)
            yy = xx * slope + intercept
            line = hv.Curve((xx, yy)).opts(line_dash="dashed", color="grey")
            pc_ = hv.Scatter((pc2, pc1)).opts(color="k", alpha = 0.4, size = 6)

            options2 = dict(
                    tools = ["box_select", "tap"],
                    alpha=0.7,
                    size=3,
                    xlabel = "observed",
                    ylabel = "model",
                    xlim = x_range,
                    ylim = y_range,
                    shared_axes = True,
                    title = f"model vs. observed peaks.\nSlope: {slope:.3f}, Offset: {intercept:.2f}",
                    cmap=cmap
                )

            scat2_ = filtered_df.sort_values("density", ascending=False).hvplot.scatter(
                x=KDIMS2[0], 
                y=KDIMS2[1],
                c="density").opts(**options2)
            scat2_ *= line * pc_
            def highlight_points2(index):
                if not index or len(index) == 0:
                    return hv.Scatter([]).opts(**options2)
                selected_df = filtered_df.iloc[index]
                options2["size"] = 8
                options2["line_color"] = "k"
                options2["alpha"] = 0.9
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
                pn.pane.HoloViews(table),
                pn.pane.HoloViews(composition),
                pn.pane.HoloViews(composition2)
                )
        
      
        # Layout with the selector and plots
        return pn.Row(
            pn.Column(
                pn.pane.Markdown(f"### Filter Data {len(unique_stations)} total stations"),
                cross_selector
            ),
            create_plots
        )