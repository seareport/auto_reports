import holoviews as hv
import auto_reports._render as rr
from holoviews.streams import Selection1D
import panel as pn

KDIMS = ["tdiff", "diff"]
KDIMS2 = ["observed", "model"]
WINDOW = 12 # hours
lag_range = (-WINDOW, WINDOW)

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
            options = dict(
                    color = color,
                    tools=["box_select", "tap"],
                    alpha=0.5,
                    size=5,
                    ylim=y_range,
                    xlabel="time lag [hours]",
                    ylabel = "model - observed [m]",
                    shared_axes=True, 
                    title = "metric and time differences model vs. observed"
                )
            scat_ = filtered_df.hvplot.scatter(
                x=KDIMS[0], 
                y=KDIMS[1]).opts(**options)
            
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
            options2 = dict(
                    color = color,
                    tools = ["box_select", "tap"],
                    alpha = 0.5,
                    size = 5,
                    xlabel = "observed",
                    ylabel = "model",
                    xlim = x_range,
                    ylim = y_range,
                    shared_axes = True,
                    title = "model vs. observed storm peaks"
                )
            scat2_ = filtered_df.hvplot.scatter(
                x=KDIMS2[0], 
                y=KDIMS2[1]).opts(**options2)
            def highlight_points2(index):
                if not index or len(index) == 0:
                    return hv.Scatter([]).opts(**options2)
                selected_df = filtered_df.iloc[index]
                options2["size"] = 8
                options2["line_color"] = "k"
                options2["alpha"] = 0.9
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