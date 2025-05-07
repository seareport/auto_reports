histogram = {
    "width" : 500, 
    "height" : 70
}

histo_offset = {"margin": "0 0 0 0"}  # top right bottom left

taylor = {
    "width": 470,
    "height": 470
}
taylor_offset = {"margin": "0 0 0 0"}  # top right bottom left

map_region = {
    "width": 500,
    "height": 470,
}

radar = {
    "width": 400,
    "height": 470

}

table = {
    "width": 600,
    "height": 450,
}

cross_selector = {
    "width": 300,
    "height": 450
}

scatter = {
    "width": 450,
    "height": 450,
}

scatter_offset = {'margin': '130px 0 0 0px'}

scale = 2
spilhaus_bathy = {
    "width": 1440
}

regional_bathy = {
    "width": 1440
}

def points_opts(cmap, region, y_range):
    color = cmap[region] if region else "b"
    return dict(color=color,
        size=10,
        alpha=1.0,
        tools = ["hover"],
        line_color='black',
        line_width=2,
        ylim = y_range)

def scatter_compo(y_range): 
    return {
        **scatter,
        "ylim" : y_range,
        "shared_axes": True
    }
