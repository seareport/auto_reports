from __future__ import annotations

import datetime
import json
import logging
import typing as T

import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import natsort
import pandas as pd
import panel as pn
import pyarrow.parquet as pq
import pyextremes
import seastats
from pathlib import Path
import logging 

logger = logging.getLogger(name="auto-report")

def plot_ts(models, station, quantile):
    # params
    # obs
    obs = load_data(OBS_DIR / f"{station}.parquet")[start:end]
    obs_threshold = obs.quantile(quantile)
    logger.info("obs len: %r", len(obs))
    logger.info("obs quantile: %r", obs_threshold)
    logger.info("obs describe:\n%r", obs.describe())
    # obs = obs.resample("4min").mean().shift(freq="2min")
    obs_ext = pyextremes.get_extremes(obs, "POT", threshold=obs_threshold, r=f"{window}h")
    logger.info("obs ext:\n%r", obs_ext)
    # sims
    sims = {model: load_data(DATA_DIR / f"models/{model}/{station}.parquet")[start:end] for model in models}
    # plots
    timeseries = [obs.hvplot(label="obs", color="lightgrey")]
    for i, (model, ts) in enumerate(sims.items()):
        timeseries += [ts.hvplot(label=model).opts(color=cc.glasbey_dark[i])]
    timeseries += [hv.HLine(obs_threshold).opts(color="grey", line_dash="dashed", line_width=1)]
    timeseries += [obs_ext.hvplot.scatter(label="obs extreme")]
    return hv.Overlay(timeseries).opts(show_grid=True, active_tools=["box_zoom"], min_height=300, ylabel="sea elevation")
