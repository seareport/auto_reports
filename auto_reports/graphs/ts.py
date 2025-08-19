from __future__ import annotations

import typing as T

import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd

import auto_reports._render as rr
from auto_reports._stats import sim_on_obs

OPTS = dict(
    show_grid=True,
    active_tools=["box_zoom"],
    line_width=1.5,
    line_alpha=0.8,
    **rr.time_series_storm,
    # legend_position = "right"
)


def empty_time_series_plot(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    dates = pd.date_range(start_date, end_date, freq=pd.Timedelta("20min"))
    df_empty = pd.DataFrame(index=dates, columns=["sim", "obs"])
    df_empty = df_empty.astype(float)

    return df_empty.hvplot(
        xlabel="Date",
        ylabel="Water Level (m)",
        grid=True,
        title="Tidal TS empty",
        width=800,
        height=300,
        line_width=2,
    ).opts(
        **rr.time_series_tide,
    )


def generate_storm_ts(sims, obs, extremes, station, storm, demean):
    if demean:
        mean = obs.mean()
        obs = obs - mean
    # plots
    storm_peaks = extremes[list(extremes.keys())[0]]
    storm_peak = storm_peaks[
        (storm_peaks.station == station) & (storm_peaks.storm == storm)
    ]
    if demean:
        storm_peak.loc[:, "observed"] -= mean
    curve = hv.Curve(obs, label="obs").opts(
        color="grey",
        **OPTS,
        title=station,
    ) * storm_peak.hvplot.scatter(
        x="time observed",
        y="observed",
        s=50,
        c="grey",
        label="obs",
    )
    for i, (model, ts) in enumerate(sims.items()):
        storm_peaks = extremes[model]
        storm_peak = storm_peaks[
            (storm_peaks.station == station) & (storm_peaks.storm == storm)
        ]
        curve *= hv.Curve(ts, label=f"{model}").opts(
            color=cc.glasbey_dark[i],
            **OPTS,
        ) * storm_peak.hvplot.scatter(
            x="time model",
            y="model",
            s=50,
            c=cc.glasbey_dark[i],
            label=f"{model}",
        )

    return curve.opts(
        title=storm,
        shared_axes=False,
        **rr.time_series_storm,
    )


def data_availability(series: pd.Series, freq="60min") -> float:
    resampled = series.resample(freq).mean()
    data_avail_ratio = 1 - resampled.isna().sum() / len(resampled)
    return float(data_avail_ratio)


def generate_tide_ts(
    obs: pd.Series,
    sim: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cmap: dict,
) -> T.Tuple[hv.Curve, float]:
    obs = obs.drop_duplicates()
    sim = sim.drop_duplicates()
    ts_sim_obs, _ = sim_on_obs(sim, obs)
    ts_sim_obs = ts_sim_obs[~ts_sim_obs.index.duplicated(keep="first")]
    sim = sim[~sim.index.duplicated(keep="first")]
    obs = obs[~obs.index.duplicated(keep="first")]
    df_sim_obs_corr = pd.concat(
        {
            "sim": ts_sim_obs,
            "obs": obs,
        },
        axis=1,
    )

    df_sim_obs_plot = pd.concat(
        {
            "sim": sim.loc[start_date:end_date],
            "obs": obs.loc[start_date:end_date],
        },
        axis=1,
    )

    corr_matrix = df_sim_obs_corr.corr(method="pearson")
    corr_sim = corr_matrix.loc["sim", "obs"]

    ts_plot = (
        df_sim_obs_plot.resample("20min")
        .mean()
        .shift(freq="10min")
        .hvplot(
            xlabel="Date",
            ylabel="Water Level (m)",
            grid=True,
            title=f"Model vs Obs - corr={corr_sim:.2f}",
            line_width=1.5,
            color=list(cmap.values()),
            **rr.time_series_tide,
        )
    )

    return ts_plot, corr_sim
