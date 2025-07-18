from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import utide
from pytides2.tide import Tide

from auto_reports.graphs.tidal_plots import SHORT

logger = logging.getLogger(name="auto-report")

UTIDE_OPTS = {
    "constit": "auto",
    "method": "ols",
    "order_constit": "frequency",
    "Rayleigh_min": 0.97,  # High threshold for constituent resolution
    "verbose": True,
}


def utide_get_coefs(ts: pd.Series, lat: float, resample: int = None) -> dict:
    UTIDE_OPTS["lat"] = lat
    if resample is not None:
        ts = ts.resample(f"{resample}min").mean()
        ts = ts.shift(freq=f"{resample / 2}min")  # Center the resampled points
    return utide.solve(ts.index, ts, **UTIDE_OPTS)


def utide_surge(ts: pd.Series, lat: float, resample: int = None) -> pd.Series:
    ts0 = ts.copy()
    coef = utide_get_coefs(ts, lat, resample)
    tidal = utide.reconstruct(ts0.index, coef, verbose=UTIDE_OPTS["verbose"])
    return pd.Series(data=ts0.values - tidal.h, index=ts0.index)


def pytide_get_coefs(ts: pd.Series, resample: int = None) -> dict:
    if resample is not None:
        ts = ts.resample(f"{resample}min").mean()
        ts = ts.shift(freq=f"{resample / 2}min")  # Center the resampled points
    ts = ts.dropna()
    return Tide.decompose(ts, ts.index.to_pydatetime())[0]


def pytides_surge(ts: pd.Series, resample: int = None) -> pd.Series:
    ts0 = ts.copy()
    tide = pytide_get_coefs(ts, resample)
    t0 = ts.index.to_pydatetime()[0]
    hours = (ts.index - ts.index[0]).total_seconds() / 3600
    times = Tide._times(t0, hours)
    return pd.Series(ts0.values - tide.at(times), index=ts0.index)


def reduce_coef_to_fes(df: pd.DataFrame, cnst: list, verbose: bool = False):
    res = pd.DataFrame(0.0, index=cnst, columns=df.columns)
    common_constituents = df.index.intersection(cnst)
    res.loc[common_constituents] = df.loc[common_constituents]

    not_in_fes_df = df[~df.index.isin(cnst)]
    not_in_fes = not_in_fes_df.index.tolist()
    not_in_fes_amps = not_in_fes_df["amplitude"].round(3).tolist()
    missing_fes = set(cnst) - set(df.index)

    if verbose:
        print(f"Constituents found but not in FES: {not_in_fes}")
        print(f"Their amplitudes: {not_in_fes_amps}")
        if missing_fes:
            print(
                f"FES constituents missing from analysis (set to 0): {sorted(missing_fes)}",
            )

    return res


def pytides_to_df(pytides_tide: Tide) -> pd.DataFrame:
    constituent_names = [c.name.upper() for c in pytides_tide.model["constituent"]]
    return pd.DataFrame(pytides_tide.model, index=constituent_names).drop(
        "constituent",
        axis=1,
    )


def utide_to_df(utide_coef: utide.utilities.Bunch) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amplitude": utide_coef["A"],
            "phase": utide_coef["g"],
            "amplitude_CI": utide_coef["A_ci"],
            "phase_CI": utide_coef["g_ci"],
        },
        index=utide_coef["name"],
    )


def concat_tides_constituents(dict_tides):
    multi_df = pd.concat(dict_tides)
    multi_df.index.names = ["method", "constituent"]
    multi_df = multi_df.swaplevel().sort_index()

    available_constituents = multi_df.index.get_level_values("constituent").unique()
    filtered_order = [c for c in SHORT if c in available_constituents][::-1]
    return multi_df.reindex(filtered_order, level="constituent")


def compute_rss(df: pd.DataFrame, param: str, a: str, b: str):
    df_ = df[param].unstack(level="method")
    df_["rss"] = (df_[a] - df_[b]) ** 2
    return df_["rss"].sum()


def compute_score(corr: float, rss: float) -> float:
    return np.max([0, corr]) * (1 - np.min([rss, 1]))
