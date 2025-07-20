from __future__ import annotations

import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd

import auto_reports._render as rr

CATEGORY_MAP = {
    "Semi-diurnal": [
        "M2",
        "S2",
        "N2",
        "K2",
        "2N2",
        "L2",
        "T2",
        "R2",
        "NU2",
        "MU2",
        "EPS2",
        "LAMBDA2",
    ],
    "Diurnal": ["K1", "O1", "P1", "Q1", "J1", "S1"],
    "Long-period": ["MF", "MM", "MSF", "SA", "SSA", "MSQM", "MTM"],
    "Short-period": ["M4", "MS4", "M6", "MN4", "N4", "S4", "M8", "M3", "MKS2"],
}

SHORT = [
    "M2",
    "S2",
    "N2",
    "K2",
    "2N2",  # Semi-diurnal (twice daily)
    "K1",
    "O1",
    "P1",
    "Q1",
    "J1",
    "S1",  # Diurnal (once daily)
    "MF",
    "MM",
    "MSF",
    "SA",
    "SSA",  # Long period (fortnightly to annual)
    "M4",
    "MS4",
    "M6",
    "MN4",  # Short period (higher harmonics)
]
FULL = [
    "M2",
    "S2",
    "N2",
    "K2",
    "2N2",
    "L2",
    "T2",
    "R2",
    "NU2",
    "MU2",
    "EPS2",
    "LAMBDA2",  # Semi-diurnal (twice daily)
    "K1",
    "O1",
    "P1",
    "Q1",
    "J1",
    "S1",  # Diurnal (once daily)
    "MF",
    "MM",
    "MSF",
    "SA",
    "SSA",
    "MSQM",
    "MTM",  # Long period (fortnightly to annual)
    "M4",
    "MS4",
    "M6",
    "MN4",
    "N4",
    "S4",
    "M8",
    "M3",
    "MKS2",  # Short period (higher harmonics)
]

START = np.datetime64("2024-10-01T00:00:00")
END = np.datetime64("2024-12-31T00:00:00")
STEP = np.timedelta64(20, "m")


def return_ylabel(param):
    if param == "amplitude":
        return "Amplitude [m]"
    elif param == "phase":
        return "Phase (°)"
    else:
        raise ValueError(f"{param} not in ['amplitude', 'phase']")


constituent_to_category = {
    constituent: category
    for category, constituents in CATEGORY_MAP.items()
    for constituent in constituents
}


def plot_comparative_amplitudes(
    df: pd.DataFrame,
    param: str,
    rss: float,
    cmap: dict,
    const: list,
):
    ordered_constituents = const[::-1]
    methods = df.index.levels[1]
    desired_index = [
        (c, m) for c in ordered_constituents for m in methods if (c, m) in df.index
    ]
    df_ordered = df.loc[desired_index]

    return (
        df_ordered[param]
        .hvplot.barh(
            ylabel=return_ylabel(param),
            xlabel="Tidal Constituent",
            grid=True,
            title=f"Tidal {return_ylabel(param)}: Model vs Obs. rss={rss:.2f}",
            legend="top_right",
            rot=90,
        )
        .opts(
            **rr.tidal_barchart_absolute,
            fontsize={"title": 13, "labels": 12, "xticks": 8, "yticks": 8},
            cmap=cmap,
            line_color=None,
            show_legend=True,
            bar_width=0.8,
        )
    )


def plot_relative_amplitudes(df: pd.DataFrame, param: str, const: list):
    df_ = df[param].unstack(level="method")
    df_["relative_difference"] = (df_["sim"] - df_["obs"]) * 100
    df_["category"] = df_.index.map(constituent_to_category).fillna("Other")
    df_["label"] = df_["category"] + "\n" + df_.index

    ordered_index = [c for c in const if c in df_.index][::-1]
    df_ = df_.loc[ordered_index]
    df_ = df_.drop(columns="category")

    return df_.hvplot.barh(
        y="relative_difference",
        x="label",
        ylabel=return_ylabel(param),
        xlabel="Tidal Constituent",
        grid=True,
        title=f"Error (sim - obs) in {return_ylabel(param)}",
        color="relative_difference",
        cmap="coolwarm",
        legend=False,
    ).opts(
        **rr.tidal_barchart_relative,
        fontsize={"title": 13, "labels": 12, "xticks": 8, "yticks": 8},
    )


def empty_plot_relative_amplitudes(param: str):
    df_empty = pd.DataFrame(index=pd.Index(SHORT, name="constituent"))
    df_empty["sim"] = np.nan
    df_empty["obs"] = np.nan
    df_empty["relative_difference"] = np.nan
    df_empty["label"] = df_empty.index.map(lambda x: f"Other\n{x}")

    return df_empty.hvplot.barh(
        y="relative_difference",
        x="label",
        ylabel=return_ylabel(param),
        xlabel="Tidal Constituent",
        grid=True,
        title=f"Error (sim - obs) in {return_ylabel(param)}",
        color="relative_difference",
        cmap="coolwarm",
        legend=False,
    ).opts(
        **rr.tidal_barchart_relative,
        fontsize={"title": 13, "labels": 12, "xticks": 8, "yticks": 8},
    )


def empty_plot_comparative_amplitudes(param: str, cmap: dict):
    methods = ["obs", "sim"]
    index = pd.MultiIndex.from_product(
        [SHORT, methods],
        names=["constituent", "method"],
    )

    df_empty = pd.DataFrame(index=index)
    df_empty[param] = np.nan

    return df_empty.hvplot.barh(
        ylabel=return_ylabel(param),
        xlabel="Tidal Constituent",
        grid=True,
        title=f"Tidal {return_ylabel(param)}: Model vs Obs",
        legend="top_right",
        rot=90,
    ).opts(
        **rr.tidal_barchart_absolute,
        fontsize={"title": 13, "labels": 12, "xticks": 8, "yticks": 8},
        cmap=cmap,
        line_color=None,
        show_legend=True,
        bar_width=0.8,
    )
