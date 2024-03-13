import logging
import sys

import pandas as pd
import pytest

from auto_reports.reports import create_storm_surge_report

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


STORMS = pytest.mark.parametrize(
    "storm",
    # tuple of (start, end, [list of regions])
    [
        pytest.param(
            (
                pd.Timestamp("2023-09-30"),
                pd.Timestamp("2023-10-30"),
                ["Denmark and Germany", "Belgium and NL", "UK West and Ireland"],
            ),
            id="Babet",
        ),
        pytest.param(
            (
                pd.Timestamp("2023-10-08"),
                pd.Timestamp("2023-11-08"),
                [
                    "English Channel",
                    "French Channel",
                    "French Gulf of Gascogne",
                    "Spain Gulf of Gascogne",
                    "Portugal",
                    "Spain Atlantic",
                    "Med Spain",
                    "Med France",
                    "Med Ligure",
                    "Italy mainland west",
                    "Adriatic",
                ],
            ),
            id="Ciarán",
        ),
    ],
)


@STORMS
def test_create_storm_surge_report(storm):
    create_storm_surge_report(**storm)
