import glob
import logging
import sys

import pytest
import xarray as xr

from auto_reports.reports import create_skill_report

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

MODEL = pytest.mark.parametrize(
    "file", [glob.glob("model/oper/*.nc")], id="global-v2-TELEMAC"
)


@MODEL
def test_create_skill_report(file):
    ds = xr.open_mfdataset(file, combine="by_coords")
    create_skill_report(ds)
