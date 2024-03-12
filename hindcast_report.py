from plots import create_skill_report
import glob 
import os
import xarray as xr

WORKDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
ds = xr.open_mfdataset(glob.glob("hindcast/ds*.nc"), combine="by_coords")

create_skill_report(WORKDIR, ds)
