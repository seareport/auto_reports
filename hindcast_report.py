from plots import create_skill_report
import os


WORKDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
hind = [
    "hindcast/ds2012.nc",
    "hindcast/ds2013.nc",
    "hindcast/ds2014.nc",
    "hindcast/ds2015.nc",
]
create_skill_report(WORKDIR, hind)
