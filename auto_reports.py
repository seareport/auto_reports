from plots import create_storm_surge_report
import pandas as pd
import sys, os
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


STORMS = { # tuple of (start, end, regions)
    'Babet': (pd.Timestamp("2023-09-30"), pd.Timestamp("2023-10-30"), 
              ['Denmark and Germany', 'Belgium and NL', 'UK West and Ireland']),
    
    'Ciarán': (pd.Timestamp("2023-10-08"), pd.Timestamp("2023-11-08"),
        ['English Channel', 'French Channel', 'French Gulf of Gascogne', 
        'Spain Gulf of Gascogne', 'Portugal', 'Spain Atlantic','Med Spain', 
        'Med France', 'Med Ligure', 'Italy mainland west', 'Adriatic'])
}

WORKDIR = os.path.dirname(os.path.realpath(__file__))

for storm in STORMS.keys(): 
    start, end, regions = STORMS[storm]
    create_storm_surge_report(start, end, regions,storm, WORKDIR)  