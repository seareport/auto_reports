
import azure.identity
import pandas as pd
import matplotlib.patheffects as pe

def get_credential() -> azure.identity.ChainedTokenCredential:
   credential_chain = (
       azure.identity.EnvironmentCredential(),
       azure.identity.AzureCliCredential(),
       azure.identity.ManagedIdentityCredential(),
   )
   credential = azure.identity.ChainedTokenCredential(*credential_chain)
   return credential
# 


# GLOBAL VARIABLES
NCPU = 5
STORAGE_AZ = {'account_name': "seareport", "credential": get_credential()}
CONTAINER = "global-v1"
SKILL_SETTINGS =[
              {"skill_param" : 'RMSE',                            "vmin" : 0,   "vmax" : 1},
              {"skill_param" : 'Mean Absolute Error',             "vmin" : 0,   "vmax" : 1},
              {"skill_param" : 'BIAS or mean error',              "vmin" : -.5, "vmax" : .5},
              {"skill_param" : 'Standard deviation of residuals', "vmin" : -.5, "vmax" : .5}]
KURTOSIS_THRESHOLD = 20
# 
NOW = pd.Timestamp.now(tz='UTC')
START = NOW - pd.Timedelta(days=30)
BASE_URL = "https://www.ioc-sealevelmonitoring.org/service.php?query=data&timestart={start}&timestop={end}&code={ioc_code}"
VALID_SENSORS = {'rad', 'prs', 'enc', 'pr1', 'PR2', 'pr2', 'ra2', 'bwl', 'wls', 'aqu', 'ras', 'pwl', 'bub', 'enb', 'atm', 'flt', 'ecs', 'stp', 'prte', 'prt', 'ra3'}
SEASET_CATALOG = pd.read_csv("https://raw.githubusercontent.com/tomsail/seaset/main/Notebooks/catalog_full.csv", index_col=0)

OPTS = { # tidal analysis options
    "conf_int": "linear",
    "constit": "auto",
    "method": "ols",  # ols is faster and good for missing data (Ponchaut et al., 2001)
    "order_constit": "frequency",
    "Rayleigh_min": 0.97,
    "lat": None,
    "verbose": False,
}  # careful if there is only one Nan parameter, the analysis crashes
# plot
PROPS = dict(boxstyle='round', facecolor='white', alpha=0.5)
WHITE_STROKE = [pe.withStroke(linewidth=2, foreground='w')]
# 
COASTLINES = { # tuple of lon min, lon max, lat min, lat max
    "UK East": (-1.76, 1.76, 51.0, 55.81),  # East coast of the UK from roughly the Thames estuary to the Scottish border
    "Denmark and Germany": (6.0, 14.0, 53.39, 57.75),  # German North Sea coastline
    "Belgium and NL": (2.54, 7.22, 51.0, 53.55),  # Belgian coastline
    "UK West and Ireland": (-10.56, -5.34, 50.0, 58.67),  # West coast of the UK and the entire coast of Ireland
    "English Channel": (-5.72, 1.76, 49.95, 51.14),  # English Channel coastline
    "French Channel": (-5.13, 2.54, 48.63, 51.1),  # French coastline along the English Channel
    "French Gulf of Gascogne": (-4.79, 3.09, 43.39, 48.63),  # French coast along the Bay of Biscay
    "Spain Gulf of Gascogne": (-9.3, -1.77, 43.33, 48.0),  # Spanish coast along the Bay of Biscay
    "Portugal": (-9.5, -6.19, 36.98, 42.15),  # Portuguese coastline along the Atlantic
    "Spain Atlantic": (-9.5, -6.19, 36.0, 43.79),  # Spanish coastline along the Atlantic excluding the Gulf of Gascogne
    "Med Spain": (-0.32, 4.32, 36.0, 43.79),  # Spanish Mediterranean coastline
    "Med France": (3.09, 7.85, 42.33, 43.93),  # French Mediterranean coastline
    "Med Ligure": (7.0, 10.0, 43.0, 44.4),  # Ligurian Sea coast of Italy and France
    "Italy mainland west": (10.0, 18.5, 38.0, 43.0),  # West coast of mainland Italy
    "Med Sardegna": (8.13, 9.7, 38.86, 41.26),  # Coastline of Sardinia, Italy
    "Med Corsica": (8.56, 9.56, 41.37, 43.01),  # Coastline of Corsica, France
    "Med Sicilia": (12.0, 16.0, 36.65, 38.7),  # Coastline of Sicily, Italy
    "Med Ionian": (15.65, 18.17, 36.5, 40.3),  # Ionian Sea coast of Italy and Greece
    "Adriatic": (12.0, 19.6, 40.0, 45.8)  # Adriatic Sea coast from Italy to Croatia
}
