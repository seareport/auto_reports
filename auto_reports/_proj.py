import pyproj
from pyproj import Transformer, CRS, Proj
from numpy import sin, cos, tan, arcsin, arctan, arctan2, pi
import numpy as np

if pyproj.PROJ_VERSION >= (9, 6, 0): 
    def wgs84_to_spilhaus(x,y): 
        """
        https://github.com/OSGeo/PROJ/pull/4402 is available in pyproj
        """
        spilhaus_proj = CRS.from_string("+proj=spilhaus")
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(wgs84, spilhaus_proj, always_xy=True)
        x_spilhaus, y_spilhaus = transformer.transform(x, y)
        return x_spilhaus, y_spilhaus
else: 
    def wgs84_to_spilhaus(x,y):
        """
        from https://github.com/rtlemos/spilhaus/blob/main/spilhaus.py
        """
        # constants (https://github.com/OSGeo/PROJ/issues/1851)
        e = np.sqrt(0.00669438)
        lat_center_deg = -49.56371678
        lon_center_deg = 66.94970198
        azimuth_deg = 40.17823482
        
        # parameters derived from constants
        lat_center_rad = lat_center_deg * pi / 180 
        lon_center_rad = lon_center_deg * pi / 180 
        azimuth_rad = azimuth_deg * pi / 180
        conformal_lat_center = -pi / 2 + 2 * arctan(
            tan(pi/4 + lat_center_rad/2) *
            ((1 - e * sin(lat_center_rad)) / (1 + e * sin(lat_center_rad))) ** (e / 2)
        )
        alpha = -arcsin(cos(conformal_lat_center) * cos(azimuth_rad))
        lambda_0 = lon_center_rad + arctan2(tan(azimuth_rad), -sin(conformal_lat_center))
        beta = pi + arctan2(-sin(azimuth_rad), -tan(conformal_lat_center))
        
        # coordinates in radians
        lon = x * pi / 180
        lat = y * pi / 180
        
        # conformal latitude, in radians
        lat_c = -pi / 2 + 2 * arctan(
            tan(pi/4 + lat/2) * ((1 - e * sin(lat)) / (1 + e * sin(lat))) ** (e / 2)
        )
        
        # transformed lat and lon, in degrees
        lat_s = 180 / pi * arcsin(sin(alpha) * sin(lat_c) - cos(alpha) * cos(lat_c) * cos(lon - lambda_0))
        lon_s = 180 / pi * (
            beta + arctan2(
                cos(lat_c) * sin(lon - lambda_0), 
                (sin(alpha) * cos(lat_c) * cos(lon - lambda_0) + cos(alpha) * sin(lat_c))
            )
        )
        
        # projects transformed coordinates onto plane (Adams World in a Square II)
        p = Proj(proj='adams_ws2')
        adams_x, adams_y = p(lon_s, lat_s)
        spilhaus_x = -(adams_x + adams_y) / np.sqrt(2)
        spilhaus_y = (adams_x - adams_y) / np.sqrt(2)
        
        return spilhaus_x, spilhaus_y
