import math
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from numpy import gradient
from numpy import pi
from numpy import arctan
from numpy import arctan2
from numpy import sin
from numpy import cos
from numpy import sqrt
from numpy import zeros
from numpy import uint8
import rasterio

# from http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html
def hillshade(array, azimuth, angle_altitude):

    x, y = gradient(array, 2, 2)
    slope = pi/2. - arctan(sqrt(x*x + y*y))
    aspect = arctan2(-x, y)
    azimuthrad = azimuth*pi / 180.
    altituderad = angle_altitude*pi / 180.


    shaded = sin(altituderad) * sin(slope)\
     + cos(altituderad) * cos(slope)\
     * cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2


# fron http://help.arcgis.com/en/arcgisdesktop/10.0/help/index.html#/How_Hillshade_works/009z000000z2000000/, courtesy of GeoTrellis
def gt_hillshade(array, azimuth, angle_altitude, z_factor):
    azimuth = np.radians(90.0 - azimuth)
    zenith = np.radians(90.0 - angle_altitude)

    cos_z = cos(zenith)
    sin_z = sin(zenith)
    cos_az = cos(azimuth)
    sin_az = sin(azimuth)

    x, y = gradient(array, 2, 2)
    # slope = arctan(z_factor * sqrt(x*x + y*y))

    def _aspect(x, y):
        # flat (this is so cos(azimuth - aspect) == 0.5 == 50% grey)
        a = -1 * (pi/4 - azimuth)

        if x == 0:
            if y > 0:
                a = pi / 2
            elif y < 0:
                a = 2 * pi - pi / 2
        else:
            a = arctan2(y, -x)

            if a < 0:
                a += 2 * pi

        return a


    def _cos_aspect(x, y):
        if x == 0:
            if y == 0:
                return -1
            else:
                return 0
        else:
            if y == 0:
                if x < 0:
                    return 1
                else:
                    return -1
            else:
                return -x / math.sqrt(y*y + x*x)


    def _sin_aspect(x, y):
        if y == 0:
            return 0
        else:
            if x == 0:
                if y < 0:
                    return -1
                elif y > 0:
                    return 1
                else:
                    return 0
            else:
                return y / math.sqrt(y*y + x*x)


    def _sin_slope(x, y):
        denom = math.sqrt(x*x + y*y + 1)
        if denom == 0:
            return np.nan
        else:
            return math.sqrt(x*x + y*y) / denom


    def _cos_slope(x, y):
        denom = math.sqrt(x*x + y*y + 1)
        if denom == 0:
            return np.nan
        else:
            return 1/denom


    aspect = np.vectorize(_aspect)
    cos_aspect = np.vectorize(_cos_aspect)
    sin_aspect = np.vectorize(_sin_aspect)
    sin_slope = np.vectorize(_sin_slope)
    cos_slope = np.vectorize(_cos_slope)

    # c = cos_az * cos_aspect(x, y) + sin_az * sin_aspect(x, y)
    c = cos(azimuth - aspect(x, y))
    # v = (cos_z * cos_slope(x, y)) + (sin_z * sin_slope(x, y) * c)
    v = (cos_z * cos(arctan(z_factor * sqrt(x*x + y*y)))) + (sin_z * sin(arctan(z_factor * sqrt(x*x + y*y))) * c)

    return (255.0 * np.maximum(0.0, v)).astype(np.uint8)


with rasterio.open("srtm-30-9_82_178.tif") as src:
    arr = src.read(1)
    dx = src.meta["affine"][0]
    dy = -src.meta["affine"][4]

    profile = src.profile
    profile.update(
        dtype=rasterio.int32,
        predictor=1,
        nodata=None,
    )
    with rasterio.open("np.tif", "w", **profile) as dst:
        start = timer()
        hs_array = hillshade(arr, 315, 45)
        end = timer()
        print "naive hillshade took %dms" % (end - start)
        dst.write(hs_array.astype(rasterio.int32), 1)

    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        predictor=1,
        nodata=None,
    )
    with rasterio.open("gt.tif", "w", **profile) as dst:
        start = timer()
        hs = gt_hillshade(arr, 315, 45, 0.05)
        end = timer()
        print "hillshade took %dms" % (end - start)
        dst.write(hs, 1)

    with rasterio.open("plt.tif", "w", **profile) as dst:
        start = timer()
        ls = LightSource()
        hs = ls.hillshade(arr,
            dx=dx,
            dy=dy,
        )
        hs = (255.0 * hs).astype(np.uint8)
        end = timer()
        print "hillshade took %dms" % (end - start)
        dst.write(hs, 1)
