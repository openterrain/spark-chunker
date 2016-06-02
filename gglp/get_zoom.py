import math

import rasterio
from rasterio.warp import (calculate_default_transform, transform)


CHUNK_SIZE = 2048 # 4096 produces ~50MB files for NED


def get_zoom(input, dst_crs="EPSG:3857"):
    input = input.replace("s3://", "/vsicurl/http://s3.amazonaws.com/")
    with rasterio.drivers():
        with rasterio.open(input) as src:
            # Compute the geographic bounding box of the dataset.
            (west, east), (south, north) = transform(
                src.crs, "EPSG:4326", src.bounds[::2], src.bounds[1::2])

            affine, _, _ = calculate_default_transform(src.crs, dst_crs,
                src.width, src.height, *src.bounds, resolution=None)

            # grab the lowest resolution dimension
            resolution = max(abs(affine[0]), abs(affine[4]))

            return int(round(math.log((2 * math.pi * 6378137) /
                                      (resolution * CHUNK_SIZE)) / math.log(2)))


print get_zoom("s3://gglp.openterrain.org/source/45506450_dem.tif")
