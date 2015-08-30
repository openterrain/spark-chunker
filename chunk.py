import errno
import math
import multiprocessing
import os
from urlparse import urlparse

import boto3
import mercantile
import numpy
import rasterio
from rasterio import crs
from rasterio.transform import from_bounds
from rasterio.warp import (reproject, RESAMPLING, calculate_default_transform, transform)
from rasterio._io import virtual_file_to_buffer

APP_NAME = "Reproject and chunk"
CHUNK_SIZE = 512


def mkdir_p(dir):
    try:
        os.makedirs(dir)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else: raise


def process_tile(tile, base_kwds, src):
    """Process a single tile."""

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(
        *mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(
        *mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    kwds = base_kwds.copy()
    kwds["transform"] = from_bounds(ulx, lry, lrx, uly, CHUNK_SIZE, CHUNK_SIZE)

    tmp_path = "/vsimem/%d/%d/%d" % (tile.z, tile.x, tile.y)

    with rasterio.open(tmp_path, "w", **kwds) as tmp:
        # Reproject the src dataset into image tile.
        for bidx in src.indexes:
            reproject(
                rasterio.band(src, bidx),
                rasterio.band(tmp, bidx))

        tile_data = tmp.read()
        if tile_data.all() and tile_data[0][0][0] == src.nodata:
            return

    return tmp_path


def chunk(input, out_dir):
    """
    Intended for conversion from whatever the source format is to matching
    filenames containing 4326 data, etc.
    """
    resampling = "bilinear"
    driver = "GTiff"
    dst_crs = "EPSG:3857"
    threads = multiprocessing.cpu_count() / 2
    creation_options = {
        "tiled": True,
        "compress": "deflate",
        "predictor":   2, # 3 for floats, 2 otherwise
        "sparse_ok": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # NOTE: not using vsicurl, as the entire file needs to be read anyway
    input = input.replace("s3://", "http://s3.amazonaws.com/")
    input_uri = urlparse(input)
    base_name = os.path.splitext(os.path.basename(input_uri.path))[0]


    resampling = getattr(RESAMPLING, resampling)

    with rasterio.drivers():
        with rasterio.open(input) as src:
            int_kwargs = src.meta.copy()
            int_kwargs["driver"] = driver
            int_kwargs["crs"] = dst_crs
            int_kwargs["height"] = CHUNK_SIZE
            int_kwargs["width"] = CHUNK_SIZE
            int_kwargs.update(**creation_options)

            # Compute the geographic bounding box of the dataset.
            (west, east), (south, north) = transform(
                src.crs, "EPSG:4326", src.bounds[::2], src.bounds[1::2])

            affine, _, _ = calculate_default_transform(src.crs, dst_crs,
                src.width, src.height, *src.bounds, resolution=None)

            # grab the lowest resolution dimension
            resolution = max(abs(affine[0]), abs(affine[4]))

            zoom = int(round(math.log((2 * math.pi * 6378137) /
                                      (resolution * CHUNK_SIZE)) / math.log(2)))

            print "target zoom", zoom

            # Initialize iterator over output tiles.
            # TODO process a band at a time
            tiles = mercantile.tiles(
                west, south, east, north, range(zoom, zoom + 1))

            for tile in tiles:
                tmp_path = process_tile(tile, int_kwargs, src)

                if tmp_path is None:
                    continue

                contents = bytearray(virtual_file_to_buffer(tmp_path))

                # if uri.scheme == "s3":
                #     client = boto3.client("s3")
                #
                #     response = client.put_object(
                #         ACL="public-read",
                #         Body=bytes(contents),
                #         Bucket=uri.netloc,
                #         # CacheControl="TODO",
                #         ContentType="image/tiff",
                #         Key=uri.path[1:]
                #     )
                # else:
                output = os.path.join(out_dir, "%d/%d/%d.tif" % (tile.z, tile.x, tile.y))
                mkdir_p(os.path.dirname(output))

                f = open(output, "w")
                f.write(contents)
                f.close()


def main(sc):
    pass


if __name__ == "__main__":
    # from pyspark import SparkConf, SparkContext
    #
    # conf = SparkConf().setAppName(APP_NAME)
    # sc = SparkContext(conf=conf)
    #
    # main(sc)
    chunk("imgn19w065_13.tif", "chunks")
