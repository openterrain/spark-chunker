from __future__ import print_function

import errno
import fileinput
import itertools
import json
import math
import multiprocessing
import os
from urlparse import urlparse

import boto3
import mercantile
import numpy as np
import numpy.ma as ma
from pyspark import StorageLevel
import quadtree
import rasterio
from rasterio import crs
from rasterio.transform import from_bounds
from rasterio.warp import (reproject, calculate_default_transform, transform)
from rasterio._io import virtual_file_to_buffer

APP_NAME = "Chunk"
CHUNK_SIZE = 2048 # 4096 produces ~50MB files for NED

CORNERS = {
    (0, 0): "ul",
    (0, 1): "ll",
    (1, 0): "ur",
    (1, 1): "lr",
}

OFFSETS = {
    "ul": (0, 0),
    "ur": (CHUNK_SIZE / 2, 0),
    "ll": (0, CHUNK_SIZE / 2),
    "lr": (CHUNK_SIZE / 2, CHUNK_SIZE / 2),
}


def mkdir_p(dir):
    try:
        os.makedirs(dir)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else: raise


def read_chunk(tile, prefix):
    """Reads a single tile."""

    print("Reading", tile)

    input = prefix.replace("s3://", "/vsicurl/http://s3.amazonaws.com/") + "/%d/%d/%d.tif" % (tile.z, tile.x, tile.y)
    tmp_path = "/vsimem/tile"

    with rasterio.drivers():
        try:
            with rasterio.open(input, "r") as src:
                data = src.read(masked=True)
        except:
            return

    if data.mask.all():
        return

    # TODO hard-coded for the first band
    return (tile, data[0])


# NOTE: assumes 1 band
def downsample((tile, data)):
    if data is None:
        return

    print("Downsampling", tile)

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(
        *mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(
        *mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    # TODO constantize
    tmp_path = "/vsimem/tile"

    # create GeoTIFF
    meta = {
        "driver": "GTiff",
        "crs": "EPSG:3857",
        "nodata": data.fill_value,
        "count": 1,
        "dtype": data.dtype,
        "width": CHUNK_SIZE,
        "height": CHUNK_SIZE,
        "transform": from_bounds(ulx, lry, lrx, uly, CHUNK_SIZE, CHUNK_SIZE),
    }

    with rasterio.drivers():
        with rasterio.open(tmp_path, "w", **meta) as tmp:
            # use GDAL to resample by writing an ndarray and immediately reading
            # it out into a smaller array
            tmp.write(data, 1)
            resampled = tmp.read(
                indexes=1,
                masked=True,
                out=ma.array(np.empty((CHUNK_SIZE / 2, CHUNK_SIZE / 2), data.dtype)),
            )

            if resampled.mask.all():
                return

            corner = CORNERS[(tile.x % 2, tile.y % 2)]
            return (mercantile.parent(tile), (corner, resampled))


def contains_data(data):
    if data is not None:
        (tile, data) = data
        if not isinstance(data, tuple):
            return tile is not None and not data.mask.all()

        return True

    return False


def write(creation_options, out_dir):
    def _write((tile, data)):

        print("Writing", tile)

        # Get the bounds of the tile.
        ulx, uly = mercantile.xy(
            *mercantile.ul(tile.x, tile.y, tile.z))
        lrx, lry = mercantile.xy(
            *mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

        # TODO constantize
        tmp_path = "/vsimem/tile"

        # create GeoTIFF
        meta = creation_options.copy()
        meta["count"] = 1
        meta["nodata"] = data.fill_value
        meta["dtype"] = data.dtype
        meta["width"] = CHUNK_SIZE
        meta["height"] = CHUNK_SIZE
        meta["transform"] = from_bounds(ulx, lry, lrx, uly, CHUNK_SIZE, CHUNK_SIZE)

        with rasterio.drivers():
            with rasterio.open(tmp_path, "w", **meta) as tmp:
                tmp.write(data, 1)

        # write out
        output_uri = urlparse(out_dir)
        contents = bytearray(virtual_file_to_buffer(tmp_path))

        if output_uri.scheme == "s3":
            # TODO use mapPartitions so that the client only needs to be
            # instantiated once per partition
            client = boto3.client("s3")

            bucket = output_uri.netloc
            # TODO strip out trailing slashes on the path if necessary
            key = "%s/%d/%d/%d.tif" % (output_uri.path[1:], tile.z, tile.x, tile.y)

            response = client.put_object(
                ACL="public-read",
                Body=bytes(contents),
                Bucket=bucket,
                # CacheControl="TODO",
                ContentType="image/tiff",
                Key=key
            )
        else:
            output_path = os.path.join(out_dir, "%d/%d/%d.tif" % (tile.z, tile.x, tile.y))
            mkdir_p(os.path.dirname(output_path))

            f = open(output_path, "w")
            f.write(contents)
            f.close()

    return _write


def merge((_, out), (tile, (corner, data))):
    print("Merging corner %s for tile %s" % (corner, tile))
    (dx, dy) = OFFSETS[corner]

    out[dy:dy + (CHUNK_SIZE / 2), dx:dx + (CHUNK_SIZE / 2)] = data

    return (tile, out)


def pyramid(sc, zoom, dtype, nodata, tiles, prefix, resampling="average"):
    meta = dict(
        driver="GTiff",
        crs="EPSG:3857",
        tiled=True,
        compress="deflate",
        predictor=2,
        sparse_ok=True,
        nodata=nodata,
        dtype=dtype,
        blockxsize=256,
        blockysize=256,
    )

    if np.dtype(dtype).kind == "f":
        meta["predictor"] = 3

    empty = ma.masked_array(np.full((CHUNK_SIZE, CHUNK_SIZE), nodata, dtype), fill_value=nodata)

    tile_count = tiles.count()

    print("%d tiles to process" % (tile_count))

    # TODO deal with multiple bands (probably with flatMapValues)
    for z in range(zoom - 1, -1, -1):
        print("Processing zoom %d" % (z))
        tile_count = max(tile_count / 4, 1)

        print("Tile count: %d" % (tile_count))

        #  downsample its children
        #  merge them
        #  write out the result

        # generate a list of tiles at the current zoom (from available children)
        tiles = tiles.map(
            lambda child: mercantile.parent(child)
        ).distinct()

        tiles.map(
            # for each parent tile:
            lambda parent: reduce(
                # 3. merge
                merge,
                # 2. downsample them
                filter(
                    None,
                    itertools.imap(
                        downsample,
                        # 1. fetch children
                        filter(
                            None,
                            itertools.imap(
                                lambda tile: read_chunk(tile, prefix),
                                mercantile.children(parent)
                            )
                        )
                    )
                ),
                (None, empty)
            )
        ).filter(
            contains_data
        ).foreach(
            write(meta, prefix)
        )


if __name__ == "__main__":
    from pyspark import SparkConf, SparkContext

    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)

    # input = "/Users/seth/src/openterrain/spark-chunker/imgn19w065_13.tif"

    # zoom = get_zoom(input)
    zoom = 11
    meta = {
        "dtype": "float32",
        "nodata": -3.4028234663852886e+38,
    }
    # meta = get_meta(input)

    # TODO pull zoom, dtype, nodata, input, out_dir using argparse

    # TODO fileinput.input() supposedly reads from STDIN, except where it doesn't
    tiles = sc.parallelize(itertools.imap(lambda line: mercantile.Tile(*json.loads(line)), fileinput.input())).distinct()

    pyramid(sc,
        zoom=zoom,
        dtype=meta["dtype"],
        nodata=meta["nodata"],
        tiles=tiles,
        prefix="s3://ned-13arcsec.openterrain.org/3857",
    )
