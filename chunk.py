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
CHUNK_SIZE = 4096


def mkdir_p(dir):
    try:
        os.makedirs(dir)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else: raise


def process_chunk(tile, input, creation_options, resampling="near", out_dir="."):
    """Process a single tile."""

    print tile

    input = input.replace("s3://", "/vsicurl/http://s3.amazonaws.com/")
    input_uri = urlparse(input)

    if input_uri.scheme == "s3":
        client = boto.client("s3")

        bucket = input_uri.netloc
        key = input_uri.path[1:]

        response = client.head_object(
            Bucket=bucket,
            Prefix=key
        )

        if response.get("Contents") is not None:
            return

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(
        *mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(
        *mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    tmp_path = "/vsimem/tile"

    with rasterio.open(input, "r") as src:
        meta = src.meta.copy()
        meta.update(creation_options)
        meta["height"] = CHUNK_SIZE
        meta["width"] = CHUNK_SIZE
        meta["transform"] = from_bounds(ulx, lry, lrx, uly, CHUNK_SIZE, CHUNK_SIZE)

        with rasterio.open(tmp_path, "w", **meta) as tmp:
            # Reproject the src dataset into image tile.
            for bidx in src.indexes:
                reproject(
                    source=rasterio.band(src, bidx),
                    destination=rasterio.band(tmp, bidx),
                    resampling=getattr(RESAMPLING, resampling),
                    num_threads=multiprocessing.cpu_count() / 2,
                )

            # check for chunks containing only NODATA
            tile_data = tmp.read()
            if tile_data.all() and tile_data[0][0][0] == src.nodata:
                return

    output_uri = urlparse(out_dir)
    contents = bytearray(virtual_file_to_buffer(tmp_path))

    if output_uri.scheme == "s3":
        client = boto3.client("s3")

        bucket = output_uri.netloc
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


def get_tiles(zoom, input, dst_crs="EPSG:3857"):
    print "getting tiles for", input
    input = input.replace("s3://", "/vsicurl/http://s3.amazonaws.com/")
    with rasterio.drivers():
        with rasterio.open(input) as src:
            # Compute the geographic bounding box of the dataset.
            (west, east), (south, north) = transform(
                src.crs, "EPSG:4326", src.bounds[::2], src.bounds[1::2])

            # Initialize an iterator over output tiles.
            return mercantile.tiles(
                west, south, east, north, range(zoom, zoom + 1))


def chunk(input, out_dir):
    """
    Intended for conversion from whatever the source format is to matching
    filenames containing 4326 data, etc.
    """
    resampling = "bilinear"

    creation_options = {
        "driver": "GTiff",
        "crs": "EPSG:3857",
        "tiled": True,
        "compress": "deflate",
        "predictor":   3, # 3 for floats, 2 otherwise
        "sparse_ok": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    tiles = get_tiles(input, dst_crs=creation_options["crs"])

    outputs = [process_chunk(tile, input, creation_options, resampling=resampling, out_dir=out_dir) for tile in tiles]

    outputs = filter(lambda x: x is not None, outputs)

    print outputs


def main(sc, input, out_dir):
    creation_options = {
        "driver": "GTiff",
        "crs": "EPSG:3857",
        "tiled": True,
        "compress": "deflate",
        "predictor":   3, # 3 for floats, 2 otherwise
        "sparse_ok": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    zoom = get_zoom(input)

    client = boto3.client("s3")

    paginator = client.get_paginator("list_objects")
    source_pages = paginator.paginate(Bucket="ned-13arcsec.openterrain.org", Prefix="4326/")

    tiles = sc.parallelize(source_pages).flatMap(lambda page: page["Contents"]).map(lambda item: "s3://ned-13arcsec.openterrain.org/" + item["Key"]).repartition(sc.defaultParallelism).flatMap(lambda source: get_tiles(zoom, source)).distinct().cache()

    tiles.foreach(lambda tile: process_chunk(tile, input, creation_options, resampling="bilinear", out_dir=out_dir))


if __name__ == "__main__":
    from pyspark import SparkConf, SparkContext

    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)

    main(sc, "http://s3.amazonaws.com/ned-13arcsec.openterrain.org/4326.vrt", "s3://ned-13arcsec.openterrain.org/3857")
    # main(sc, "4326.vrt", "ned")
    # main(sc, "imgn19w065_13.tif", "s3://ned-13arcsec.openterrain.org/3857")

    # chunk("imgn19w065_13.tif", "s3://ned-13arcsec.openterrain.org/3857")
    # chunk("imgn19w065_13.tif", "chunks")
