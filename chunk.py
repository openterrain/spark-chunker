from __future__ import print_function

import errno
import math
import multiprocessing
import os
from urlparse import urlparse

import boto3
import mercantile
import numpy as np
import numpy.ma as ma
import quadtree
import rasterio
from rasterio import crs
from rasterio.transform import from_bounds
from rasterio.warp import (reproject, calculate_default_transform, transform)
from rasterio._io import virtual_file_to_buffer

APP_NAME = "Reproject and chunk"
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


def process_chunk(tile, input, creation_options, resampling="bilinear", out_dir="."):
    """Process a single tile."""

    from rasterio.warp import RESAMPLING

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

    print("Chunking initial image for", tile)

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(
        *mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(
        *mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    tmp_path = "/vsimem/tile"

    with rasterio.drivers():
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
                tile_data = tmp.read(masked=True)
                if tile_data.mask.all():
                    return

                # TODO hard-coded for the first band
                return (tile, tile_data[0])


# NOTE: assumes 1 band
def downsample((tile, data)):
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
            tmp.write(data, 1)
            resampled = tmp.read(1,
                                 masked=True,
                                 out=np.empty((CHUNK_SIZE / 2, CHUNK_SIZE / 2), data.dtype))

            if resampled.mask.all():
                return

            corner = CORNERS[(tile.x % 2, tile.y % 2)]
            return (mercantile.parent(tile), (corner, resampled))


def contains_data(data):
    return data is not None


def z_key(tile):
    if tile.z > 1:
        return quadtree.encode(*mercantile.ul(*tile), precision=tile.z)
    else:
        return ""


def write(creation_options, out_dir):
    def _write((tile, data)):

        print("Writing:", tile)

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

        return (tile, data)

    return _write


def merge((_, out), (tile, (corner, data))):
    (dx, dy) = OFFSETS[corner]

    out[dy:dy + (CHUNK_SIZE / 2), dx:dx + (CHUNK_SIZE / 2)] = data

    return (tile, out)


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


def get_meta(input):
    input = input.replace("s3://", "/vsicurl/http://s3.amazonaws.com/")
    with rasterio.drivers():
        with rasterio.open(input) as src:
            return src.meta


def get_tiles(zoom, input, dst_crs="EPSG:3857"):
    print("getting tiles for", input)
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

    print(outputs)


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

    # repartition tiles so a given task only processes the children of a given
    # tile
    tiles = tiles.map(lambda tile: (quadtree.encode(*mercantile.ul(*tile), precision=tile.z), tile)).sortByKey(numPartitions=tiles.count() / 4)

    chunks = tiles.map(lambda (q,tile): process_chunk(tile, input, creation_options, resampling="bilinear", out_dir=out_dir)).filter(contains_data)

    chunks.map(downsample).filter(contains_data).map(z_key)

    # partitioning isn't ideal here, as empty tiles will have been dropped
    subtiles = subtiles.sortByKey(numPartitions=subtiles.count() / 4)

    # TODO need nodata, dtype
    # TODO deal with multiple bands (probably with flatMapValues)
    subtiles.foldByKey(np.full((CHUNK_SIZE, CHUNK_SIZE), "-1", "float32"), merge).map(write).map(downsample).filter(contains_data).map(z_key)


def main2(sc, input, out_dir):
    zoom = get_zoom(input)
    meta = get_meta(input)
    resampling = "bilinear"

    meta.update({
        "driver": "GTiff",
        "crs": "EPSG:3857",
        "tiled": True,
        "compress": "deflate",
        "sparse_ok": True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    if np.dtype(meta["dtype"]).kind == "f":
        meta["predictor"] = 3
    else:
        meta["predictor"] = 2

    # generate a list of tiles covering input
    tiles = sc.parallelize([input]).flatMap(lambda source: get_tiles(zoom, source)).distinct().cache()

    # repartition tiles so a given task only processes the children of a given
    # tile
    # output: (quadkey, tile)
    tiles = tiles.keyBy(z_key).sortByKey(numPartitions=tiles.count() / 4).cache()

    print("%d tiles to process" % (tiles.count()))

    # chunk initial zoom level and fetch contents
    # output: (quadkey, (tile, ndarray))
    chunks = tiles.mapValues(lambda tile: process_chunk(tile, input, meta, resampling=resampling, out_dir=out_dir)).values().filter(contains_data).cache()

    # write out chunks
    chunks.foreach(write(meta, out_dir))

    print("%d chunks at zoom %d" % (chunks.count(), zoom))

    # TODO deal with multiple bands (probably with flatMapValues)
    for z in range(zoom - 1, -1, -1):
        print("Processing zoom %d" % (z))

        # downsample and re-key according to new tile
        # output: (quadkey, (tile, data))
        subtiles = chunks.map(downsample).filter(contains_data).keyBy(lambda (tile, _): z_key(tile)).cache()

        print("%d subtiles at zoom %d" % (subtiles.count(), z))

        # partitioning isn't ideal here, as empty tiles will have been dropped,
        # unsettling the balance
        # output: (quadkey, (tile, data))
        subtiles = subtiles.sortByKey(numPartitions=max(1, subtiles.count() / 4)).cache()

        # merge subtiles
        # output: (quadkey, (tile, data))
        chunks = subtiles.foldByKey((None, ma.masked_array(data=np.empty((CHUNK_SIZE, CHUNK_SIZE), meta["dtype"]), mask=True, fill_value=meta["nodata"])), merge).values().filter(contains_data).cache()

        # write out chunks
        chunks.foreach(write(meta, out_dir))

if __name__ == "__main__":
    from pyspark import SparkConf, SparkContext

    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)

    # main(sc, "http://s3.amazonaws.com/ned-13arcsec.openterrain.org/4326.vrt", "s3://ned-13arcsec.openterrain.org/3857")
    main2(sc, "imgn19w065_13.tif", "chunks")
    # main(sc, "4326.vrt", "ned")
    # main(sc, "imgn19w065_13.tif", "s3://ned-13arcsec.openterrain.org/3857")

    # chunk("imgn19w065_13.tif", "s3://ned-13arcsec.openterrain.org/3857")
    # chunk("imgn19w065_13.tif", "chunks")
