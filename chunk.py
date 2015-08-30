import errno
import math
import multiprocessing
import os
from urlparse import urlparse

import boto3
import numpy
import rasterio
from rasterio import crs
from rasterio.warp import (reproject, RESAMPLING, calculate_default_transform)
# from rasterio._io import virtual_file_to_buffer

APP_NAME = "Reproject and chunk"
CHUNK_SIZE = 512


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

    # TODO only do this when writing locally
    try:
        os.makedirs(out_dir)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else: raise


    resampling = getattr(RESAMPLING, resampling)

    with rasterio.drivers():
        with rasterio.open(input) as src:
            int_kwargs = src.meta.copy()
            int_kwargs["driver"] = driver

            # TODO calculate resolution according to nearest zoom

            dst_crs = crs.from_string(dst_crs)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=None)

            int_kwargs.update({
                "crs": dst_crs,
                "transform": dst_transform,
                "affine": dst_transform,
                "width": dst_width,
                "height": dst_height
            })

            int_kwargs.update(**creation_options)

            # TODO use less memory by only processing a band at a time,
            # memoizing window information
            with rasterio.open("/vsimem/img", "w", **int_kwargs) as interim:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(interim, i),
                        src_transform=src.affine,
                        src_crs=src.crs,
                        dst_transform=int_kwargs["transform"],
                        dst_crs=int_kwargs["crs"],
                        resampling=resampling,
                        num_threads=threads)

            with rasterio.open("/vsimem/img", "r") as interim:
                cols = interim.meta["width"]
                rows = interim.meta["height"]

                # determine the number of tiles necessary for coverage
                tile_cols = int(math.ceil(cols / float(CHUNK_SIZE)))
                tile_rows = int(math.ceil(rows / float(CHUNK_SIZE)))

                # create windows
                # TODO turn this into a generator
                windows = {}
                for tile_row in xrange(0, tile_rows):
                    windows[tile_row] = {}

                    start = tile_row * CHUNK_SIZE
                    row_window = (start, start + CHUNK_SIZE)

                    for tile_col in xrange(0, tile_cols):
                        start = tile_col * CHUNK_SIZE
                        col_window = (start, start + CHUNK_SIZE)
                        window = (row_window, col_window)
                        windows[tile_row][tile_col] = window

                # Logic for doing extent windowing.
                affine = interim.affine
                (xmin, ymin) = affine * (0, 0)

                # globally align chunks
                # TODO globally aligned chunks are only useful at fixed
                # resolutions
                origin = ~interim.affine * (0, 0)
                xoffset = CHUNK_SIZE - (int(math.floor(origin[0]))
                                        % CHUNK_SIZE)
                yoffset = CHUNK_SIZE - (int(math.floor(origin[1]))
                                        % CHUNK_SIZE)


                def get_affine(col, row):
                    (tx, ty) = affine * (col, row)
                    ta = affine.translation(tx - xmin, ty - ymin) * affine
                    # (ntx, nty) = ta * (0, 0)
                    (ntx, nty) = ta * (-xoffset, -yoffset)

                    return ta


                for i in range(1, interim.count + 1):
                    for tile_row in xrange(0, tile_rows):
                        for tile_col in xrange(0, tile_cols):
                            read_window = windows[tile_row][tile_col]

                            tile_data = interim.read(i, window=read_window)

                            # skip tiles that are all NODATA
                            if numpy.all(tile_data) and tile_data[0][0] == src.nodata:
                                continue

                            (data_rows, data_cols) = tile_data.shape

                            name = "%s-%d_%d_%d" % (base_name, i, tile_col,
                                                    tile_row)

                            tile_affine = get_affine(tile_col * CHUNK_SIZE,
                                                     tile_row * CHUNK_SIZE)

                            tile_meta = interim.meta.copy()
                            tile_meta.update(creation_options)
                            tile_meta.update({
                                "height":      data_rows,
                                "width":       data_cols,
                                "count":       1,
                                "transform":   tile_affine,
                            })

                            with rasterio.open(os.path.join(out_dir, name
                                                            + ".tif"), "w",
                                               **tile_meta) as dst:
                                dst.write(tile_data, 1)

            # contents = bytearray(virtual_file_to_buffer("/vsimem/img"))
            #
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
            #     f = open(output, "w")
            #     f.write(contents)
            #     f.close()
            #
            # return output


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
