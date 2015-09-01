import multiprocessing
import os
from urlparse import urlparse

import boto3
import rasterio
from rasterio import crs
from rasterio.warp import (reproject, RESAMPLING, calculate_default_transform)
from rasterio._io import virtual_file_to_buffer

from pyspark import SparkConf, SparkContext

APP_NAME = "SRTM 90m conversion"

"""
spark-submit --master local[*] app.py
"""

def convert(input, output):
    """
    Intended for conversion from whatever the source format is to matching
    filenames containing 4326 data, etc.
    """
    resampling = "bilinear"
    driver = "GTiff"
    dst_crs = "EPSG:4326"
    threads = multiprocessing.cpu_count() / 2
    creation_options = {
        "tiled": True,
        "compress": "deflate",
        "predictor": 2,
        "sparse_ok": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    input = input.replace("s3://", "/vsicurl/http://s3.amazonaws.com/")
    uri = urlparse(output)

    input_uri = urlparse(input)

    input = "/vsizip%s/%s" % (input, os.path.basename(input_uri.path).replace("zip", "tif"))

    print input


    resampling = getattr(RESAMPLING, resampling)

    with rasterio.drivers():
        with rasterio.open(input) as src:
            out_kwargs = src.meta.copy()
            out_kwargs["driver"] = driver

            dst_crs = crs.from_string(dst_crs)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=None)

            out_kwargs.update({
                "crs": dst_crs,
                "transform": dst_transform,
                "affine": dst_transform,
                "width": dst_width,
                "height": dst_height
            })

            out_kwargs.update(**creation_options)

            with rasterio.open("/vsimem/img", "w", **out_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.affine,
                        src_crs=src.crs,
                        dst_transform=out_kwargs["transform"],
                        dst_crs=out_kwargs["crs"],
                        resampling=resampling,
                        num_threads=threads)

            contents = bytearray(virtual_file_to_buffer("/vsimem/img"))

            if uri.scheme == "s3":
                client = boto3.client("s3")

                response = client.put_object(
                    ACL="public-read",
                    Body=bytes(contents),
                    Bucket=uri.netloc,
                    # CacheControl="TODO",
                    ContentType="image/tiff",
                    Key=uri.path[1:]
                )
            else:
                f = open(output, "w")
                f.write(contents)
                f.close()

            return output

def main(sc):
    client = boto3.client("s3")

    bucket = "cgiar-csi-srtm.openterrain.org"

    paginator = client.get_paginator("list_objects")
    source_pages = paginator.paginate(Bucket=bucket, Prefix="source/")
    target_pages = paginator.paginate(Bucket=bucket, Prefix="4326/")

    existing_targets = reduce(
        lambda a,b: a + b,
        map(
            lambda page: map(
                lambda o: "s3://%s/%s" % (bucket, o["Key"]),
                page["Contents"]
            ),
            target_pages),
        [])

    sources = filter(
        lambda x: "zip" in x and x.replace("source", "4326").replace(".zip", ".tif") not in existing_targets,
        reduce(
            lambda a,b: a + b,
            map(
                lambda page: map(
                    lambda o: "s3://%s/%s" % (bucket, o["Key"]),
                    page["Contents"]
                ),
                source_pages),
            []))

    # sources = sources[:1]

    # manually set the number of partitions to avoid large variance in loads
    # (create more partitions than Spark would otherwise)
    sc.parallelize(sources).map(
        lambda src: convert(src, src.replace("source", "4326").replace(".zip", ".tif"))
    ).collect()


if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)

    main(sc)
