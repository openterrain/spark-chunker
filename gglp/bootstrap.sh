#!/bin/sh -e

sudo yum-config-manager --enable epel
sudo yum -y install geos proj proj-nad proj-epsg libcurl-devel.x86_64
sudo ln -s /usr/lib64/libproj.so.0 /usr/lib64/libproj.so
aws s3 cp s3://emr.openterrain.org/deps/gdal-1.11.2-amz2.tar.gz - | sudo tar zxf - -C /usr/local
sudo GDAL_CONFIG=/usr/local/bin/gdal-config pip-2.7 install boto3 rasterio==0.29.0 mercantile psutil python-geohash

aws s3 cp s3://emr.openterrain.org/gglp/tiles.txt /tmp/tiles.txt
aws s3 cp s3://emr.openterrain.org/gglp/gglp.vrt /tmp/gglp.vrt
aws s3 cp s3://emr.openterrain.org/gglp/chunk.py /tmp/
aws s3 cp s3://emr.openterrain.org/gglp/pyramid.py /tmp/
