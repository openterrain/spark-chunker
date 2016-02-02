#!/bin/sh

while read line; do
  dir=$(mktemp -d)
  basename=$(basename $line)
  filename=${basename%.*}

  curl $line -o ${dir}/${basename}

  gdalwarp \
    -t_srs EPSG:4326 \
    -multi \
    -wm 256 \
    -wo NUM_THREADS=ALL_CPUS \
    -co tiled=yes \
    -co compress=deflate \
    -co predictor=3 \
    -co sparse_ok=true \
    -co blockxsize=256 \
    -co blockysize=256 \
    -r average \
    ${dir}/${basename} \
    ${dir}/${filename}.tif

  aws s3 cp \
    ${dir}/${filename}.tif \
    s3://ned-13arcsec.openterrain.org/4326/${filename}.tif \
    --acl public-read \
    --content-type image/tiff \
    --cache-control "public, max-age=2592000"

  rm ${dir}/${basename}
  rm ${dir}/${filename}.tif
done < "${1:-/dev/stdin}"
