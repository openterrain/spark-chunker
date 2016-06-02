# Golden Gate LiDAR Project

## Step 1: copy data from original source to S3

Present in `s3://gglp.openterrain.org/source`

## Step 2: convert source data to GeoTIFF

(matching filenames + sizes to original dataset)

Already in GeoTIFF format (UTM zone 10N), e.g.:

```
$ gdalinfo /vsicurl/http://gglp.openterrain.org.s3.amazonaws.com/source/00500350_dem.tif
Driver: GTiff/GeoTIFF
Files: /vsicurl/http://gglp.openterrain.org.s3.amazonaws.com/source/00500350_dem.tif
       /vsicurl/http://gglp.openterrain.org.s3.amazonaws.com/source/00500350_dem.tif.ovr
       /vsicurl/http://gglp.openterrain.org.s3.amazonaws.com/source/00500350_dem.tif.aux.xml
Size is 1500, 1500
Coordinate System is:
PROJCS["NAD83 / UTM zone 10N",
    GEOGCS["NAD83",
        DATUM["North_American_Datum_1983",
            SPHEROID["GRS 1980",6378137,298.2572221010002,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6269"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433],
        AUTHORITY["EPSG","4269"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-123],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AUTHORITY["EPSG","26910"]]
Origin = (500500.000000000000000,4205000.000000000000000)
Pixel Size = (1.000000000000000,-1.000000000000000)
Metadata:
  AREA_OR_POINT=Area
Image Structure Metadata:
  INTERLEAVE=BAND
Corner Coordinates:
Upper Left  (  500500.000, 4205000.000) (122d59'39.50"W, 37d59'33.55"N)
Lower Left  (  500500.000, 4203500.000) (122d59'39.50"W, 37d58'44.89"N)
Upper Right (  502000.000, 4205000.000) (122d58'38.00"W, 37d59'33.55"N)
Lower Right (  502000.000, 4203500.000) (122d58'38.02"W, 37d58'44.88"N)
Center      (  501250.000, 4204250.000) (122d59' 8.76"W, 37d59' 9.22"N)
Band 1 Block=1500x1 Type=Float32, ColorInterp=Gray
  Min=0.000 Max=34.430
  Minimum=0.000, Maximum=34.430, Mean=0.029, StdDev=0.679
  Overviews: 750x750, 375x375, 188x188
  Metadata:
    STATISTICS_MAXIMUM=34.430000305176
    STATISTICS_MEAN=0.028501311677273
    STATISTICS_MINIMUM=0
    STATISTICS_STDDEV=0.67924124775851
```

Note that the block size is `1500x1`, which means that each tile will need to re-fetch each source
file it contains. This means that we'd be better off rewriting the TIFFs (even without
reprojecting).

## Step 3: create a tile index

Create a tile index with the bounding box of each constituent file. This will be
used to determine which spherical Mercator tiles need to be created.

(Best to do this on EC2 to be closer to the data, but it's not that big a deal as only metadata is
read.)

```bash
aws s3 ls s3://gglp.openterrain.org/source/ | \
  grep -e "\.tif$" | \
  awk '{print "/vsicurl/http://s3.amazonaws.com/gglp.openterrain.org/source/" $4}' | \
  xargs gdaltindex gglp.shp
```

In GeoJSON:

```bash
ogr2ogr -F GeoJSON -t_srs EPSG:4326 gglp.json gglp.shp
```

http://geojson.io/#id=gist:anonymous/842e35a7b6e18ae9a6889c744c2fb6d2&map=10/37.9052/-122.6467

## Step 4: create VRTs (using VSICurl) pointing to converted source data

Create a VRT containing all constituent files. This is used when chunking, as it
allows files that share the same tile to be simultaneously chunked, rather than
being chunked to the same tile twice.

(Best to do this on EC2 to be closer to the data.)

```bash
gdalbuildvrt -resolution highest gglp.vrt gglp.shp
```

## Step 5: determine optimal zoom

```bash
$ python get_zoom.py
14
```

Edit `chunk.py` to set zoom and paths.

## Step 6: generate a list of tiles to chunk out

Using the GeoJSON index and `mercantile`, we can produce a list of tiles at our
target zoom (determined with the help of `get_zoom()` in `chunk.py`, as it's
dependent on the source resolution and the target chunk size):

```bash
jq -rc .features[] gglp.json | mercantile tiles 14 > tiles.txt
```

## Step 7: reproject + chunk source data (`chunk.py`)

This will reprojectt + chunk source data into 2048x2048 GeoTIFFs at the closest zoom to the original
resolution.

## Step 8: pyramid data down to zoom 0 (`pyramid.py`)

Steps 7 and 8 can be run together using:

```bash
./run-emr.sh
```

## Step 9: create a TMS mini-driver config

See `gglp.xml`.

Spark is just used for its "map" capabilities--only information about the tile coordinates being processed is passed between nodes. S3 is used for source + output data.

I was able to process 10m data for the US on a 10 node cluster in under 2 hours (probably better now that I've figured out how to deal with stragglers).

The TMS contains 69,899 files (with files containing NODATA dropped) spanning 454.7 GiB (the source data was ~200GiB). At 2048x2048 Float32 tiles (vs. Int32 so we can get sub-meter vertical resolution), each tile is < 10MB (but still large enough to make sense on S3).

## Usage (local)

### Installation

```bash
brew install apache-spark
pip install -r requirements.txt
```

### Running

```bash
spark-submit chunk.py
```

Create a tile index with the bounding box of each constituent file. This will be
used to determine which spherical Mercator tiles need to be created.

```bash
aws s3 ls ned-13arcsec.openterrain.org/4326/ | \
  awk '{print "/vsicurl/http://s3.amazonaws.com/ned-13arcsec.openterrain.org/4326/" $4}' | \
  xargs gdaltindex ned-13arcsec.shp
```

It's also handy to have around as GeoJSON (we could have created it as GeoJSON,
but it's suitable as input for `gdalbuildvrt`, so we'll keep the Shapefile
around):

```bash
ogr2ogr -F GeoJSON ned-13arcsec.json ned-13arcsec.shp
```

Create a VRT containing all constituent files. This is used when chunking, as it
allows files that share the same tile to be simultaneously chunked, rather than
being chunked to the same tile twice.

```bash
gdalbuildvrt -resolution highest ned-13arcsec.vrt ned-13arcsec.shp
```

The VRT needs to be manually tweaked to set the `NoDataValue` for the entire
group to be the same as individual `NODATA` values.

Using the GeoJSON index and `mercantile`, we can produce a list of tiles at our
target zoom (determined with the help of `get_zoom()` in `chunk.py`, as it's
dependent on the source resolution and the target chunk size):

```bash
jq -rc .features[] ned-13arcsec.json | mercantile tiles 11 > tiles.txt
```

Then, to actually run the chunking task:

```bash
spark-submit chunk.py tiles.txt
```
