# spark-chunker

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
