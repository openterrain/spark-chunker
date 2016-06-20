import copy

from matplotlib.colors import LightSource
import mercantile
import numpy as np
import rasterio

BUFFER = 2
SRC_TILE_ZOOM = 14
SRC_TILE_WIDTH = 512
SRC_TILE_HEIGHT = 512
tile = mercantile.Tile(82, 178, 9)
tile = mercantile.Tile(5252, 11446, 15)

DST_TILE_WIDTH = 256
DST_TILE_HEIGHT = 256
DST_BLOCK_SIZE = 256

# calculate these in z14 pixels
dz = SRC_TILE_ZOOM - tile.z
x = int(2**dz * tile.x)
y = int(2**dz * tile.y)
mx = 2**dz * (tile.x + 1)
my = 2**dz * (tile.y + 1)
dx = mx - x
dy = my - y
top = (2**SRC_TILE_ZOOM * SRC_TILE_HEIGHT) - 1

# y, x
window = [
          [
           top - (top - (SRC_TILE_HEIGHT * y)),
           top - (top - ((SRC_TILE_HEIGHT * y) + int(SRC_TILE_HEIGHT * dy)))
          ],
          [
           SRC_TILE_WIDTH * x,
           (SRC_TILE_WIDTH * x) + int(SRC_TILE_WIDTH * dx)
          ]
         ]

buffered_window = copy.deepcopy(window)

# buffer so we have neighboring pixels
buffered_window[0][0] -= BUFFER
buffered_window[0][1] += BUFFER
buffered_window[1][0] -= BUFFER
buffered_window[1][1] += BUFFER

with rasterio.open("mapzen.xml") as src:
    data = src.read(1, window=buffered_window, masked=True)
    dx = src.meta["affine"][0]
    dy = -src.meta["affine"][4]

    meta = src.meta.copy()
    del meta["transform"]
    meta.update(
        driver='GTiff',
        height=buffered_window[0][1] - buffered_window[0][0],
        width=buffered_window[1][1] - buffered_window[1][0],
        affine=src.window_transform(buffered_window),
    )
    with rasterio.open("windowed_src.tif", "w", **meta) as dst:
        dst.write(data, 1)

    meta = src.meta.copy()
    del meta["transform"]

    meta.update(
        driver='GTiff',
        dtype=rasterio.uint8,
        compress="deflate",
        predictor=1,
        nodata=None,
        tiled=True,
        sparse_ok=True,
        blockxsize=DST_BLOCK_SIZE,
        blockysize=DST_BLOCK_SIZE,
        height=DST_TILE_HEIGHT,
        width=DST_TILE_WIDTH,
    )

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(*mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(*mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    meta["affine"] = src.window_transform(window)

    ls = LightSource()
    hs = ls.hillshade(data,
        dx=dx,
        dy=dy,
    )
    hs = (255.0 * hs).astype(np.uint8)

    with rasterio.open("windowed.tif", "w", **meta) as dst:
        # ignore the border pixels when writing
        dst.write(hs[BUFFER:-BUFFER, BUFFER:-BUFFER], 1)
