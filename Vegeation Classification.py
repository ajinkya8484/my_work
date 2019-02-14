import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

filename = '/home/ajinkya/PycharmProjects/vasundharaa/Change Detetction/clipped/Clip_Clip_347_2012 comp11.tif'
with rasterio.open(filename) as src:
    band_red = src.read(2)
with rasterio.open(filename) as src:
    band_nir = src.read(3)
with rasterio.open(filename) as src:
    band_green = src.read(1)


# Do not display error when divided by zero
np.seterr(divide='ignore', invalid='ignore')

print(np.argwhere(np.isnan(band_red)))
print(np.argwhere(np.isnan(band_nir)))



print(band_nir)
print(band_red)

# NDVI
#ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir.astype(float) + band_red.astype(float))
ndvi = (band_red.astype(float) - band_nir.astype(float)) / (band_red.astype(float) + band_nir.astype(float))
print(np.argwhere(np.isinf(ndvi)))
print(np.nanmin(ndvi))
print(np.nanmax(ndvi))
print(np.median(ndvi))

# NDWI
ndwi = (band_green.astype(float) - band_nir.astype(float)) / (band_green.astype(float) + band_nir.astype(float))

# get the metadata of original GeoTIFF:
meta = src.meta
print(meta)

# get the dtype of our NDVI array:
ndvi_dtype = ndvi.dtype
print(ndvi_dtype)

# set the source metadata as kwargs we'll use to write the new data:
kwargs = meta

# update the 'dtype' value to match our NDVI array's dtype:
kwargs.update(dtype=ndvi_dtype)

# update the 'count' value since our output will no longer be a 4-band image:
kwargs.update(count=1)

# Finally, use rasterio to write new raster file 'data/ndvi.tif':
with rasterio.open('ndbi_2.tif', 'w', **kwargs) as dst:
    dst.write(ndvi, 1)


for i in range(ndvi.shape[0]):
    for j in range(ndvi.shape[1]):
        if ndvi[i, j] < 0.1 or ndwi[i, j] > -0.1:
            ndvi[i, j] = 0
        elif ndvi[i, j] < 0.3:
            ndvi[i, j] = 128
        else:
            ndvi[i, j] = 256


# Finally, use rasterio to write new raster file 'data/ndvi.tif':
with rasterio.open('ndvi_classes.tif', 'w', **kwargs) as dst:
    dst.write(ndvi, 1)

class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# Set min/max values from NDVI range for image
min = np.nanmin(ndvi)
max = np.nanmax(ndvi)

# Set our custom midpoint for most effective NDVI analysis
mid = 0.1

# Setting color scheme ref:https://matplotlib.org/users/colormaps.html as a reference
colormap = plt.cm.RdYlGn
norm = MidpointNormalize(vmin=min, vmax=max, midpoint=mid)
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(111)

# Use 'imshow' to specify the input data, colormap, min, max, and norm for the colorbar
cbar_plot = ax.imshow(ndvi, cmap=colormap, vmin=min, vmax=max, norm=norm)

# Turn off the display of axis labels
ax.axis('off')

# Set a title
ax.set_title('(NDVI) Normalized Difference Vegetation Index', fontsize=17, fontweight='bold')

# Configure the colorbar
# cbar = fig.colorbar(cbar_plot, orientation='horizontal', shrink=0.65)

# Call 'savefig' to save this plot to an image file
fig.savefig("ndvi-image.png", dpi=200, bbox_inches='tight', pad_inches=0.7)

# let's visualize
plt.show()
