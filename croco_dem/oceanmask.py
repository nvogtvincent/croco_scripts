# -*- coding: utf-8 -*-
"""
Apply an ocean mask to an unmasked ocean DEM (run after demgen.py)
Noam Vogt-Vincent (2020)
"""

##############################################################################
# Import Modules #############################################################
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ct
import cmocean.cm as cm
import gdal

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset

##############################################################################
# Global Variables ###########################################################
##############################################################################

source_x = ('/home/noam/Documents/DEMgen/files/WIO_X.npy')
source_y = ('/home/noam/Documents/DEMgen/files/WIO_Y.npy')
source_z = ('/home/noam/Documents/DEMgen/files/WIO_Z_prelim.npy')

gshhg_source = ('/home/noam/Documents/DEMgen/files/MASK2019.tif')

z0 = -25   # Minimum depth (m)
z1 = -50  # Adjustment threshold (m)
z2 = 1    # Minimum elevation (m)

# Plotting variables
lon_min = 36
lon_max = 68
lat_min = -23
lat_max = 2

# Where to export data
gebco_write = ('/home/noam/Documents/DEMgen/files/GEBCO2019_DEMgen.nc')

##############################################################################
# Import & Format Data #######################################################
##############################################################################

# Import DEM
x_in = np.load(source_x)
y_in = np.load(source_y)
z_in = np.load(source_z)

# Import GSHHG raster
# 0 = ocean, 1 = land
mask_object = gdal.Open(gshhg_source)
gshhg = mask_object.ReadAsArray()
gshhg = np.flip(gshhg, axis=0)

##############################################################################
# Main Script ################################################################
##############################################################################

# This script applies an ocean mask to an inadequately masked DEM
# (e.g. GEBCO2019)

# We define two depths, z0 and z1.
# z0 = minimum depth (ALL ocean cells are deeper than this)
# z1 = masking threshold (script will act on all cells shallower than this)
# For all cells within the water mask (i.e. known water cells), apply the
# following transformation:
# z(out) = z1 + (z0-z1)tanh((z(in)-z1)/(z0-z1))
# This has the effect of smoothly nudging all depths above z1 towards the
# minimum depth z0

# Copy z_gebco to generate an array that only applies to cells which are (1)
# ocean cells (defined through GSHHG) and (2) shallower than z1
z_ocean = np.ma.masked_where(gshhg == 1, np.copy(z_in))
z_ocean = np.ma.masked_where(z_ocean < z1, z_ocean)
z_ocean = z1 + (z0-z1)*np.tanh((z_ocean - z1)/(z0-z1))

# Now splice this with the unmodified data
# Do this by masking z with the inverse of z_ocean, setting the masks to zero
# and adding
z_mask = ~np.ma.getmask(z_ocean)
z = np.ma.array(np.copy(z_in), mask=z_mask)

z = np.ma.filled(z, 0)
z_ocean = np.ma.filled(z_ocean, 0)

z = z + z_ocean

# Now adjust all land cells that are below 1m elevation to 1m elevation
z_land = np.ma.masked_where(gshhg == 0, np.copy(z))
z_land = np.ma.masked_where(z_land > z2, z_land)
z_land = z_land*0 + z2

# Now splice this with the ocean-corrected data, as above
z_mask = ~np.ma.getmask(z_land)
z = np.ma.array(np.copy(z), mask=z_mask)

z = np.ma.filled(z, 0)
z_land = np.ma.filled(z_ocean, 0)

z = z + z_land

# Save the coordinate grids and corrected depth
np.save('/home/noam/Documents/DEMgen/files/WIO_Z.npy', z)

# Save to netCDF
fh = Dataset(gebco_write, mode='r+')
fh['elevation'][:, :] = z
fh.close()

##############################################################################
# Plotting ###################################################################
##############################################################################

# Set up plot
f0 = plt.figure(figsize=(40, 30))
a0 = f0.add_subplot(1, 1, 1, projection=ct.PlateCarree())
a0.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ct. PlateCarree())

# Plot corrected data
depth_map = a0.imshow(z, extent=(lon_min, lon_max, lat_min, lat_max),
                      origin='lower', transform=ct.PlateCarree(), vmin=-5000,
                      vmax=5000, cmap=cm.topo)

gl0 = a0.gridlines(crs=ct.PlateCarree(), draw_labels=True, linewidth=2,
                   color='k')
gl0.xlabels_top = False
gl0.xlocator = mticker.FixedLocator(np.linspace(0, 360, num=73))
gl0.xformatter = LONGITUDE_FORMATTER
gl0.xlabel_style = {'size': 20}
gl0.ylabels_right = False
gl0.ylocator = mticker.FixedLocator(np.linspace(-90, 90, num=37))
gl0.yformatter = LATITUDE_FORMATTER
gl0.ylabel_style = {'size': 20}

plt.title('Stage 2 DEM processing', fontsize=30)
plt.savefig('files/figures/DEM_stage_2.jpg', dpi=300)
plt.close()
