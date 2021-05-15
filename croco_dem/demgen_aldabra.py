# -*- coding: utf-8 -*-
"""
Generate a gridded DEM from drying height, coastline and point depth data
Version for Aldabra Atoll
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

from scipy.interpolate import griddata
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


##############################################################################
# Global Variables ###########################################################
##############################################################################

grid_spacing = 0.001                 # Degrees
lon_min, lon_max = 46.1, 46.6          # Degrees
lat_min, lat_max = -9.55, -9.3       # Degrees

mhws = 3.4                           # m above LAT

point_depth_file = ('/home/noam/Documents/DEMgen/files/aldabra_depths.csv')
l0_file = ('/home/noam/Documents/DEMgen/files/aldabra_l0.csv')
l1_file = ('/home/noam/Documents/DEMgen/files/aldabra_l1.csv')

l0_z = 0                             # m above LAT
l1_z = mhws                          # m above LAT

##############################################################################
# Import & Format Data #######################################################
##############################################################################

point_depth_data = np.loadtxt(point_depth_file, delimiter=',', skiprows=1)
l0_data = np.loadtxt(l0_file, delimiter=',', skiprows=1)
l1_data = np.loadtxt(l1_file, delimiter=',', skiprows=1)

l0_data = np.append(l0_z*np.ones((np.shape(l0_data)[0], 1)), l0_data, axis=1)
l1_data = np.append(l1_z*np.ones((np.shape(l1_data)[0], 1)), l1_data, axis=1)

##############################################################################
# Main Script ################################################################
##############################################################################

# This script generates an interpolated DEM based on drying height/coastline
# data, and point depths from a hydrographic map. A regular lat-lon grid is
# generated, and values are interpolated onto this grid.

# Generate the coordinate and data grid
nx = round(((lon_max - lon_min)/grid_spacing) + 1)
ny = round(((lat_max - lat_min)/grid_spacing) + 1)
x, y = np.meshgrid(np.linspace(lon_min, lon_max, num=nx),
                   np.linspace(lat_min, lat_max, num=ny))
z = np.zeros(np.shape(x))

# Combine all data into a single array for interpolation
# depth || lon || lat
data = np.concatenate((point_depth_data, l0_data, l1_data))

# Carry out the interpolation
print('Interpolating data')
z = griddata((data[:, 1], data[:, 2]), (data[:, 0]), (x, y), method='cubic')
print('Interpolation complete')

# Set up plot
f0 = plt.figure(figsize=(10, 5))
a0 = f0.add_subplot(1, 1, 1, projection=ct.PlateCarree())
a0.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ct. PlateCarree())
a0.coastlines()
gl0 = a0.gridlines(crs=ct.PlateCarree(), draw_labels=True, linewidth=0.2,
                   color='k', alpha=0.5)
gl0.xlabels_top = False
gl0.xlocator = mticker.FixedLocator(np.linspace(46, 47, num=11))
gl0.xformatter = LONGITUDE_FORMATTER
gl0.ylabels_right = False
gl0.ylocator = mticker.FixedLocator(np.linspace(-10, -9, num=11))
gl0.yformatter = LATITUDE_FORMATTER

# Plot interpolated data
depth_map = a0.imshow(z, extent=(lon_min, lon_max, lat_min, lat_max),
                      origin='lower', transform=ct.PlateCarree(), vmin=0,
                      vmax=3500, cmap=cm.tempo)
coast_l0 = a0.plot(l0_data[:, 1], l0_data[:, 2], linewidth='0.5', color='k',
                   alpha=1.0)
coast_l1 = a0.scatter(l1_data[:, 1], l1_data[:, 2], 0.0001, c='k')
a0_cb = plt.colorbar(depth_map, ax=a0)

# Plot depth points
depth_points = a0.scatter(point_depth_data[:, 1], point_depth_data[:, 2],
                          0.01, c='k')

plt.title('Aldabra - 0.1km')
plt.savefig('files/figures/aldabra_dem.jpg', dpi=300)
