# -*- coding: utf-8 -*-
"""
Modifies coastline-corrected GEBCO 2020 grid with constraints from point data
Noam Vogt-Vincent (2020)
Run first (then oceanmask.py)
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
from netCDF4 import Dataset


##############################################################################
# Global Variables ###########################################################
##############################################################################

# GEBCO source
gebco_source = ('/home/noam/Documents/DEMgen/files/GEBCO2019.nc')

# Island specific variables ##################################################
# Aldabra-Assumption
lon_min_aldass, lon_max_aldass = 46, 46.7          # Degrees
lat_min_aldass, lat_max_aldass = -9.85, -9.15      # Degrees

mhws_ald = 3.4                                     # m above LAT
mhws_ass = 3.0                                     # m above LAT

l_ald = 30                                         # km

point_depth_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_depths.csv')
l0_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l0.csv')
l1_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l1.csv')
l500_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l500.csv')
l1000_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l1000.csv')
l2000_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l2000.csv')
l3000_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l3000.csv')
l4000_file_ald = ('/home/noam/Documents/DEMgen/files/aldabra_l4000.csv')

point_depth_file_ass = ('/home/noam/Documents/DEMgen/files/'
                        'assumption_depths.csv')
l0_file_ass = ('/home/noam/Documents/DEMgen/files/assumption_l0.csv')
l1_file_ass = ('/home/noam/Documents/DEMgen/files/assumption_l1.csv')
l100_file_ass = ('/home/noam/Documents/DEMgen/files/assumption_l100.csv')
l500_file_ass = ('/home/noam/Documents/DEMgen/files/assumption_l500.csv')
l1000_file_ass = ('/home/noam/Documents/DEMgen/files/assumption_l1000.csv')
l2000_file_ass = ('/home/noam/Documents/DEMgen/files/assumption_l2000.csv')

point_depth_file_int = ('/home/noam/Documents/DEMgen/files/'
                        'interisland_depths_mod.csv')

l0_z_ald = 0                                   # m above LAT
l1_z_ald = mhws_ald                            # m above LAT

l0_z_ass = 0                                   # m above LAT
l1_z_ass = mhws_ass                            # m above LAT

# Cosmoledo
lon_min_cos, lon_max_cos = 47.30, 47.85        # Degrees
lat_min_cos, lat_max_cos = -9.95, -9.50      # Degrees

mhws_cos = 2.4                                    # m above LAT

l_cos = 20                                         # km (blending LS)

point_depth_file_cos = ('/home/noam/Documents/DEMgen/files/cosmoledo_depths.csv')
l0_file_cos = ('/home/noam/Documents/DEMgen/files/cosmoledo_l0.csv')
l1_file_cos = ('/home/noam/Documents/DEMgen/files/cosmoledo_l1.csv')
l500_file_cos = ('/home/noam/Documents/DEMgen/files/cosmoledo_l500.csv')
l1000_file_cos = ('/home/noam/Documents/DEMgen/files/cosmoledo_l1000.csv')
l2000_file_cos = ('/home/noam/Documents/DEMgen/files/cosmoledo_l2000.csv')

point_depth_file_int = ('/home/noam/Documents/DEMgen/files/'
                        'interisland_depths_mod.csv')

l0_z_cos = 0                                   # m above LAT
l1_z_cos = mhws_cos                           # m above LAT

# Iles-glorieuses
lon_min_ig, lon_max_ig = 47.24, 47.47        # Degrees
lat_min_ig, lat_max_ig = -11.63, -11.41      # Degrees

mhws_ig = 3.4                                    # m above LAT

l_ig = 10                                         # km (blending LS)

point_depth_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_depths.csv')
l0_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l0.csv')
l1_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l1.csv')
l10_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l1.csv')
l100_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l1.csv')
l500_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l500.csv')
l1000_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l1000.csv')
l2000_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l2000.csv')
l3000_file_ig = ('/home/noam/Documents/DEMgen/files/iles_glorieuses_l3000.csv')

point_depth_file_int = ('/home/noam/Documents/DEMgen/files/'
                        'interisland_depths_mod.csv')

l0_z_ig = 0                                   # m above LAT
l1_z_ig = mhws_ig                           # m above LAT

# Plotting variables
lon_min = 34
lon_max = 78
lat_min = -23.5
lat_max = 0

# Astove
lon_min_ast, lon_max_ast = 47.65, 47.81        # Degrees
lat_min_ast, lat_max_ast = -10.2, -10      # Degrees

mhws_ast = 2.4                                    # m above LAT

l_ast = 10                                         # km (blending LS)

point_depth_file_ast = ('/home/noam/Documents/DEMgen/files/astove_depths.csv')
l0_file_ast = ('/home/noam/Documents/DEMgen/files/astove_l0.csv')
l1_file_ast = ('/home/noam/Documents/DEMgen/files/astove_l1.csv')
l500_file_ast = ('/home/noam/Documents/DEMgen/files/astove_l500.csv')
l1000_file_ast = ('/home/noam/Documents/DEMgen/files/astove_l1000.csv')

point_depth_file_int = ('/home/noam/Documents/DEMgen/files/'
                        'interisland_depths_mod.csv')

l0_z_ast = 0                                   # m above LAT
l1_z_ast = mhws_ast                           # m above LAT

##############################################################################
# Import & Format Data #######################################################
##############################################################################

# Import base GEBCO 2020
# Skip row 1 due to mismatch with mask - change in future if necessary (OLD)
fh = Dataset(gebco_source, mode='r')
x_gebco = np.array(fh.variables['lon'][:])
y_gebco = np.array(fh.variables['lat'][:])
Z = np.array(fh.variables['elevation'][:, :])
X, Y = np.meshgrid(x_gebco, y_gebco)
Z_gebco = np.copy(Z)  # Store original GEBCO elevation for comparison
fh.close()

# Format depth points
# Load files
point_depth_data_ald = np.loadtxt(point_depth_file_ald, delimiter=',',
                                  skiprows=1)
point_depth_data_ass = np.loadtxt(point_depth_file_ass, delimiter=',',
                                  skiprows=1)
point_depth_data_cos = np.loadtxt(point_depth_file_cos, delimiter=',',
                                  skiprows=1)
point_depth_data_ig = np.loadtxt(point_depth_file_ig, delimiter=',',
                                 skiprows=1)
point_depth_data_ast = np.loadtxt(point_depth_file_ast, delimiter=',',
                                  skiprows=1)
point_depth_data_int = np.loadtxt(point_depth_file_int, delimiter=',',
                                  skiprows=1)

l0_data_ald = np.loadtxt(l0_file_ald, delimiter=',', skiprows=1)
l1_data_ald = np.loadtxt(l1_file_ald, delimiter=',', skiprows=1)
l500_data_ald = np.loadtxt(l500_file_ald, delimiter=',', skiprows=1)
l1000_data_ald = np.loadtxt(l1000_file_ald, delimiter=',', skiprows=1)
l2000_data_ald = np.loadtxt(l2000_file_ald, delimiter=',', skiprows=1)
l3000_data_ald = np.loadtxt(l3000_file_ald, delimiter=',', skiprows=1)
l4000_data_ald = np.loadtxt(l4000_file_ald, delimiter=',', skiprows=1)

l0_data_ass = np.loadtxt(l0_file_ass, delimiter=',', skiprows=1)
l1_data_ass = np.loadtxt(l1_file_ass, delimiter=',', skiprows=1)
l100_data_ass = np.loadtxt(l100_file_ass, delimiter=',', skiprows=1)
l500_data_ass = np.loadtxt(l500_file_ass, delimiter=',', skiprows=1)
l1000_data_ass = np.loadtxt(l1000_file_ass, delimiter=',', skiprows=1)
l2000_data_ass = np.loadtxt(l2000_file_ass, delimiter=',', skiprows=1)

l0_data_cos = np.loadtxt(l0_file_cos, delimiter=',', skiprows=1)
l1_data_cos = np.loadtxt(l1_file_cos, delimiter=',', skiprows=1)
l500_data_cos = np.loadtxt(l500_file_cos, delimiter=',', skiprows=1)
l1000_data_cos = np.loadtxt(l1000_file_cos, delimiter=',', skiprows=1)
l2000_data_cos = np.loadtxt(l2000_file_cos, delimiter=',', skiprows=1)

l0_data_ast = np.loadtxt(l0_file_ast, delimiter=',', skiprows=1)
l1_data_ast = np.loadtxt(l1_file_ast, delimiter=',', skiprows=1)
l500_data_ast = np.loadtxt(l500_file_ast, delimiter=',', skiprows=1)
l1000_data_ast = np.loadtxt(l1000_file_ast, delimiter=',', skiprows=1)

l0_data_ig = np.loadtxt(l0_file_ig, delimiter=',', skiprows=1)
l1_data_ig = np.loadtxt(l1_file_ig, delimiter=',', skiprows=1)
l10_data_ig = np.loadtxt(l10_file_ig, delimiter=',', skiprows=1)
l100_data_ig = np.loadtxt(l100_file_ig, delimiter=',', skiprows=1)
l500_data_ig = np.loadtxt(l500_file_ig, delimiter=',', skiprows=1)
l1000_data_ig = np.loadtxt(l1000_file_ig, delimiter=',', skiprows=1)
l2000_data_ig = np.loadtxt(l2000_file_ig, delimiter=',', skiprows=1)
l3000_data_ig = np.loadtxt(l3000_file_ig, delimiter=',', skiprows=1)

# Format data
l0_data_ald = np.append(l0_z_ald*np.ones((np.shape(l0_data_ald)[0], 1)),
                        l0_data_ald, axis=1)
l1_data_ald = np.append(l1_z_ald*np.ones((np.shape(l1_data_ald)[0], 1)),
                        l1_data_ald, axis=1)
l500_data_ald = np.append(500*np.ones((np.shape(l500_data_ald)[0], 1)),
                          l500_data_ald, axis=1)
l1000_data_ald = np.append(1000*np.ones((np.shape(l1000_data_ald)[0], 1)),
                           l1000_data_ald, axis=1)
l2000_data_ald = np.append(2000*np.ones((np.shape(l2000_data_ald)[0], 1)),
                           l2000_data_ald, axis=1)
l3000_data_ald = np.append(3000*np.ones((np.shape(l3000_data_ald)[0], 1)),
                           l3000_data_ald, axis=1)
l4000_data_ald = np.append(4000*np.ones((np.shape(l4000_data_ald)[0], 1)),
                           l4000_data_ald, axis=1)

l0_data_ass = np.append(l0_z_ass*np.ones((np.shape(l0_data_ass)[0], 1)),
                        l0_data_ass, axis=1)
l1_data_ass = np.append(l1_z_ass*np.ones((np.shape(l1_data_ass)[0], 1)),
                        l1_data_ass, axis=1)
l100_data_ass = np.append(100*np.ones((np.shape(l100_data_ass)[0], 1)),
                          l100_data_ass, axis=1)
l500_data_ass = np.append(500*np.ones((np.shape(l500_data_ass)[0], 1)),
                          l500_data_ass, axis=1)
l1000_data_ass = np.append(1000*np.ones((np.shape(l1000_data_ass)[0], 1)),
                           l1000_data_ass, axis=1)
l2000_data_ass = np.append(2000*np.ones((np.shape(l2000_data_ass)[0], 1)),
                           l2000_data_ass, axis=1)

l0_data_cos = np.append(l0_z_cos*np.ones((np.shape(l0_data_cos)[0], 1)),
                        l0_data_cos, axis=1)
l1_data_cos = np.append(l1_z_cos*np.ones((np.shape(l1_data_cos)[0], 1)),
                        l1_data_cos, axis=1)
l500_data_cos = np.append(500*np.ones((np.shape(l500_data_cos)[0], 1)),
                          l500_data_cos, axis=1)
l1000_data_cos = np.append(1000*np.ones((np.shape(l1000_data_cos)[0], 1)),
                           l1000_data_cos, axis=1)
l2000_data_cos = np.append(2000*np.ones((np.shape(l2000_data_cos)[0], 1)),
                           l2000_data_cos, axis=1)

l0_data_ast = np.append(l0_z_ast*np.ones((np.shape(l0_data_ast)[0], 1)),
                        l0_data_ast, axis=1)
l1_data_ast = np.append(l1_z_ast*np.ones((np.shape(l1_data_ast)[0], 1)),
                        l1_data_ast, axis=1)
l500_data_ast = np.append(500*np.ones((np.shape(l500_data_ast)[0], 1)),
                          l500_data_ast, axis=1)
l1000_data_ast = np.append(1000*np.ones((np.shape(l1000_data_ast)[0], 1)),
                           l1000_data_ast, axis=1)


l0_data_ig = np.append(l0_z_ig*np.ones((np.shape(l0_data_ig)[0], 1)),
                       l0_data_ig, axis=1)
l1_data_ig = np.append(l1_z_ig*np.ones((np.shape(l1_data_ig)[0], 1)),
                       l1_data_ig, axis=1)
l10_data_ig = np.append(10*np.ones((np.shape(l10_data_ig)[0], 1)),
                        l10_data_ig, axis=1)
l100_data_ig = np.append(100*np.ones((np.shape(l100_data_ig)[0], 1)),
                         l100_data_ig, axis=1)
l500_data_ig = np.append(500*np.ones((np.shape(l500_data_ig)[0], 1)),
                         l500_data_ig, axis=1)
l1000_data_ig = np.append(1000*np.ones((np.shape(l1000_data_ig)[0], 1)),
                          l1000_data_ig, axis=1)
l2000_data_ig = np.append(2000*np.ones((np.shape(l2000_data_ig)[0], 1)),
                          l2000_data_ig, axis=1)
##############################################################################
# Main Script ################################################################
##############################################################################

# This script firstly generates free-standing DEMs for ocean islands with
# good hydrographic constraints from UKHO charts, and then merges these DEMs
# with the previously-generated GEBCO-derived DEM

##############################################################################
# Aldabra-Assumption #########################################################
##############################################################################

# Generate the coordinate and data grid (extracted from GEBCO)
# Firstly extract the indices of the coordinates in the master grid (X,Y) that
# best match those given in the parameters above (lon_min_X, lon_max_X, etc.)

lon_index_left = np.searchsorted(X[0, :], lon_min_aldass)
lon_index_right = np.searchsorted(X[0, :], lon_max_aldass)
lat_index_bottom = np.searchsorted(Y[:, 0], lat_min_aldass)
lat_index_top = np.searchsorted(Y[:, 0], lat_max_aldass)

x_ald = X[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]
y_ald = Y[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Combine all relevant hydrographic data into a single array for interpolation
# depth || lon || lat
data = np.concatenate((l0_data_ald, l1_data_ald, l500_data_ald, l1000_data_ald,
                       l3000_data_ald, l4000_data_ald, point_depth_data_ald,
                       l0_data_ass, l1_data_ass, l100_data_ass, l500_data_ass,
                       l1000_data_ass, l2000_data_ass, point_depth_data_ass,
                       point_depth_data_int))

# Carry out the interpolation
print('Interpolating data for Aldabra...')
z_ald = griddata((data[:, 1], data[:, 2]), (data[:, 0]), (x_ald, y_ald),
                 method='linear')
print('Interpolation complete')

# Blend with master grid (linear blending across l_ald cells)
# Copy of master grid for region
l_ald = l_ald*2  # assuming GEBCO resolution of 0.5km
z_geb_ald = Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Template for gradient
z_trans = np.ones_like(z_ald)*(1/l_ald)

for grad_level in range(1, l_ald):
    grad_val = (1/l_ald)*(1+grad_level)
    z_trans[grad_level:-grad_level, grad_level:-grad_level] = grad_val

z_ald = -z_ald*z_trans
z_geb_ald = z_geb_ald*(1-z_trans)
z_ald = z_ald + z_geb_ald

# Splice into master grid
Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right] = z_ald

##############################################################################
# Cosmoledo ##################################################################
##############################################################################

# Generate the coordinate and data grid (extracted from GEBCO)
# Firstly extract the indices of the coordinates in the master grid (X,Y) that
# best match those given in the parameters above (lon_min_X, lon_max_X, etc.)

lon_index_left = np.searchsorted(X[0, :], lon_min_cos)
lon_index_right = np.searchsorted(X[0, :], lon_max_cos)
lat_index_bottom = np.searchsorted(Y[:, 0], lat_min_cos)
lat_index_top = np.searchsorted(Y[:, 0], lat_max_cos)

x_cos = X[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]
y_cos = Y[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Combine all relevant hydrographic data into a single array for interpolation
# depth || lon || lat
data = np.concatenate((l0_data_cos, l1_data_cos, l500_data_cos, l1000_data_cos,
                       l2000_data_cos, point_depth_data_cos,
                       point_depth_data_int))

# Carry out the interpolation
print('Interpolating data for Cosmoledo...')
z_cos = griddata((data[:, 1], data[:, 2]), (data[:, 0]), (x_cos, y_cos),
                 method='linear')
print('Interpolation complete')

# Blend with master grid (linear blending across l_cos cells)
# Copy of master grid for region
l_cos = l_cos*2  # assuming GEBCO resolution of 0.5km
z_geb_cos = Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Template for gradient
z_trans = np.ones_like(z_cos)*(1/l_cos)

for grad_level in range(1, l_cos):
    grad_val = (1/l_cos)*(1+grad_level)
    z_trans[grad_level:-grad_level, grad_level:-grad_level] = grad_val

z_cos = -z_cos*z_trans
z_geb_cos = z_geb_cos*(1-z_trans)
z_cos = z_cos + z_geb_cos

# Splice into master grid
Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right] = z_cos

##############################################################################
# Astove ##################################################################
##############################################################################

# Generate the coordinate and data grid (extracted from GEBCO)
# Firstly extract the indices of the coordinates in the master grid (X,Y) that
# best match those given in the parameters above (lon_min_X, lon_max_X, etc.)

lon_index_left = np.searchsorted(X[0, :], lon_min_ast)
lon_index_right = np.searchsorted(X[0, :], lon_max_ast)
lat_index_bottom = np.searchsorted(Y[:, 0], lat_min_ast)
lat_index_top = np.searchsorted(Y[:, 0], lat_max_ast)

x_ast = X[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]
y_ast = Y[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Combine all relevant hydrographic data into a single array for interpolation
# depth || lon || lat
data = np.concatenate((l0_data_ast, l1_data_ast, l500_data_ast, l1000_data_ast,
                       point_depth_data_ast, point_depth_data_int))

# Carry out the interpolation
print('Interpolating data for Astove...')
z_ast = griddata((data[:, 1], data[:, 2]), (data[:, 0]), (x_ast, y_ast),
                 method='linear')
print('Interpolation complete')

# Blend with master grid (linear blending across l_cos cells)
# Copy of master grid for region
l_ast = l_ast*2  # assuming GEBCO resolution of 0.5km
z_geb_ast = Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Template for gradient
z_trans = np.ones_like(z_ast)*(1/l_ast)

for grad_level in range(1, l_ast):
    grad_val = (1/l_ast)*(1+grad_level)
    z_trans[grad_level:-grad_level, grad_level:-grad_level] = grad_val

z_ast = -z_ast*z_trans
z_geb_ast = z_geb_ast*(1-z_trans)
z_ast = z_ast + z_geb_ast

# Splice into master grid
Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right] = z_ast

##############################################################################
# Iles-glorieuses ############################################################
##############################################################################

# Generate the coordinate and data grid (extracted from GEBCO)
# Firstly extract the indices of the coordinates in the master grid (X,Y) that
# best match those given in the parameters above (lon_min_X, lon_max_X, etc.)

lon_index_left = np.searchsorted(X[0, :], lon_min_ig)
lon_index_right = np.searchsorted(X[0, :], lon_max_ig)
lat_index_bottom = np.searchsorted(Y[:, 0], lat_min_ig)
lat_index_top = np.searchsorted(Y[:, 0], lat_max_ig)

x_ig = X[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]
y_ig = Y[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Combine all relevant hydrographic data into a single array for interpolation
# depth || lon || lat
data = np.concatenate((l0_data_ig, l1_data_ig, l10_data_ig, l100_data_ig,
                       l500_data_ig, l1000_data_ig, l2000_data_ig,
                       point_depth_data_ig, point_depth_data_int))

# Carry out the interpolation
print('Interpolating data for Iles-Glorieuses...')
z_ig = griddata((data[:, 1], data[:, 2]), (data[:, 0]), (x_ig, y_ig),
                 method='linear')
print('Interpolation complete')

# Blend with master grid (linear blending across l_cos cells)
# Copy of master grid for region
l_ig = l_ig*2  # assuming GEBCO resolution of 0.5km
z_geb_ig = Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right]

# Template for gradient
z_trans = np.ones_like(z_ig)*(1/l_ig)

for grad_level in range(1, l_ig):
    grad_val = (1/l_ig)*(1+grad_level)
    z_trans[grad_level:-grad_level, grad_level:-grad_level] = grad_val

z_ig = -z_ig*z_trans
z_geb_ig = z_geb_ig*(1-z_trans)
z_ig = z_ig + z_geb_ig

# Splice into master grid
Z[lat_index_bottom:lat_index_top, lon_index_left:lon_index_right] = z_ig

# Save the coordinate grids and corrected depth
np.save('/home/noam/Documents/DEMgen/files/WIO_X.npy', X)
np.save('/home/noam/Documents/DEMgen/files/WIO_Y.npy', Y)
np.save('/home/noam/Documents/DEMgen/files/WIO_Z_prelim.npy', Z)

##############################################################################
# Plotting ###################################################################
##############################################################################

# Set up plot
f0 = plt.figure(figsize=(40, 30))
a0 = f0.add_subplot(1, 1, 1, projection=ct.PlateCarree())
a0.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ct. PlateCarree())

# Plot corrected data
depth_map = a0.imshow(Z, extent=(lon_min, lon_max, lat_min, lat_max),
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

plt.title('Stage 1 DEM processing', fontsize=30)
plt.savefig('files/figures/DEM_stage_1.jpg', dpi=300)
plt.close()


# Plot the original GEBCO data
f0 = plt.figure(figsize=(40, 30))
a0 = f0.add_subplot(1, 1, 1, projection=ct.PlateCarree())
a0.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ct. PlateCarree())

# Plot corrected data
depth_map = a0.imshow(Z_gebco, extent=(lon_min, lon_max, lat_min, lat_max),
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

plt.title('Preprocessed GEBCO 2019 DEM', fontsize=30)
plt.savefig('files/figures/DEM_stage_0.jpg', dpi=300)
plt.close()
