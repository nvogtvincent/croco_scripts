#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise marine debris results
@author: noam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cmasher as cmr
import xarray as xr
from osgeo import gdal, osr
from skimage.measure import block_reduce
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio

# PARAMETERS
param = {'grid_res': 1.0,                          # Grid resolution in degrees
         'lon_range': [20, 130],                   # Longitude range for output
         'lat_range': [-40, 30],                   # Latitude range for output

         # Analysis parameters
         'us_d': 1825,    # Sinking timescale (days)
         'ub_d': 20,      # Beaching timescale (days)

         # Time range
         'y0'  : 1993,
         'y1'  : 2012,

         # Physics
         'mode': '0010',

         # Sink sites
         'sites': np.array([1]),

         # Plot ship tracks
         'tracks': True,

         # Export
         'export': True}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'fisheries': os.path.dirname(os.path.realpath(__file__)) + '/../FISHERIES/DATA/PROC/',
        'shipping': os.path.dirname(os.path.realpath(__file__)) + '/../SHIPPING/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'flux': dirs['script'] + '/marine_flux_' + param['mode'] + '_' + np.array2string(param['sites'], separator='-') + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'drift_time': dirs['script'] + '/marine_drift_time_' + param['mode'] + '_' + np.array2string(param['sites'], separator='-') + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc',
      'debris': dirs['fisheries'] + 'GFW_gridded.nc',
      'shipping': dirs['shipping'] + 'shipping_2008.tif'}

n_years = param['y1']-param['y0']+1
##############################################################################
# LOAD DATA                                                                  #
##############################################################################

fmatrix = xr.open_dataarray(fh['flux'])
ftmatrix = xr.open_dataarray(fh['drift_time'])

# To find the proportion of released debris that arrived at site, divide the
# total flux from a cell by the total released, which is (36*4*12) per year.
fmatrix_mean = fmatrix.sum(dim='sink_time')/(1728*n_years)
ftmatrix_mean = ftmatrix.sum(dim='sink_time')/(1728*n_years)
tmatrix_mean = ftmatrix_mean/fmatrix_mean

# Now degrade resolution by factor 12
fmatrix_mean_12 = block_reduce(fmatrix_mean, block_size=(12, 12), func=np.mean)
tmatrix_mean_12 = block_reduce(tmatrix_mean, block_size=(12, 12), func=np.mean)

# Open debris input functions (from fisheries)
debris = xr.open_dataset(fh['debris'])

# Open ship tracks

with rasterio.open(fh['shipping']) as src:
    img = src.read(1)
    height = img.shape[0]
    width = img.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)

img = block_reduce(img[:10520, :15700], (20, 20), np.sum)
lons = lons[:10520, :15700][::20, ::20]
lats = lats[:10520, :15700][::20, ::20]

# if param['tracks']:
#     ship_data_obj = gdal.Open(fh['shipping'])
#     ship_data = ship_data_obj.ReadAsArray()[:]

#     # Coordinate system
#     old_cs = osr.SpatialReference()
#     old_cs.ImportFromWkt(ship_data_obj.GetProjectionRef())
#     wgs84_wkt = """
#     GEOGCS["WGS 84",
#         DATUM["WGS_1984",
#             SPHEROID["WGS 84",6378137,298.257223563,
#                 AUTHORITY["EPSG","7030"]],
#             AUTHORITY["EPSG","6326"]],
#         PRIMEM["Greenwich",0,
#             AUTHORITY["EPSG","8901"]],
#         UNIT["degree",0.01745329251994328,
#             AUTHORITY["EPSG","9122"]],
#         AUTHORITY["EPSG","4326"]]"""
#     new_cs = osr.SpatialReference()
#     new_cs.ImportFromWkt(wgs84_wkt)
#     transform = osr.CoordinateTransformation(old_cs, new_cs)

#     xdim = ship_data_obj.RasterXSize
#     ydim = ship_data_obj.RasterYSize
#     gt = ship_data_obj.GetGeoTransform()
#     x_lim = [gt[0], gt[0] + xdim*gt[1] + ydim*gt[2]]
#     y_lim = [gt[3], gt[3] + xdim*gt[4] + ydim*gt[5]]

#     img_lon_lim = [transform.TransformPoint(x_lim[0], y_lim[0])[1],
#                     transform.TransformPoint(x_lim[1], y_lim[0])[1]]
#     img_lat_lim = [transform.TransformPoint(x_lim[0], y_lim[0])[0],
#                     transform.TransformPoint(x_lim[0], y_lim[1])[0]]

#     lon_arr = np.linspace(img_lon_lim[0], img_lon_lim[1], num=xdim)
#     lat_arr = np.linspace(img_lat_lim[0], img_lat_lim[1], num=ydim)

#     SA = 9.55**2 # (see https://allencoralatlas.org/methods/)

#     img_lon, img_lat = np.meshgrid(lon_arr, lat_arr)

##############################################################################
# PLOT                                                                       #
##############################################################################

f = plt.figure(constrained_layout=True, figsize=(45, 20))
gs = GridSpec(3, 4, figure=f, width_ratios=[0.91, 0.01, 0.29, 0.01])
ax = []
ax.append(f.add_subplot(gs[:, 0], projection = ccrs.PlateCarree())) # Main figure (flux probability)
ax.append(f.add_subplot(gs[:, 1])) # Colorbar for main figure
ax.append(f.add_subplot(gs[0, 2], projection = ccrs.PlateCarree())) # Fisheries 1
ax.append(f.add_subplot(gs[1, 2], projection = ccrs.PlateCarree())) # Fisheries 2
ax.append(f.add_subplot(gs[2, 2], projection = ccrs.PlateCarree())) # Fisheries 3
ax.append(f.add_subplot(gs[:, 3])) # Colorbar for fisheries

# f.subplots_adjust(hspace=0.02, wspace=0.02)
gl = []
hist = []

land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='white',
                                        facecolor='black',
                                        zorder=1)

for i in [0, 2, 3, 4]:
    ax[i].set_aspect(1)
    ax[i].set_facecolor('k')

# Total flux
hist.append(ax[0].pcolormesh(fmatrix.coords['longitude'], fmatrix.coords['latitude'], fmatrix_mean,
                             cmap=cmr.ember, norm=colors.LogNorm(vmin=1e-6, vmax=1e-2),
                             transform=ccrs.PlateCarree(), rasterized=True))

ax[0].add_feature(land_10m)
gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='white', linestyle='--', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].xlabels_top = False
gl[0].ylabels_right = False
gl[0].ylabel_style = {'size': 24}
gl[0].xlabel_style = {'size': 24}
ax[0].text(21, -39, 'Likelihood of debris beaching at Aldabra', fontsize=42, color='w', fontweight='bold')
ax[0].set_xlim([20, 130])
ax[0].set_ylim(-40, 30)
# Add an overlay with ship tracks
thresh = 500
img = np.ma.masked_where(img < thresh, img)
ax[0].pcolormesh(lons, lats, img, cmap=cmr.neutral_r, vmin=thresh-1, vmax=thresh,
                 transform=ccrs.PlateCarree(), alpha=0.3, rasterized=True)

cb0 = plt.colorbar(hist[0], cax=ax[1], fraction=0.1)
cb0.set_label('Mass fraction', size=24)
ax[1].tick_params(axis='y', labelsize=24)

# Input from purse seiners
ps_input = fmatrix_mean_12*debris['purse_seiners_A_time_integral']
hist.append(ax[2].pcolormesh(debris.coords['lon_bnd'], debris.coords['lat_bnd'],
                             ps_input/ps_input.max(),
                             cmap=cmr.ghostlight, norm=colors.LogNorm(vmin=1e-3, vmax=1e0),
                             transform=ccrs.PlateCarree(), rasterized=True))
ax[2].add_feature(land_10m)
ax[2].text(21, -39, 'Purse seiners', fontsize=24, color='w', fontweight='bold')

# Input from drifting and set longlines
ll_input = fmatrix_mean_12*debris['longlines_A_time_integral']
hist.append(ax[3].pcolormesh(debris.coords['lon_bnd'], debris.coords['lat_bnd'],
                             ll_input/ll_input.max(),
                             cmap=cmr.ghostlight, norm=colors.LogNorm(vmin=1e-3, vmax=1e0),
                             transform=ccrs.PlateCarree(), rasterized=True))
ax[3].add_feature(land_10m)
ax[3].text(21, -39, 'Set/drifting longlines', fontsize=24, color='w', fontweight='bold')

# Input from pole and line and trollers
pl_input = fmatrix_mean_12*debris['pole_and_line_and_trollers_A_time_integral']
hist.append(ax[4].pcolormesh(debris.coords['lon_bnd'], debris.coords['lat_bnd'],
                             pl_input/pl_input.max(),
                             cmap=cmr.ghostlight, norm=colors.LogNorm(vmin=1e-3, vmax=1e0),
                             transform=ccrs.PlateCarree(), rasterized=True))
ax[4].add_feature(land_10m)
ax[4].text(21, -39, 'Pole and line/trollers', fontsize=24, color='w', fontweight='bold')

cb1 = plt.colorbar(hist[1], cax=ax[5])
cb1.set_label('Normalised source strength', size=24)
ax[5].tick_params(axis='y', labelsize=24)

for ii, i in enumerate([2, 3, 4]):
    gl.append(ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='white', linestyle='--', zorder=11))
    gl[ii+1].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
    gl[ii+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[ii+1].xlabels_top = False
    gl[ii+1].ylabels_right = False
    gl[ii+1].ylabels_left = False
    gl[ii+1].xlabels_bottom = False

if param['export']:
    plt.savefig(dirs['fig'] + 'marine_sources_' + param['mode'] + '_' + np.array2string(param['sites'], separator='-') + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.pdf', bbox_inches='tight', dpi=300)








