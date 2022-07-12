#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods to set up a marine debris simulation using CMEMS data
@author: noam
"""

import numpy as np
import os.path
import numba
import geopandas as gpd
import matplotlib.pyplot as plt
import cmasher as cmr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from netCDF4 import Dataset
from calendar import monthrange
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from datetime import datetime, timedelta
from osgeo import gdal


@numba.jit(nopython=True)
def haversine_np(lon1, lat1, lon2, lat2):
    # From https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

@numba.jit(nopython=True)
def cdist_calc(cnearest, lon, lat, efs):
    # Firstly evaluate target longitude and latitude
    lon_target = np.zeros_like(lon)
    lat_target = np.zeros_like(lat)

    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            target_i = cnearest[0, i, j]
            target_j = cnearest[1, i, j]

            lon_target[i, j] = lon[target_i, target_j]
            lat_target[i, j] = lat[target_i, target_j]

    # Now calculate distances between (lon, lat) and (lon_target, lat_target)
    cdist = haversine_np(lon, lat, lon_target, lat_target)

    return cdist

@numba.jit(nopython=True)
def cplastic_calc(plastic_data, cnearest):
    cplastic = np.zeros_like(plastic_data)

    for i in range(plastic_data.shape[0]):
        for j in range(plastic_data.shape[1]):
            target_i = cnearest[0, i, j]
            target_j = cnearest[1, i, j]

            cplastic[target_i, target_j] += plastic_data[i, j]

    return cplastic

@numba.jit(nopython=True)
def id_coast_cells(coast, country_id, country_id_nearest):
    coast_id = np.copy(coast)

    for i in range(coast_id.shape[0]):
        for j in range(coast_id.shape[1]):
            if coast[i, j] == 1:
                # Firstly check if the coast already has a country ID
                id0 = country_id[i, j]

                if id0 == -32768:
                    target_i = country_id_nearest[0, i, j]
                    target_j = country_id_nearest[1, i, j]
                    id0 = country_id[target_i, target_j]

                coast_id[i, j] = id0

    return coast_id

@numba.jit(nopython=True)
def psi_mask(lsm_rho, psi_grid):
    # This function generates a land-sea mask on an empty psi grid from the
    # rho grid
    # Land nodes on the psi grid are defined as being at the centre of 4 land
    # nodes on the rho grid:
    # Coast nodes on the psi grid are defined as being at the centre of 1-3
    # land nodes on the rho grid:

    # Function also assigns an id to each cell with the following format:
    # id: [XXXXYYYY]
    # XXXX: i index
    # YYYY: j index

    # p_______p
    # |   :   |
    # |...r...|
    # |   :   |
    # p_______p

    lsm_psi   = np.zeros_like(psi_grid, dtype=np.int8)
    coast_psi = np.zeros_like(psi_grid, dtype=np.int8)
    id_psi    = np.zeros_like(psi_grid, dtype=np.int32)

    for i in range(np.shape(lsm_psi)[0]):
        for j in range(np.shape(lsm_psi)[1]):
            if j == np.shape(lsm_psi)[1]-1:
                # Wraparound (special case for last column)
                val = (lsm_rho[i, j] +
                       lsm_rho[i + 1, j] +
                       lsm_rho[i, 0] +
                       lsm_rho[i + 1, 0])
            else:
                val = (lsm_rho[i, j] +
                       lsm_rho[i+1, j] +
                       lsm_rho[i, j+1] +
                       lsm_rho[i+1, j+1])

            if val == 4:
                lsm_psi[i, j] = 1
            elif val in [1, 2, 3]:
                coast_psi[i, j] = 1

            id_psi[i, j] = 10000*i + j

    return lsm_psi, coast_psi, id_psi

@numba.jit(nopython=True)
def cnorm(lsm_rho):
    # This function generates a vector field normal to the land-sea mask, at
    # coastal rho points

    cnormx = np.zeros_like(lsm_rho, dtype=np.float32)
    cnormy = np.zeros_like(lsm_rho, dtype=np.float32)

    for i in range(lsm_rho.shape[0]-2):
        for j in range(lsm_rho.shape[1]-2):
            if (i > 0)*(j > 0):
                # Extract 3x3 stencil
                block = lsm_rho[i-1:i+2, j-1:j+2]

                # Check if coastal rho point (is land AND has >0 adj. ocean)
                if (lsm_rho[i, j])*(np.sum(block) < 9):
                    dx = block[1, 2] - block[1, 0]
                    dy = block[2, 1] - block[0, 1]

                    # If dx or dy are nonzero, it is not a diagonal coast cell
                    if (np.abs(dx) + np.abs(dy) == 0):
                        dxy = block[2, 2] - block[0, 0]
                        dyx = block[2, 0] - block[0, 2]

                        dx = (dxy - dyx)
                        dy = (dxy + dyx)

                    # Normalise
                    mag = np.sqrt(dx**2 + dy**2)

                    if mag != 0:
                        cnormx[i, j] = -dx/mag
                        cnormy[i, j] = -dy/mag

    return cnormx, cnormy

def release_time(param, **kwargs):
    # Generate a list of release times in seconds from the start of the model
    # data

    try:
        if kwargs['mode'] == 'START':
            start_mode = True
            print('Releasing at start of month...')
        else:
            start_mode = False
            print('Releasing at end of month...')
    except:
        start_mode = True
        print('Releasing at end of month...')

    Year  = param['Y']
    Month = param['M']
    RPM   = param['RPM']
    release = param['release']

    days_in_month = monthrange(Year, Month)[-1]

    if start_mode:
        release_day = np.linspace(0, days_in_month, num=RPM+1)[:-1][release]
    else:
        release_day = np.linspace(0, days_in_month, num=RPM+1)[1:][release]

    release_time = datetime(year=Year, month=Month, day=1) + timedelta(days=release_day)

    try:
        if param['delay_start']:
            release_time = release_time + timedelta(hours=12)
    except:
        pass

    return release_time


def release_loc_ocean(param, fh):
    # Generates initial coordinates for particle releases at a single time
    # frame for the ocean. We release within a geographical region of interest
    # (taking into account the land mask) and assign the particle ID. The
    # particle ID can later be used to assign the fishing intensity value.

    pn  = param['pn']

    # Firstly load the requisite fields:
    # - rho & psi coordinates
    # - lsm_rho mask
    # - id_psi mask (cell ids)

    with Dataset(fh['grid'], mode='r') as nc:
        lon_psi = np.array(nc.variables['lon_psi'][:])
        lat_psi = np.array(nc.variables['lat_psi'][:])

        lon_rho = np.array(nc.variables['lon_rho'][:])

        id_psi    = np.array(nc.variables['source_id_psi'][:])
        lsm_psi   = np.array(nc.variables['lsm_psi'][:])

    marine_release_loc = np.ones_like(id_psi, dtype=np.int32)

    plt.imshow(lsm_psi)
    plt.scatter(2819, 934)

    # Mask with limits
    y_idx_max = np.searchsorted(lat_psi, param['lat_north'])
    y_idx_min = np.searchsorted(lat_psi, param['lat_south'])
    x_idx_max = np.searchsorted(lon_psi, param['lon_east'])
    x_idx_min = np.searchsorted(lon_psi, param['lon_west'])

    marine_release_loc[:y_idx_min, :] = 0
    marine_release_loc[y_idx_max:, :] = 0
    marine_release_loc[:, :x_idx_min] = 0
    marine_release_loc[:, x_idx_max:] = 0

    # Remove Mediterranean
    x_idx_max_med = np.searchsorted(lon_psi, 60)
    y_idx_min_med = np.searchsorted(lat_psi, 30)

    marine_release_loc[y_idx_min_med:, :x_idx_max_med] = 0

    # Mask with lsm
    marine_release_loc *= (1-lsm_psi)

    # We also need to calculate how many particles are being released within
    # this GFW cell (i.e. out of 144*pn_cell)

    # Assign a unique ID to each 1x1 degree (GFW) cell
    gfw_id = np.arange(np.shape(id_psi)[0]*np.shape(id_psi)[1]/144, dtype=np.int32)
    gfw_id = gfw_id.reshape((int(np.shape(id_psi)[0]/12), -1))

    # Expand to 1/12 grid and mask with lsm
    gfw_id12 = np.kron(gfw_id, np.ones((12,12), dtype=np.int32))
    gfw_id12[lsm_psi == 1] = -1
    gfw_id12[:y_idx_min, :] = -1
    gfw_id12[y_idx_max:, :] = -1
    gfw_id12[:, :x_idx_min] = -1
    gfw_id12[:, x_idx_max:] = -1

    gfw_uniques = np.unique(gfw_id12, return_counts=True)
    gfw_dict = dict(zip(gfw_uniques[0], gfw_uniques[1]))

    # Extract cell grid indices
    idx = list(np.where(marine_release_loc == 1))

    # Calculate the total number of particles
    nl  = idx[0].shape[0]  # Number of locations
    id_list = id_psi[tuple(idx)]

    pn_cell = pn**2
    pn_tot = nl*pn_cell

    print('')
    print('Total number of particles generated per release: ' + str(pn_tot))

    dX = lon_rho[1] - lon_rho[0]  # Grid spacing

    lon_out = np.zeros((pn_tot,), dtype=np.float64)
    lat_out = np.zeros((pn_tot,), dtype=np.float64)
    id_out = np.zeros((pn_tot,), dtype=np.int32)
    np_per_gfw_out = np.zeros((pn_tot,), dtype=np.int32)

    for loc in range(nl):
        # Find cell location
        loc_yidx = idx[0][loc]
        loc_xidx = idx[1][loc]

        # Calculate initial positions
        dx = dX/pn                # Particle spacing
        gx = np.linspace((-dX/2 + dx/2), (dX/2 - dx/2), num=pn)
        gridx, gridy = [grid.flatten() for grid in np.meshgrid(gx, gx)]

        loc_y = lat_psi[loc_yidx]
        loc_x = lon_psi[loc_xidx]

        loc_id = id_psi[loc_yidx, loc_xidx]
        gfw_id_cell = gfw_id12[loc_yidx, loc_xidx]

        s_idx = loc*pn_cell
        e_idx = (loc+1)*pn_cell

        lon_out[s_idx:e_idx] = gridx + loc_x
        lat_out[s_idx:e_idx] = gridy + loc_y
        id_out[s_idx:e_idx] = np.ones(np.shape(gridx), dtype=np.int32)*loc_id
        np_per_gfw_out[s_idx:e_idx] = np.ones(np.shape(gridx), dtype=np.int32)*gfw_dict[gfw_id_cell]

    pos0 = {'lon': lon_out,
            'lat': lat_out,
            'id': id_out,
            'gfw': np_per_gfw_out}

    return pos0


def release_loc_land(param, fh):
    # Generates initial coordinates for particle releases at a single time
    # frame based on the ISO codes for countries of interest.

    ids = param['source_iso_list']

    # Firstly load the requisite fields:
    # - rho & psi coordinates
    # - lsm_rho mask
    # - id_psi mask (cell ids)
    # - coast_id mask (country codes)

    with Dataset(fh['grid'], mode='r') as nc:
        lon_psi = np.array(nc.variables['lon_psi'][:])
        lat_psi = np.array(nc.variables['lat_psi'][:])

        lon_rho = np.array(nc.variables['lon_rho'][:])

        iso_psi   = np.array(nc.variables['iso_psi'][:])
        lsm_psi   = np.array(nc.variables['lsm_psi'][:])
        cp_psi    = np.array(nc.variables['cplastic_psi'][:])
        rp_psi    = np.array(nc.variables['rplastic_psi'][:])

    # Now find the cells matching the provided ISO codes
    cp_psi[~np.isin(iso_psi, ids)] = 0
    rp_psi[~np.isin(iso_psi, ids)] = 0

    # Filtering
    lon_psi_grid, lat_psi_grid = np.meshgrid(lon_psi, lat_psi)
    # Remove Mediterannean coasts of Egypt
    cp_psi[(lon_psi_grid > 23)*
           (lon_psi_grid < 35)*
           (lat_psi_grid > 30)] = 0
    rp_psi[(lon_psi_grid > 23)*
           (lon_psi_grid < 35)*
           (lat_psi_grid > 30)] = 0

    # Remove everything below 60S
    cp_psi[(lat_psi_grid < -60)] = 0
    rp_psi[(lat_psi_grid < -60)] = 0

    # Remove northern part of Brazil
    cp_psi[(lon_psi_grid > -55)*
           (lon_psi_grid < -30)*
           (lat_psi_grid > -10)] = 0
    rp_psi[(lon_psi_grid > -55)*
           (lon_psi_grid < -30)*
           (lat_psi_grid > -10)] = 0

    # Now distribute plastic flux per grid cell across particles
    # Apply plastic threshold
    print('')
    print('Total cells with plastic: ' + str(np.count_nonzero(cp_psi + rp_psi)))

    tot_cp = np.sum(cp_psi)
    tot_rp = np.sum(rp_psi)

    cp_psi[(cp_psi/10) + rp_psi < param['threshold']] = 0
    rp_psi[(cp_psi/10) + rp_psi < param['threshold']] = 0

    print('Cells remaining after threshold: ' + str(np.count_nonzero(cp_psi + rp_psi)))
    print('Riverine plastic remaining after threshold: ' + str(100*np.sum(rp_psi)/tot_rp) + '%')
    print('Direct coastal plastic remaining after threshold: ' + str(100*np.sum(cp_psi)/tot_cp) + '%')

    if param['plot_input'] == 'all':
        with Dataset(fh['ref_o'], mode='r') as nc:
            ref_o_lon = nc.variables['longitude'][:]
            ref_o_lat = nc.variables['latitude'][:]
            ref_o_uo = nc.variables['uo'][:, 0, :, :]
            ref_o_vo = nc.variables['vo'][:, 0, :, :]

        with Dataset(fh['ref_w'], mode='r') as nc:
            subset_ref = 12
            ref_w_lon = nc.variables['longitude'][:]
            ref_w_lat = nc.variables['latitude'][:]
            ref_w_u10 = nc.variables['u10'][:, :, :]
            ref_w_v10 = nc.variables['v10'][:, :, :]

        f = plt.figure(figsize=(48, 28), constrained_layout=True)
        gs = GridSpec(2, 4, figure=f, width_ratios=[1, 0.02, 1, 0.02])
        ax = []
        ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # Currents (W)
        ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree())) # Currents (S)
        ax.append(f.add_subplot(gs[:, 1])) # Colorbar for currents
        ax.append(f.add_subplot(gs[0, 2], projection = ccrs.PlateCarree())) # River input
        ax.append(f.add_subplot(gs[1, 2], projection = ccrs.PlateCarree())) # Coastal input
        ax.append(f.add_subplot(gs[:, 3])) # Colorbar for plastics

        gl = []
        scatter = []
        cplot = []

        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                edgecolor='w',
                                                facecolor='w',
                                                linewidth=0.5,
                                                zorder=1)

        for i in [0, 1, 3, 4]:
            ax[i].set_aspect(1)
            ax[i].set_facecolor('w')
            ax[i].set_xlim([20, 140])
            ax[i].set_ylim([-40, 40])

        # Plot currents + wind
        winter_uo = (ref_o_uo[11, :, ]+ref_o_uo[0, :, ]+ref_o_uo[1, :, ])/3
        summer_uo = (ref_o_uo[5, :, ]+ref_o_uo[6, :, ]+ref_o_uo[7, :, ])/3
        winter_vo = (ref_o_vo[11, :, ]+ref_o_vo[0, :, ]+ref_o_vo[1, :, ])/3
        summer_vo = (ref_o_vo[5, :, ]+ref_o_vo[6, :, ]+ref_o_vo[7, :, ])/3

        winter_u10 = (ref_w_u10[11, :, ]+ref_w_u10[0, :, ]+ref_w_u10[1, :, ])/3
        summer_u10 = (ref_w_u10[5, :, ]+ref_w_u10[6, :, ]+ref_w_u10[7, :, ])/3
        winter_v10 = (ref_w_v10[11, :, ]+ref_w_v10[0, :, ]+ref_w_v10[1, :, ])/3
        summer_v10 = (ref_w_v10[5, :, ]+ref_w_v10[6, :, ]+ref_w_v10[7, :, ])/3

        winter_o_vel = np.sqrt(winter_uo**2 + winter_vo**2)
        summer_o_vel = np.sqrt(summer_uo**2 + summer_vo**2)

        cplot.append(ax[0].pcolormesh(ref_o_lon, ref_o_lat, winter_o_vel, cmap=cmr.ember_r,
                                      vmin=0, vmax=1.0, transform=ccrs.PlateCarree(),
                                      rasterized=True))
        ax[0].quiver(ref_w_lon[::subset_ref], ref_w_lat[::subset_ref],
                      winter_u10[::subset_ref, ::subset_ref],
                      winter_v10[::subset_ref, ::subset_ref], color='w',
                      scale=2e2, scale_units='height', width=2e-3, headwidth=4,
                      headlength=4)

        ax[0].add_feature(land_10m)
        gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='k', linestyle='-', zorder=11))
        gl[0].xlocator = mticker.FixedLocator(np.arange(-200, 200, 20))
        gl[0].ylocator = mticker.FixedLocator(np.arange(-80, 120, 20))
        gl[0].xlabels_top = False
        gl[0].ylabels_right = False
        gl[0].ylabel_style = {'size': 36}
        gl[0].xlabel_style = {'size': 36}

        ax[0].text(138, -38, 'Northeast Monsoon (DJF)', fontsize=48, color='k', zorder=20,
                   fontweight='bold', ha='right')

        cplot.append(ax[1].pcolormesh(ref_o_lon, ref_o_lat, summer_o_vel, cmap=cmr.ember_r,
                                      vmin=0, vmax=1.0, transform=ccrs.PlateCarree(),
                                      rasterized=True))

        ax[1].quiver(ref_w_lon[::subset_ref], ref_w_lat[::subset_ref],
                     summer_u10[::subset_ref, ::subset_ref],
                     summer_v10[::subset_ref, ::subset_ref], color='w',
                     scale=2e2, scale_units='height', width=2e-3, headwidth=4,
                     headlength=4)

        ax[1].add_feature(land_10m)
        gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='k', linestyle='-', zorder=11))
        gl[1].xlocator = mticker.FixedLocator(np.arange(-200, 200, 20))
        gl[1].ylocator = mticker.FixedLocator(np.arange(-80, 120, 20))
        gl[1].xlabels_top = False
        gl[1].ylabels_right = False
        gl[1].ylabel_style = {'size': 36}
        gl[1].xlabel_style = {'size': 36}

        ax[1].text(138, -38, 'Southwest monsoon (JJA)', fontsize=48, color='k', zorder=20,
                   fontweight='bold', ha='right')


        # Colorbar
        cb0 = plt.colorbar(cplot[1], cax=ax[2])
        cb0.set_label('Mean surface current velocity (m/s)', size=42)
        ax[2].tick_params(axis='y', labelsize=36)

        # Mask plastic input
        cp_lon_list = np.ma.masked_where(cp_psi<=1e6, lon_psi_grid).compressed()
        cp_lat_list = np.ma.masked_where(cp_psi<=1e6, lat_psi_grid).compressed()
        cp_list = np.ma.masked_where(cp_psi<=1e6, cp_psi).compressed()*0.25/1e3
        rp_lon_list = np.ma.masked_where(rp_psi<=1e5, lon_psi_grid).compressed()
        rp_lat_list = np.ma.masked_where(rp_psi<=1e5, lat_psi_grid).compressed()
        rp_list = np.ma.masked_where(rp_psi<=1e5, rp_psi).compressed()/1e3

        norm = colors.LogNorm(vmin=1e2, vmax=1e4)
        cmcp = cmr.torch(norm(cp_list))
        cmrp = cmr.torch(norm(rp_list))

        # Coastal
        scatter.append(ax[3].scatter(cp_lon_list, cp_lat_list, s=75*(np.log(cp_list)-4.5)**2,
                                     marker='o', linewidths=1, facecolors='none',
                                     edgecolors=cmcp, zorder=5))

        ax[3].add_feature(land_10m)
        gl.append(ax[3].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='white', linestyle='-', zorder=11))
        gl[2].xlocator = mticker.FixedLocator(np.arange(-200, 200, 20))
        gl[2].ylocator = mticker.FixedLocator(np.arange(-80, 120, 20))
        gl[2].xlabels_top = False
        gl[2].ylabels_right = False
        gl[2].ylabel_style = {'size': 36}
        gl[2].xlabel_style = {'size': 36}

        ax[3].text(138, -38, 'Direct coastal input', fontsize=48, color='w', zorder=20,
                   fontweight='bold', ha='right')

        # Riverine
        scatter.append(ax[4].scatter(rp_lon_list, rp_lat_list, s=75*(np.log(rp_list)-4.5)**2,
                                     marker='o', linewidths=1, facecolors='none',
                                     edgecolors=cmrp, zorder=5))

        ax[4].add_feature(land_10m)
        gl.append(ax[4].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='white', linestyle='-', zorder=11))
        gl[3].xlocator = mticker.FixedLocator(np.arange(-200, 200, 20))
        gl[3].ylocator = mticker.FixedLocator(np.arange(-80, 120, 20))
        gl[3].xlabels_top = False
        gl[3].ylabels_right = False
        gl[3].ylabel_style = {'size': 36}
        gl[3].xlabel_style = {'size': 36}

        ax[4].text(138, -38, 'Riverine input', fontsize=48, color='w', zorder=20,
                   fontweight='bold', ha='right')

        # Colorbar
        cb1 = plt.colorbar(ScalarMappable(norm=norm, cmap=cmr.torch),
                           cax=ax[5], orientation='vertical')
        cb1.set_label('Annual plastic flux (tonnes)', size=42)
        ax[5].tick_params(axis='y', labelsize=36)
        ax[5].minorticks_on()
        # ax[5].tick_params(which='major', length=15)
        # ax[5].tick_params(which='minor', length=10)

        s_lons = [46.35, 55.47, 39.8, 53.9, 72.75, 73.25, 72.0, 57.6, 43.5]
        s_lats = [-9.4, -4.67, -5.15, 12.5, 11.5, 3.75, -6.25, -20.25, -12]
        s_names = ['Aldabra', 'Mahé', 'Pemba', 'Socotra', 'Lakshadweep', 'Maldives', 'Chagos', 'Mauritius', 'Comoros']

        for i in [3, 4]:
            for s_lon, s_lat, s_name in zip(s_lons, s_lats, s_names):
                ax[i].plot(s_lon, s_lat, marker='D', ms=15, color='w', zorder=14)
                ax[i].text(s_lon+0.5, s_lat+0.5, s_name, fontsize=36, color='w', zorder=15)

        plt.savefig(fh['fig'] + '.pdf', dpi=200)
        print('MD input figure exported!')
    elif param['plot_input'] == 'plastic':

        f = plt.figure(constrained_layout=True, figsize=(22, 28))
        gs = GridSpec(2, 2, figure=f, width_ratios=[1, 0.03])
        ax = []
        ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())) # River input
        ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree())) # Coastal input
        ax.append(f.add_subplot(gs[:, 1])) # Colorbar for plastics

        gl = []
        scatter = []
        cplot = []

        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                edgecolor='white',
                                                facecolor='black',
                                                linewidth=0.5,
                                                zorder=1)

        for i in [0, 1]:
            ax[i].set_aspect(1)
            ax[i].set_facecolor('k')
            ax[i].set_xlim([20, 140])
            ax[i].set_ylim([-40, 40])

        # Mask plastic input
        cp_lon_list = np.ma.masked_where(cp_psi<=1e6, lon_psi_grid).compressed()
        cp_lat_list = np.ma.masked_where(cp_psi<=1e6, lat_psi_grid).compressed()
        cp_list = np.ma.masked_where(cp_psi<=1e6, cp_psi).compressed()*0.25/1e3
        rp_lon_list = np.ma.masked_where(rp_psi<=1e5, lon_psi_grid).compressed()
        rp_lat_list = np.ma.masked_where(rp_psi<=1e5, lat_psi_grid).compressed()
        rp_list = np.ma.masked_where(rp_psi<=1e5, rp_psi).compressed()/1e3

        norm = colors.LogNorm(vmin=1e2, vmax=1e4)
        cmcp = cmr.torch(norm(cp_list))
        cmrp = cmr.torch(norm(rp_list))

        # Coastal
        scatter.append(ax[0].scatter(cp_lon_list, cp_lat_list, s=75*(np.log(cp_list)-4.5)**2,
                                     marker='o', linewidths=1, facecolors='none',
                                     edgecolors=cmcp, zorder=5))

        ax[0].add_feature(land_10m)
        gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='white', linestyle='-', zorder=11))
        gl[0].xlocator = mticker.FixedLocator(np.arange(-200, 200, 20))
        gl[0].ylocator = mticker.FixedLocator(np.arange(-80, 120, 20))
        gl[0].xlabels_top = False
        gl[0].ylabels_right = False
        gl[0].ylabel_style = {'size': 28}
        gl[0].xlabel_style = {'size': 28}

        ax[0].text(23, -37, 'Direct coastal input', fontsize=48, color='w', zorder=20,
                   fontweight='bold', ha='left')

        # Riverine
        scatter.append(ax[1].scatter(rp_lon_list, rp_lat_list, s=75*(np.log(rp_list)-4.5)**2,
                                     marker='o', linewidths=1, facecolors='none',
                                     edgecolors=cmrp, zorder=5))

        ax[1].add_feature(land_10m)
        gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='white', linestyle='-', zorder=11))
        gl[1].xlocator = mticker.FixedLocator(np.arange(-200, 200, 20))
        gl[1].ylocator = mticker.FixedLocator(np.arange(-80, 120, 20))
        gl[1].xlabels_top = False
        gl[1].ylabels_right = False
        gl[1].ylabel_style = {'size': 28}
        gl[1].xlabel_style = {'size': 28}

        ax[1].text(23, -37, 'Riverine input', fontsize=48, color='w', zorder=20,
                   fontweight='bold', ha='left')

        # Colorbar
        cb1 = plt.colorbar(ScalarMappable(norm=norm, cmap=cmr.torch),
                           cax=ax[2], orientation='vertical')
        cb1.set_label('Annual plastic flux (tonnes)', size=28)
        ax[2].tick_params(axis='y', labelsize=24)
        ax[2].minorticks_on()

        s_lons = [46.35, 55.47, 39.8, 53.9, 72.75, 73.25, 72.0, 57.6, 43.5]
        s_lats = [-9.4, -4.67, -5.15, 12.5, 11.5, 3.75, -6.25, -20.25, -12]
        s_names = ['Aldabra', 'Mahé', 'Pemba', 'Socotra', 'Lakshadweep', 'Maldives', 'Chagos', 'Mauritius', 'Comoros']

        for i in [0, 1]:
            for s_lon, s_lat, s_name in zip(s_lons, s_lats, s_names):
                ax[i].plot(s_lon, s_lat, marker='D', ms=15, color='w', zorder=14)
                ax[i].text(s_lon+0.5, s_lat+0.5, s_name, fontsize=22, color='w', zorder=15)

        plt.savefig(fh['fig'] + '.pdf', dpi=300)
        print('MD input figure exported!')

    # Extract cell grid indices
    p_psi = (cp_psi/10) + rp_psi # Assume 10% of coastal plastic enters for this calculation
    idx = list(np.where(p_psi > 0)) # Contains grid indices of all grid cells with plastic above the threshold

    # Calculate the total number of particles
    nl  = idx[0].shape[0]  # Number of locations
    p_psi_list = cp_psi[tuple(idx)]/10 + rp_psi[tuple(idx)]

    # Calculate number of particles per cell using the following formula:
    # pn = (m*(log10[total_plastic]-log10[plastic_threshold]+(1/3)))**2,
    # i.e. all cells have at least (m/3)**2 particles
    const = np.log10(param['threshold'])-(1/3)
    pn_list = np.array(np.ceil(param['log_mult']*(np.log10(p_psi_list)-const)), dtype=np.int)
    pn_list_cs = np.cumsum(pn_list**2)
    pn_list_cs = np.concatenate((np.array([0]), pn_list_cs))
    pn_tot = pn_list_cs[-1]

    print('')
    print('Total number of particles generated per release: ' + str(pn_tot))

    # For cell psi[i, j], the surrounding rho cells are:
    # rho[i, j]     (SW)
    # rho[i, j+1]   (SE)
    # rho[i+1, j]   (NW)
    # rho[i+1, j+1] (NE)

    # For each psi cell of interest, we are now going to:
    # 1. Find the coordinates of the surrounding rho points
    # 2. Find the land mask at those surrounding rho points
    # 3. Calculate the valid part of that psi cell to populate
    # 4. Calculate the coordinates of the resulting particles

    dX = lon_rho[1] - lon_rho[0]  # Grid spacing

    lon_out = np.zeros((pn_tot,), dtype=np.float64)
    lat_out = np.zeros((pn_tot,), dtype=np.float64)
    cp0_out = np.zeros((pn_tot,), dtype=np.float32)
    rp0_out = np.zeros((pn_tot,), dtype=np.float32)
    iso_out = np.zeros((pn_tot,), dtype=np.int16)

    for loc in range(nl):
        # Find cell location
        loc_yidx = idx[0][loc]
        loc_xidx = idx[1][loc]

        # Calculate initial plastic and particle number required
        cp_cell = cp_psi[loc_yidx, loc_xidx]
        rp_cell = rp_psi[loc_yidx, loc_xidx]

        pn_cell = int(np.ceil(param['log_mult']*(np.log10(cp_cell/10 + rp_cell)-const)))

        cp_part = cp_cell/pn_cell**2
        rp_part = rp_cell/pn_cell**2

        # Calculate initial positions
        dx = dX/pn_cell                    # Particle spacing
        gx = np.linspace((-dX/2 + dx/2), (dX/2 - dx/2), num=pn_cell)
        gridx, gridy = [grid.flatten() for grid in np.meshgrid(gx, gx)]

        loc_y = lat_psi[loc_yidx]
        loc_x = lon_psi[loc_xidx]

        loc_iso = iso_psi[loc_yidx, loc_xidx]

        s_idx = pn_list_cs[loc]
        e_idx = pn_list_cs[loc+1]

        lon_out[s_idx:e_idx] = gridx + loc_x
        lat_out[s_idx:e_idx] = gridy + loc_y
        cp0_out[s_idx:e_idx] = np.ones(np.shape(gridx), dtype=np.float32)*cp_part
        rp0_out[s_idx:e_idx] = np.ones(np.shape(gridx), dtype=np.float32)*rp_part
        iso_out[s_idx:e_idx] = np.ones(np.shape(gridx), dtype=np.int16)*loc_iso

    pos0 = {'lon': lon_out,
            'lat': lat_out,
            'iso': iso_out,
            'cp0': cp0_out,
            'rp0': rp0_out}

    return pos0

def release_loc_sey(param, fh):
    # Generates initial coordinates for particle releases at a single time
    # frame based on the ISO codes for countries of interest. Particles are
    # only released at least 0.5 cells away from the land mask.

    ids = param['id']
    pn  = param['pn']

    # Convert pn to the number along a square
    pn = int(np.ceil(pn**0.5))

    # Firstly load the requisite fields:
    # - rho & psi coordinates
    # - lsm_rho mask
    # - id_psi mask (cell ids)
    # - coast_id mask (country codes)

    with Dataset(fh['grid'], mode='r') as nc:
        lon_psi = np.array(nc.variables['lon_psi'][:])
        lat_psi = np.array(nc.variables['lat_psi'][:])

        lon_rho = np.array(nc.variables['lon_rho'][:])

        id_psi    = np.array(nc.variables['source_id_psi'][:])
        iso_psi   = np.array(nc.variables['iso_psi'][:])

    # Now find the cells matching the provided ISO codes
    idx = np.where(np.isin(iso_psi, ids))

    print()

    # For cell psi[i, j], the surrounding rho cells are:
    # rho[i, j]     (SW)
    # rho[i, j+1]   (SE)
    # rho[i+1, j]   (NW)
    # rho[i+1, j+1] (NE)

    # For each psi cell of interest, we are now going to:
    # 1. Find the coordinates of the surrounding rho points
    # 2. Find the land mask at those surrounding rho points
    # 3. Calculate the valid part of that psi cell to populate
    # 4. Calculate the coordinates of the resulting particles

    # Firstly calculate the basic particle grid
    dX = lon_rho[1] - lon_rho[0]  # Grid spacing
    dx = dX/pn                    # Particle spacing
    gx = gy = np.linspace((-dX/2 + dx/2), (dX/2 - dx/2), num=pn)
    gridx, gridy = [grid.flatten() for grid in np.meshgrid(gx, gy)]

    nl  = idx[0].shape[0]  # Number of locations

    lon_out = np.array([], dtype=np.float64)
    lat_out = np.array([], dtype=np.float64)
    id_out  = np.array([], dtype=np.int32)
    iso_out  = np.array([], dtype=np.int16)

    for loc in range(nl):
        loc_yidx = idx[0][loc]
        loc_xidx = idx[1][loc]

        loc_y = lat_psi[loc_yidx]
        loc_x = lon_psi[loc_xidx]

        loc_id   = id_psi[loc_yidx, loc_xidx]
        loc_iso  = iso_psi[loc_yidx, loc_xidx]

        lon_out = np.append(lon_out, gridx + loc_x)
        lat_out = np.append(lat_out, gridy + loc_y)
        iso_out = np.append(iso_out, np.ones(np.shape(gridx),
                                             dtype=np.int16)*loc_iso)
        id_out = np.append(id_out, np.ones(np.shape(gridx),
                                           dtype=np.int32)*loc_id)

    # Now distribute trajectories across processes
    # proc_out = np.repeat(np.arange(param['nproc'],
    #                                dtype=(np.int16 if param['nproc'] > 123 else np.int8)),
    #                      int(np.ceil(len(id_out)/param['nproc'])))
    # proc_out = proc_out[:len(id_out)]


    pos0 = {'lon': lon_out,
            'lat': lat_out,
            'iso': iso_out,
            'id': id_out}

    return pos0

def add_times(particles, param):
    # This script takes release locations at a particular time point and
    # duplicates it across multiple times

    data  = particles['loc_array']
    times = particles['time_array']

    npart  = data['lon'].shape[0]
    data['time'] = np.repeat(times, npart)

    # Also now calculation partitioning of particles
    if param['total_partitions'] > 1:
        pn_per_part = int(np.ceil(npart/param['total_partitions']))
        i0 = pn_per_part*param['partition']
        i1 = pn_per_part*(param['partition']+1)

        data['lon'] = data['lon'][i0:i1]
        data['lat'] = data['lat'][i0:i1]
        data['time'] = data['time'][i0:i1]

        try:
            data['gfw']   = data['gfw'][i0:i1]
        except:
            pass

        try:
            data['iso'] = data['iso'][i0:i1]
            data['cp0'] = data['cp0'][i0:i1]
            data['rp0'] = data['rp0'][i0:i1]
        except:
            pass

    return data


def gridgen(fh, dirs, param, **kwargs):
    # This function takes output from global CMEMS GLORYS12V1 and generates
    # Labelled coast cells
    # A land mask
    # Cell boundaries (boundary points)
    # Country IDs

    # Also optionally generates plastic coastal accumulation rates

    try:
        if kwargs['plastic']:
            plastic = True
        else:
            plastic = False
    except:
        plastic = False

    # File handles
    try:
        model_fh = fh['ocean'][0]
    except:
        model_fh = fh['ocean']

    grid_fh  = fh['grid']

    # Note:
    # Velocities are defined on the rho grid
    # Land-sea mask/coast are defined on the psi grid

    if os.path.isfile(grid_fh):
        with Dataset(grid_fh, mode='r') as nc:
            if plastic:
                print('Grid file found, loading data (with plastics)')
                grid = {'coast_psi': np.array(nc.variables['coast_psi'][:]),
                            'lsm_psi': np.array(nc.variables['lsm_psi'][:]),
                            'lsm_rho': np.array(nc.variables['lsm_rho'][:]),
                            'lon_rho': np.array(nc.variables['lon_rho'][:]),
                            'lon_psi': np.array(nc.variables['lon_psi'][:]),
                            'lat_rho': np.array(nc.variables['lat_rho'][:]),
                            'lat_psi': np.array(nc.variables['lat_psi'][:]),
                            'cdist_rho': np.array(nc.variables['cdist_rho'][:]),
                            'cnormx_rho': np.array(nc.variables['cnormx_rho'][:]),
                            'cnormy_rho': np.array(nc.variables['cnormy_rho'][:]),
                            'source_id_psi': np.array(nc.variables['source_id_psi']),
                            'sink_id_psi': np.array(nc.variables['sink_id_psi']),
                            'iso_psi': np.array(nc.variables['iso_psi']),
                            'cplastic_psi': np.array(nc.variables['cplastic_psi']),
                            'rplastic_psi': np.array(nc.variables['rplastic_psi'])}
            else:
                print('Grid file found, loading data (without plastics)')
                grid = {'coast_psi': np.array(nc.variables['coast_psi'][:]),
                            'lsm_psi': np.array(nc.variables['lsm_psi'][:]),
                            'lsm_rho': np.array(nc.variables['lsm_rho'][:]),
                            'lon_rho': np.array(nc.variables['lon_rho'][:]),
                            'lon_psi': np.array(nc.variables['lon_psi'][:]),
                            'lat_rho': np.array(nc.variables['lat_rho'][:]),
                            'lat_psi': np.array(nc.variables['lat_psi'][:]),
                            'cdist_rho': np.array(nc.variables['cdist_rho'][:]),
                            'cnormx_rho': np.array(nc.variables['cnormx_rho'][:]),
                            'cnormy_rho': np.array(nc.variables['cnormy_rho'][:]),
                            'source_id_psi': np.array(nc.variables['source_id_psi'])}

    else:
        print('Grid file not found, building grid file')
        with Dataset(model_fh, mode='r') as nc:
            lon_rho = np.array(nc.variables['longitude'][:])
            lat_rho = np.array(nc.variables['latitude'][:])

            uo  = np.array(nc.variables['uo'][0, 0, :, :])
            fill = nc.variables['uo']._FillValue

        # Generate the global psi grid
        lat_psi = np.linspace(-80 + (1/24), 90 - (1/24), num=170*12)
        lon_psi = np.linspace(-180 + (1/24), 180 - (1/24), num=360*12)

        # Now generate the mask on the lat/lon (rho) grid
        lsm_rho = np.zeros_like(uo, dtype=np.int8)
        lsm_rho[uo == fill] = 1  # Set land cells to 1

        # Now generate the ('true') mask on the lat/lon (psi) grid:
        # A node on the psi grid is 'land' if all 4 surrounding nodes on the
        # rho grid are also land
        # Also generate the coast mask on the psi grid, defined as coast if
        # 1, 2 or 3 surrounding nodes are land.
        psi_grid = np.zeros((len(lat_psi), len(lon_psi)))
        lsm_psi, coast_psi, id_psi = psi_mask(lsm_rho, psi_grid)

        # Now add the custom coast tiles for Seychelles if applicable
        try:
            add_sey = True if kwargs['add_seychelles'] == True else False
        except:
            add_sey = False

        if add_sey:
            print('Seychelles extra islands added.')
            island_loc = np.array([[-3.805, 55.667], # Denis
                                   [-3.720, 55.206], # Bird
                                   [-5.758, 53.307], # Poivre
                                   [-7.021, 52.737], # Alphonse
                                   [-9.225, 51.050], # Providence
                                   [-9.525, 50.975], # Providence
                                   [-4.485, 55.220], # Silhouette
                                   [-4.585, 55.940], # Fregate
                                   [-7.130, 56.270], # Coetivy
                                   [-9.720, 46.510], # Assomption
                                   [-10.09, 47.740], # Astove
                                   [-5.430, 53.350], # St. Joseph/D'Arros
                                   [-5.690, 53.675], # Desroches
                                   [-5.850, 55.390], # Platte
                                   ])

            island_loc = np.histogram2d(island_loc[:, 0],
                                        island_loc[:, 1],
                                        bins=[lat_rho,
                                              np.append(lon_rho, 180)])[0]
            island_loc = island_loc.astype('int8')
            island_loc[island_loc > 1] = 1

            # Check there are no shared cells
            if np.sum(island_loc*coast_psi) > 0:
                raise ValueError('Added islands overlap with existing LSM')

            coast_psi += island_loc

        # Now generate a rho grid with the distance to the coast
        cdist_rho = distance_transform_edt(1-lsm_rho)

        # Now generate vectors pointing away from the coast
        cnormx, cnormy = cnorm(lsm_rho)

        cdist_rho_land = distance_transform_edt(lsm_rho)
        cnormx_supp = -np.gradient(cdist_rho_land, axis=1)
        cnormy_supp = -np.gradient(cdist_rho_land, axis=0)
        cnorm_supp_mag = np.sqrt(cnormx_supp**2 + cnormy_supp**2)
        cnorm_supp_mag[cnorm_supp_mag == 0] = 1
        cnormx_supp /= cnorm_supp_mag
        cnormy_supp /= cnorm_supp_mag

        cnormx_supp[(cnormx != 0) + (lsm_rho == 0)] = 0
        cnormy_supp[(cnormy != 0) + (lsm_rho == 0)] = 0

        cnormx += cnormx_supp
        cnormy += cnormy_supp

        # Now generate plastic data if required
        if plastic:
            efs = 1/param['p_param']['l']
            cr  = param['p_param']['cr']

            plastic_dir = dirs['plastic']

            # Lebreton & Andrady plastic waste generation data
            # https://www.nature.com/articles/s41599-018-0212-7
            plastic_fh = plastic_dir + 'LebretonAndrady2019_MismanagedPlasticWaste.tif'

            # Country ID grid(UN WPP-Adjusted Population Density v4.11)
            # https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse
            id_fh      = plastic_dir + 'gpw_v4_national_identifier_grid_rev11_30_sec.tif'

            # Meijer et al river plastic data
            # https://advances.sciencemag.org/content/7/18/eaaz5803/tab-figures-data
            river_fh   = plastic_dir + 'Meijer2021_midpoint_emissions.zip'

            ##############################################################################
            # Coastal plastics ###########################################################
            ##############################################################################


            # Methodology
            # 1. Upscale plastic/id data to 1/12 grid
            # 2. Calculate nearest coastal cell on CMEMS grid
            # 3. Calculate the distance of each cell to coastal cell with Haversine
            # 4. Calculate the waste flux at each coastal cell (sum of weighted land cells)

            lon_bnd = np.append(lon_rho, 180)
            lat_bnd = lat_rho

            # Load plastic data
            plastic_data_obj = gdal.Open(plastic_fh)
            plastic_data = plastic_data_obj.ReadAsArray()[:-1200, :]

            # Load country codes
            country_id_obj = gdal.Open(id_fh)
            country_id = country_id_obj.ReadAsArray()[:-1200, :]

            # 1. Upscale data to 1/12 grid
            plastic_data = block_reduce(plastic_data, (10,10), np.sum)
            plastic_data = np.float64(np.flipud(plastic_data))
            country_id = block_reduce(country_id, (10,10), np.max)
            country_id = np.flipud(country_id)

            # 2. Calculate nearest coastal cells
            cnearest = distance_transform_edt(1-coast_psi,
                                              return_distances=False,
                                              return_indices=True)


            # 3. Calculate the distance of each waste flux cell to the nearest coastal
            #    cell
            lon_psi_grd, lat_psi_grd = np.meshgrid(lon_psi, lat_psi)
            cdist = cdist_calc(cnearest, lon_psi_grd, lat_psi_grd, efs)

            # Modify cells by W_mod = cr*exp(-(cdist*efs)^2)
            # Note that this parameterisation is essentially arbitrary - there
            # do not appear to be any good constraints on this. Intentionally
            # using a small length scale to avoid overlap with inland river
            # sources (Lebreton dataset more comprehensive than data used for
            # previous river-based studies)
            plastic_data *= cr*np.exp(-(cdist*efs)**2)

            # 4. Calculate the waste flux at each coastal cell
            cplastic = cplastic_calc(plastic_data, cnearest)

            ##############################################################################
            # Riverine plastics ##########################################################
            ##############################################################################

            river_file = gpd.read_file(river_fh)
            river_data = np.array(river_file['dots_exten'])*1000.  # tons -> kg
            river_lon  = np.array(river_file['geometry'].x)
            river_lat  = np.array(river_file['geometry'].y)

            # Bin onto the 1/12 degree grid
            rplastic   = np.histogram2d(river_lon, river_lat,
                                        bins=[lon_bnd, lat_bnd],
                                        weights=river_data,
                                        normed=False)[0].T

            # Shift to coastal cells
            rplastic   = cplastic_calc(rplastic, cnearest)

            ##############################################################################
            # Label coastal cells ########################################################
            ##############################################################################

            # Label coastal cells (coast) with their ISO country identifier
            # Calculate the nearest country_id to each grid cell without a country_id
            # A coastal cell with a country id takes that country id, otherwise it takes
            # the nearest country id

            country_id_lsm = np.copy(country_id)
            country_id_lsm[country_id_lsm >= 0] = 0
            country_id_lsm[country_id_lsm <  0] = 1

            country_id_nearest = distance_transform_edt(country_id_lsm,
                                                        return_distances=False,
                                                        return_indices=True)

            iso_psi = id_coast_cells(coast_psi.astype('int16'), country_id, country_id_nearest)

            # Also add land cells
            iso_psi_land = np.copy(country_id)
            iso_psi_land[iso_psi_land == -32768] = 0
            iso_psi_land[lsm_psi == 0] = 0
            iso_psi_land[iso_psi != 0] = 0
            iso_psi_land += iso_psi

            ##############################################################################
            # Specifically label Seychelles coastal cells ################################
            ##############################################################################

            sink_id_psi = np.zeros_like(iso_psi)
            sink_loc = np.where(np.isin(iso_psi, 690))
            sink_id_psi[sink_loc] = 1

            # (1) Aldabra
            sink_id_psi[(lat_psi_grd > -9.5)*(lat_psi_grd < -9.0)*
                        (lon_psi_grd > 46.0)*(lon_psi_grd < 47.0)] *= 1

            # (2) Assomption
            sink_id_psi[(lat_psi_grd > -10.0)*(lat_psi_grd < -9.5)*
                        (lon_psi_grd > 46.0)*(lon_psi_grd < 47.0)] *= 2

            # (3) Cosmoledo
            sink_id_psi[(lat_psi_grd > -10.0)*(lat_psi_grd < -9.0)*
                        (lon_psi_grd > 47.0)*(lon_psi_grd < 48.0)] *= 3

            # (4) Astove
            sink_id_psi[(lat_psi_grd > -10.5)*(lat_psi_grd < -10.0)*
                        (lon_psi_grd > 47.0)*(lon_psi_grd < 48.0)] *= 4

            # (5) Providence
            sink_id_psi[(lat_psi_grd > -10.0)*(lat_psi_grd < -9.0)*
                        (lon_psi_grd > 50.5)*(lon_psi_grd < 51.5)] *= 5

            # (6) Farquhar
            sink_id_psi[(lat_psi_grd > -10.5)*(lat_psi_grd < -10.0)*
                        (lon_psi_grd > 50.5)*(lon_psi_grd < 51.5)] *= 6

            # (7) Alphonse
            sink_id_psi[(lat_psi_grd > -7.5)*(lat_psi_grd < -7.0)*
                        (lon_psi_grd > 52.0)*(lon_psi_grd < 53.0)] *= 7

            # (8) Poivre
            sink_id_psi[(lat_psi_grd > -6.0)*(lat_psi_grd < -5.5)*
                        (lon_psi_grd > 53.0)*(lon_psi_grd < 53.5)] *= 8

            # (9) St Joseph
            sink_id_psi[(lat_psi_grd > -5.5)*(lat_psi_grd < -5.25)*
                        (lon_psi_grd > 53.0)*(lon_psi_grd < 53.5)] *= 9

            # (10) Desroches
            sink_id_psi[(lat_psi_grd > -6.0)*(lat_psi_grd < -5.5)*
                        (lon_psi_grd > 53.5)*(lon_psi_grd < 54.0)] *= 10

            # (11) Platte
            sink_id_psi[(lat_psi_grd > -6.0)*(lat_psi_grd < -5.5)*
                        (lon_psi_grd > 55.0)*(lon_psi_grd < 55.5)] *= 11

            # (12) Coetivy
            sink_id_psi[(lat_psi_grd > -7.5)*(lat_psi_grd < -7.0)*
                        (lon_psi_grd > 56.0)*(lon_psi_grd < 56.5)] *= 12

            # (13) Mahe
            sink_id_psi[(lat_psi_grd > -5.0)*(lat_psi_grd < -4.5)*
                        (lon_psi_grd > 55.0)*(lon_psi_grd < 55.5)] *= 13

            # (14) Fregate
            sink_id_psi[(lat_psi_grd > -5.0)*(lat_psi_grd < -4.5)*
                        (lon_psi_grd > 55.5)*(lon_psi_grd < 56.0)] *= 14

            # (15) Silhouette
            sink_id_psi[(lat_psi_grd > -4.5)*(lat_psi_grd < -4.0)*
                        (lon_psi_grd > 55.0)*(lon_psi_grd < 55.5)] *= 15

            # (16) Praslin
            sink_id_psi[(lat_psi_grd > -4.5)*(lat_psi_grd < -4.0)*
                        (lon_psi_grd > 55.5)*(lon_psi_grd < 56.0)] *= 16

            # (17) Denis
            sink_id_psi[(lat_psi_grd > -4.0)*(lat_psi_grd < -3.5)*
                        (lon_psi_grd > 55.5)*(lon_psi_grd < 56.0)] *= 17

            # (18) Bird
            sink_id_psi[(lat_psi_grd > -4.0)*(lat_psi_grd < -3.5)*
                        (lon_psi_grd > 55.0)*(lon_psi_grd < 55.5)] *= 18

            # Now label other countries of interest
            for country in param['sink_iso_list']:
                sink_loc = np.where(np.isin(iso_psi, country))

                # Modifications to only include Socotra (Yemen), Lakshadweep (India), and Pemba (Tanzania)
                if np.isin(country, [356, 887, 834]):
                    if country == 356:
                        idx_lon_max = np.searchsorted(lon_psi, 74.5)
                        idx_lat_max = np.searchsorted(lat_psi, 13)
                        sink_loc_mod_i = sink_loc[0][(sink_loc[0] < idx_lat_max)*
                                                     (sink_loc[1] < idx_lon_max)]
                        sink_loc_mod_j = sink_loc[1][(sink_loc[0] < idx_lat_max)*
                                                     (sink_loc[1] < idx_lon_max)]
                    elif country == 887:
                        idx_lon_min = np.searchsorted(lon_psi, 53.1)
                        idx_lon_max = np.searchsorted(lon_psi, 54.7)
                        idx_lat_min = np.searchsorted(lat_psi, 12.1)
                        idx_lat_max = np.searchsorted(lat_psi, 12.9)
                        sink_loc_mod_i = sink_loc[0][(sink_loc[0] < idx_lat_max)*(sink_loc[0] > idx_lat_min)*
                                                     (sink_loc[1] < idx_lon_max)*(sink_loc[1] > idx_lon_min)]
                        sink_loc_mod_j = sink_loc[1][(sink_loc[0] < idx_lat_max)*(sink_loc[0] > idx_lat_min)*
                                                     (sink_loc[1] < idx_lon_max)*(sink_loc[1] > idx_lon_min)]
                    elif country == 834:
                        idx_lon_min = np.searchsorted(lon_psi, 39.5)
                        idx_lon_max = np.searchsorted(lon_psi, 40.0)
                        idx_lat_min = np.searchsorted(lat_psi, -5.6)
                        idx_lat_max = np.searchsorted(lat_psi, -4.7)
                        sink_loc_mod_i = sink_loc[0][(sink_loc[0] < idx_lat_max)*(sink_loc[0] > idx_lat_min)*
                                                     (sink_loc[1] < idx_lon_max)*(sink_loc[1] > idx_lon_min)]
                        sink_loc_mod_j = sink_loc[1][(sink_loc[0] < idx_lat_max)*(sink_loc[0] > idx_lat_min)*
                                                     (sink_loc[1] < idx_lon_max)*(sink_loc[1] > idx_lon_min)]

                    sink_id_psi[(sink_loc_mod_i, sink_loc_mod_j)] = country
                else:
                    sink_id_psi[sink_loc] = country

            ##############################################################################
            # Calculate total plastic budget #############################################
            ##############################################################################

            total_coastal_plastic = '{:3.2f}'.format(np.sum(cplastic)/1e9) + ' Mt yr-1'
            total_riverine_plastic = '{:3.2f}'.format(np.sum(rplastic)/1e9) + ' Mt yr-1'


        # Save to netcdf
        with Dataset(grid_fh, mode='w') as nc:
            # Create the dimensions
            nc.createDimension('lon_rho', len(lon_rho))
            nc.createDimension('lat_rho', len(lat_rho))
            nc.createDimension('lon_psi', len(lon_psi))
            nc.createDimension('lat_psi', len(lat_psi))

            nc.createVariable('lon_psi', 'f4', ('lon_psi'), zlib=True)
            nc.variables['lon_psi'].long_name = 'longitude_on_psi_points'
            nc.variables['lon_psi'].units = 'degrees_east'
            nc.variables['lon_psi'].standard_name = 'longitude_psi'
            nc.variables['lon_psi'][:] = lon_psi

            nc.createVariable('lat_psi', 'f4', ('lat_psi'), zlib=True)
            nc.variables['lat_psi'].long_name = 'latitude_on_psi_points'
            nc.variables['lat_psi'].units = 'degrees_north'
            nc.variables['lat_psi'].standard_name = 'latitude_psi'
            nc.variables['lat_psi'][:] = lat_psi

            nc.createVariable('lon_rho', 'f4', ('lon_rho'), zlib=True)
            nc.variables['lon_rho'].long_name = 'longitude_on_rho_points'
            nc.variables['lon_rho'].units = 'degrees_east'
            nc.variables['lon_rho'].standard_name = 'longitude_rho'
            nc.variables['lon_rho'][:] = lon_rho

            nc.createVariable('lat_rho', 'f4', ('lat_rho'), zlib=True)
            nc.variables['lat_rho'].long_name = 'latitude_on_rho_points'
            nc.variables['lat_rho'].units = 'degrees_north'
            nc.variables['lat_rho'].standard_name = 'latitude_rho'
            nc.variables['lat_rho'][:] = lat_rho

            nc.createVariable('lsm_psi', 'i2', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['lsm_psi'].long_name = 'land_sea_mask_on_psi_points'
            nc.variables['lsm_psi'].units = '1 = Land, 0 = Sea'
            nc.variables['lsm_psi'].standard_name = 'land_sea_mask_psi'
            nc.variables['lsm_psi'][:] = lsm_psi

            nc.createVariable('lsm_rho', 'i2', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['lsm_rho'].long_name = 'land_sea_mask_on_rho_points'
            nc.variables['lsm_rho'].units = '1 = Land, 0 = Sea'
            nc.variables['lsm_rho'].standard_name = 'land_sea_mask_rho'
            nc.variables['lsm_rho'][:] = lsm_rho

            nc.createVariable('cdist_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['cdist_rho'].long_name = 'distance_from_coast_rho_points'
            nc.variables['cdist_rho'].units = 'grid_cells'
            nc.variables['cdist_rho'].standard_name = 'cdist_rho'
            nc.variables['cdist_rho'][:] = cdist_rho

            nc.createVariable('cnormx_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['cnormx_rho'].long_name = 'normal_at_coast_rho_x_component'
            nc.variables['cnormx_rho'].units = 'no units'
            nc.variables['cnormx_rho'].standard_name = 'cnormx_rho'
            nc.variables['cnormx_rho'][:] = cnormx

            nc.createVariable('cnormy_rho', 'f4', ('lat_rho', 'lon_rho'), zlib=True)
            nc.variables['cnormy_rho'].long_name = 'normal_at_coast_rho_y_component'
            nc.variables['cnormy_rho'].units = 'no units'
            nc.variables['cnormy_rho'].standard_name = 'cnormy_rho'
            nc.variables['cnormy_rho'][:] = cnormy

            nc.createVariable('coast_psi', 'i2', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['coast_psi'].long_name = 'coast_mask_on_psi_points'
            nc.variables['coast_psi'].units = '1 = Coast, 0 = Not coast'
            nc.variables['coast_psi'].standard_name = 'coast_mask_psi'
            nc.variables['coast_psi'][:] = coast_psi

            nc.createVariable('source_id_psi', 'i4', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['source_id_psi'].long_name = 'source_cell_id_on_psi_grid'
            nc.variables['source_id_psi'].units = 'no units'
            nc.variables['source_id_psi'].standard_name = 'source_id_psi'
            nc.variables['source_id_psi'][:] = id_psi

            # Global attributes
            date = datetime.now()
            date = date.strftime('%d/%m/%Y, %H:%M:%S')
            nc.date_created = date

            if plastic:
                nc.createVariable('cplastic_psi', 'f8', ('lat_psi', 'lon_psi'), zlib=True)
                nc.variables['cplastic_psi'].long_name = 'plastic_flux_at_coast_psi_from_coastal_sources'
                nc.variables['cplastic_psi'].units = 'kg yr-1'
                nc.variables['cplastic_psi'].standard_name = 'coast_plastic'
                nc.variables['cplastic_psi'].total_flux = total_coastal_plastic
                nc.variables['cplastic_psi'][:] = cplastic

                nc.createVariable('rplastic_psi', 'f8', ('lat_psi', 'lon_psi'), zlib=True)
                nc.variables['rplastic_psi'].long_name = 'plastic_flux_at_coast_psi_from_riverine_sources'
                nc.variables['rplastic_psi'].units = 'kg yr-1'
                nc.variables['rplastic_psi'].standard_name = 'river_plastic'
                nc.variables['rplastic_psi'].total_flux = total_riverine_plastic
                nc.variables['rplastic_psi'][:] = rplastic

                nc.createVariable('iso_psi', 'i4', ('lat_psi', 'lon_psi'), zlib=True)
                nc.variables['iso_psi'].long_name = 'ISO_3166-1_numeric_code_of_coastal_psi_cells'
                nc.variables['iso_psi'].units = 'no units'
                nc.variables['iso_psi'].standard_name = 'iso_psi'
                nc.variables['iso_psi'][:] = iso_psi

                nc.createVariable('iso_psi_all', 'i4', ('lat_psi', 'lon_psi'), zlib=True)
                nc.variables['iso_psi_all'].long_name = 'ISO_3166-1_numeric_code_of_all_psi_cells'
                nc.variables['iso_psi_all'].units = 'no units'
                nc.variables['iso_psi_all'].standard_name = 'iso_psi_all'
                nc.variables['iso_psi_all'][:] = iso_psi_land

                nc.createVariable('sink_id_psi', 'i2', ('lat_psi', 'lon_psi'), zlib=True)
                nc.variables['sink_id_psi'].long_name = 'sink_cell_id_on_psi_grid'
                nc.variables['sink_id_psi'].units = 'no units'
                nc.variables['sink_id_psi'].standard_name = 'sink_id_psi'
                nc.variables['sink_id_psi'][:] = sink_id_psi

                nc.country_id_source = 'https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse'
                nc.coast_plastic_source = 'https://www.nature.com/articles/s41599-018-0212-7'
                nc.river_plastic_source = 'https://advances.sciencemag.org/content/7/18/eaaz5803/tab-figures-data'

                grid = {'coast_psi': np.array(nc.variables['coast_psi'][:]),
                            'lsm_psi': np.array(nc.variables['lsm_psi'][:]),
                            'lsm_rho': np.array(nc.variables['lsm_rho'][:]),
                            'lon_rho': np.array(nc.variables['lon_rho'][:]),
                            'lon_psi': np.array(nc.variables['lon_psi'][:]),
                            'lat_rho': np.array(nc.variables['lat_rho'][:]),
                            'lat_psi': np.array(nc.variables['lat_psi'][:]),
                            'cdist_rho': np.array(nc.variables['cdist_rho'][:]),
                            'cnormx_rho': np.array(nc.variables['cnormx_rho'][:]),
                            'cnormy_rho': np.array(nc.variables['cnormy_rho'][:]),
                            'source_id_psi': np.array(nc.variables['source_id_psi']),
                            'sink_id_psi': np.array(nc.variables['sink_id_psi']),
                            'iso_psi': np.array(nc.variables['iso_psi']),
                            'iso_psi_all': np.array(nc.variables['iso_psi_all']),
                            'cplastic_psi': np.array(nc.variables['cplastic_psi']),
                            'rplastic_psi': np.array(nc.variables['rplastic_psi'])}
            else:
                grid = {'coast_psi': np.array(nc.variables['coast_psi'][:]),
                            'lsm_psi': np.array(nc.variables['lsm_psi'][:]),
                            'lsm_rho': np.array(nc.variables['lsm_rho'][:]),
                            'lon_rho': np.array(nc.variables['lon_rho'][:]),
                            'lon_psi': np.array(nc.variables['lon_psi'][:]),
                            'lat_rho': np.array(nc.variables['lat_rho'][:]),
                            'lat_psi': np.array(nc.variables['lat_psi'][:]),
                            'cdist_rho': np.array(nc.variables['cdist_rho'][:]),
                            'cnormx_rho': np.array(nc.variables['cnormx_rho'][:]),
                            'cnormy_rho': np.array(nc.variables['cnormy_rho'][:]),
                            'source_id_psi': np.array(nc.variables['source_id_psi'])}

    return grid
