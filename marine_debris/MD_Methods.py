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
from netCDF4 import Dataset
from calendar import monthrange
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from datetime import datetime
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

    Years  = [param['Ymin'], param['Ymax']]
    Months = [param['Mmin'], param['Mmax']]
    RPM    = param['RPM']
    t0     = int(param['t0'])

    Years = np.arange(Years[0], Years[1]+1)
    nyears = Years.shape[0]
    ntime = ((nyears*12) + ((Months[1] - Months[0]) - 11))*RPM

    time = np.zeros((ntime,), dtype=np.int32)

    pos, cumtime = 0, 0

    for yi in range(nyears):
        # Calculate number of months in this year
        m0 = Months[0] if yi == 0 else 1
        m1 = Months[1] if yi == nyears-1 else 12

        for mi in range(m0, m1+1):
            # Calculate release days in month
            days = monthrange(Years[yi], mi)[-1]

            if start_mode:
                daysi = np.linspace(0, days, num=RPM+1)[:-1]
            else:
                daysi = np.linspace(0, days, num=RPM+1)[1:]

            time[pos*RPM:(pos+1)*RPM] = daysi*86400 + cumtime

            pos += RPM
            cumtime += days*86400

    time -= t0

    return time

def release_loc(param, fh):
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

        id_psi    = np.array(nc.variables['id_psi'][:])
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

    ntimes = times.shape[0]
    npart  = data['lon'].shape[0]

    data['lon']  = np.tile(data['lon'], ntimes)
    data['lat']  = np.tile(data['lat'], ntimes)
    data['iso']  = np.tile(data['iso'], ntimes)
    data['id']   = np.tile(data['id'], ntimes)
    data['time'] = np.repeat(times, npart)

    # Also now calculation partitioning of particles
    if param['total_partitions'] > 1:
        pn_per_part = int(np.ceil(npart*ntimes/param['total_partitions']))
        i0 = pn_per_part*param['partition']
        i1 = pn_per_part*(param['partition']+1)

        data['lon'] = data['lon'][i0:i1]
        data['lat'] = data['lat'][i0:i1]
        data['iso'] = data['iso'][i0:i1]
        data['id'] = data['id'][i0:i1]
        data['time'] = data['time'][i0:i1]

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
                        'id_psi': np.array(nc.variables['id_psi']),
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
                            'id_psi': np.array(nc.variables['id_psi'])}

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

            #    Modify cells by W_mod = W*exp(-efs*cdist)*cr
            plastic_data *= cr*np.exp(-efs*cdist)

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

            nc.createVariable('id_psi', 'i4', ('lat_psi', 'lon_psi'), zlib=True)
            nc.variables['id_psi'].long_name = 'cell_id_on_psi_grid'
            nc.variables['id_psi'].units = 'no units'
            nc.variables['id_psi'].standard_name = 'id_psi'
            nc.variables['id_psi'][:] = id_psi

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
                            'id_psi': np.array(nc.variables['id_psi']),
                            'iso_psi': np.array(nc.variables['iso_psi']),
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
                            'id_psi': np.array(nc.variables['id_psi'])}

    return grid


# # Directories
# script_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
# grid_dir   = script_dir + 'CMEMS/'
# data_dir   = script_dir + 'PLASTIC_DATA/'

# # Grid from CMEMS
# grid_fh    = grid_dir + 'globmask.nc'

# # Lebreton & Andrady plastic waste generation data
# # https://www.nature.com/articles/s41599-018-0212-7
# plastic_fh = data_dir + 'LebretonAndrady2019_MismanagedPlasticWaste.tif'

# # Country ID grid(UN WPP-Adjusted Population Density v4.11)
# # https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse
# id_fh      = data_dir + 'gpw_v4_national_identifier_grid_rev11_30_sec.tif'

# # Meijer et al river plastic data
# # https://advances.sciencemag.org/content/7/18/eaaz5803/tab-figures-data
# river_fh   = data_dir + 'Meijer2021_midpoint_emissions.zip'

# # Output netcdf fh
# out_fh     = grid_dir + 'plastic_flux.nc'

##############################################################################
# Coastal plastics ###########################################################
##############################################################################

# Methodology
# 1. Upscale plastic/id data to 1/12 grid
# 2. Calculate nearest coastal cell on CMEMS grid
# 3. Calculate the distance of each cell to coastal cell with Haversine
# 4. Calculate the waste flux at each coastal cell (sum of weighted land cells)

# Load CMEMS grid
# with Dataset(grid_fh, mode='r') as nc:
#     # Cell boundaries
#     lon_bnd = nc.variables['lon_rho'][:]
#     lon_bnd = np.append(lon_bnd, 180)
#     lat_bnd = nc.variables['lat_rho'][:]

#     # Cell centres
#     lon     = nc.variables['lon_psi'][:]
#     lat     = nc.variables['lat_psi'][:]
#     lon, lat = np.meshgrid(lon, lat)

#     # Coast and lsm
#     coast   = nc.variables['coast_psi'][:]
#     lsm     = nc.variables['lsm_psi'][:]

# # Load plastic data
# plastic_data_obj = gdal.Open(plastic_fh)
# plastic_data = plastic_data_obj.ReadAsArray()[:-1200, :]

# # Load country codes
# country_id_obj = gdal.Open(id_fh)
# country_id = country_id_obj.ReadAsArray()[:-1200, :]

# # 1. Upscale data to 1/12 grid
# plastic_data = block_reduce(plastic_data, (10,10), np.sum)
# plastic_data = np.float64(np.flipud(plastic_data))
# country_id = block_reduce(country_id, (10,10), np.max)
# country_id = np.flipud(country_id)

# # 2. Calculate nearest coastal cells
# cnearest = distance_transform_edt(1-coast,
#                                   return_distances=False,
#                                   return_indices=True)


# # 3. Calculate the distance of each waste flux cell to the nearest coastal
# #    cell
# cdist = cdist_calc(cnearest, lon, lat, efs)

# #    Modify cells by W_mod = W*exp(-efs*cdist)*cr
# plastic_data *= cr*np.exp(-efs*cdist)

# # 4. Calculate the waste flux at each coastal cell
# cplastic = cplastic_calc(plastic_data, cnearest)

# ##############################################################################
# # Riverine plastics ##########################################################
# ##############################################################################

# river_file = gpd.read_file(river_fh)
# river_data = np.array(river_file['dots_exten'])*1000.  # tons -> kg
# river_lon  = np.array(river_file['geometry'].x)
# river_lat  = np.array(river_file['geometry'].y)

# # Bin onto the 1/12 degree grid
# rplastic   = np.histogram2d(river_lon, river_lat,
#                             bins=[lon_bnd, lat_bnd],
#                             weights=river_data,
#                             normed=False)[0].T

# # Shift to coastal cells
# rplastic   = cplastic_calc(rplastic, cnearest)


# ##############################################################################
# # Label coastal cells ########################################################
# ##############################################################################

# # Label coastal cells (coast) with their ISO country identifier
# # Calculate the nearest country_id to each grid cell without a country_id
# # A coastal cell with a country id takes that country id, otherwise it takes
# # the nearest country id

# country_id_lsm = np.copy(country_id)
# country_id_lsm[country_id_lsm >= 0] = 0
# country_id_lsm[country_id_lsm <  0] = 1

# country_id_nearest = distance_transform_edt(country_id_lsm,
#                                             return_distances=False,
#                                             return_indices=True)

# coast_id = id_coast_cells(coast, country_id, country_id_nearest)

# ##############################################################################
# # Save to netcdf #############################################################
# ##############################################################################

# total_coastal_plastic = '{:3.2f}'.format(np.sum(cplastic)/1e9) + ' Mt yr-1'
# total_riverine_plastic = '{:3.2f}'.format(np.sum(rplastic)/1e9) + ' Mt yr-1'

# with Dataset(out_fh, mode='w') as nc:
#     # Create dimensions
#     nc.createDimension('lon', lon.shape[1])
#     nc.createDimension('lat', lat.shape[0])

#     # Create variables
#     nc.createVariable('lon', 'f4', ('lon'), zlib=True)
#     nc.createVariable('lat', 'f4', ('lat'), zlib=True)
#     nc.createVariable('coast_plastic', 'f8', ('lat', 'lon'), zlib=True)
#     nc.createVariable('river_plastic', 'f8', ('lat', 'lon'), zlib=True)
#     nc.createVariable('coast_id', 'i4', ('lat', 'lon'), zlib=True)
#     nc.createVariable('coast', 'i4', ('lat', 'lon'), zlib=True)
#     nc.createVariable('lsm', 'i4', ('lat', 'lon'), zlib=True)

#     # Write variables
#     nc.variables['lon'].long_name = 'longitude'
#     nc.variables['lon'].units = 'degrees_east'
#     nc.variables['lon'].standard_name = 'longitude'
#     nc.variables['lon'][:] = lon[0, :]

#     nc.variables['lat'].long_name = 'latitude'
#     nc.variables['lat'].units = 'degrees_north'
#     nc.variables['lat'].standard_name = 'latitude'
#     nc.variables['lat'][:] = lat[:, 0]

#     nc.variables['coast_plastic'].long_name = 'plastic_flux_at_coast_from_coastal_sources'
#     nc.variables['coast_plastic'].units = 'kg yr-1'
#     nc.variables['coast_plastic'].standard_name = 'coast_plastic'
#     nc.variables['coast_plastic'].total_flux = total_coastal_plastic
#     nc.variables['coast_plastic'][:] = cplastic

#     nc.variables['river_plastic'].long_name = 'plastic_flux_at_coast_from_riverine_sources'
#     nc.variables['river_plastic'].units = 'kg yr-1'
#     nc.variables['river_plastic'].standard_name = 'river_plastic'
#     nc.variables['river_plastic'].total_flux = total_riverine_plastic
#     nc.variables['river_plastic'][:] = rplastic

#     nc.variables['coast_id'].long_name = 'ISO_3166-1_numeric_code_of_coastal_cells'
#     nc.variables['coast_id'].units = 'no units'
#     nc.variables['coast_id'].standard_name = 'coast_id'
#     nc.variables['coast_id'][:] = coast_id

#     nc.variables['coast'].long_name = 'coast_mask'
#     nc.variables['coast'].units = '1: coast, 0: not coast'
#     nc.variables['coast'].standard_name = 'coast_mask'
#     nc.variables['coast'][:] = coast

#     nc.variables['lsm'].long_name = 'land_sea_mask'
#     nc.variables['lsm'].units = 'no units'
#     nc.variables['lsm'].standard_name = '1: land, 0: ocean'
#     nc.variables['lsm'][:] = lsm

#     # Global attributes
#     date = datetime.now()
#     date = date.strftime('%d/%m/%Y, %H:%M:%S')
#     nc.date_created = date

#     nc.country_id_source = 'https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/sets/browse'
#     nc.coast_plastic_source = 'https://www.nature.com/articles/s41599-018-0212-7'
#     nc.river_plastic_source = 'https://advances.sciencemag.org/content/7/18/eaaz5803/tab-figures-data'

