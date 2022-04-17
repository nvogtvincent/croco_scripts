#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read event data from forward marine debris simulations
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit
from netCDF4 import Dataset
from datetime import datetime
from glob import glob
from tqdm import tqdm

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = { # Runtime parameters
         'dt': 3600,    # Simulation timestep (s)

         # Analysis parameters
         'us_d': 1825,    # Sinking timescale (days)
         'ub_d': 20,      # Beaching timescale (days)

         # Time range
         'y0'  : 1993,
         'y1'  : 2012,

         # Grid
         'grid_res': 1/12,         # Grid resolution in degrees
         'lon_range': [20, 130],  # Longitude range for output
         'lat_range': [-40, 30],

         # Physics
         'mode': '0000',

         # Source/sink time
         'time': 'sink',

         # Sink sites
         'sites': np.array([1])
         }

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/MAR/' + param['mode'] + '/'}

# FILE HANDLES
fh = {'grid':    dirs['grid'] + 'griddata.nc',
      'clist':   dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'traj':    sorted(glob(dirs['traj'] + '*FwdMar' + param['mode'] + '.nc'))}

# Convert sinking rates and beaching rates to correct units
param['us'] = 1/(param['us_d']*3600*24)
param['ub'] = 1/(param['ub_d']*3600*24)

##############################################################################
# EXTRACT EVENT DATA                                                         #
##############################################################################

def translate_events(array):
    # Create output arrays
    sink_id = np.zeros_like(array, dtype=np.int16)
    time_at_sink = np.zeros_like(array, dtype=np.int32)
    prior_tb = np.zeros_like(array, dtype=np.int32)
    prior_ts = np.zeros_like(array, dtype=np.int32)

    # Generate useful numbers for translation
    p0 = 2**20
    p1 = 2**40
    p2 = 2**52

    for row in range(len(sink_id)):
        code = array[row]
        val3 = int(code%p0)
        val2 = int(((code-val3)%p1)/p0)
        val1 = int(((code-val2)%p2)/p1)
        val0 = int((code-val2)/p2)

        sink_id[row] = val0
        time_at_sink[row] = val1
        prior_tb[row] = val2
        prior_ts[row] = val3

    return [sink_id, time_at_sink, prior_tb, prior_ts]

@njit
def translate_events_para(array):
    # Create output arrays
    sink_id = np.zeros_like(array, dtype=np.int16)
    time_at_sink = np.zeros_like(array, dtype=np.int32)
    prior_tb = np.zeros_like(array, dtype=np.int32)
    prior_ts = np.zeros_like(array, dtype=np.int32)

    # Generate useful numbers for translation
    p0 = 2**20
    p1 = 2**40
    p2 = 2**52

    for row in range(len(sink_id)):
        code = array[row]
        val3 = int(code%p0)
        val2 = int(((code-val3)%p1)/p0)
        val1 = int(((code-val2)%p2)/p1)
        val0 = int((code-val2)/p2)

        sink_id[row] = val0
        time_at_sink[row] = val1
        prior_tb[row] = val2
        prior_ts[row] = val3

    return sink_id, time_at_sink, prior_tb, prior_ts


def convert_events(fh_list, dt, us, ub, n_events, **kwargs):
    '''
    Parameters
    ----------
    fh_list : List of trajectory files
    dt : Simulation timestep (s)
    us : Sinking rate (1/s)
    ub : Beaching rate (1/s)
    n_events : Number of events per particle to record

    **kwargs :
        particles_per_file: number of particles released per file

    Returns
    -------
    data : Pandas file containing sorted data
    stats : TYPE
        DESCRIPTION.

    '''
    # Keep track of some basic data
    total_particles = 0
    total_encounters = 0
    total_full_events = 0

    if 'particles_per_file' in kwargs:
        total_particles_all = len(fh_list)*kwargs['particles_per_file']

    data_list = []

    # Open all data
    for fhi, fh in tqdm(enumerate(fh_list), total=len(fh_list)):
        with Dataset(fh, mode='r') as nc:
            e_num = nc.variables['e_num'][:]
            n_traj = np.shape(e_num)[0] # Number of trajectories in file

            if n_traj:
                # Extract origin date
                y0 = int(fh.split('/')[-1].split('_')[0])
                m0 = int(fh.split('/')[-1].split('_')[1])
                t0 = datetime(year=y0, month=m0, day=1, hour=0)

                # Firstly load primary variables into memory
                raw_event_array = np.zeros((n_traj, n_events), dtype=np.int64)
                raw_lon0_array = np.zeros((n_traj, n_events), dtype=np.float32)
                raw_lat0_array = np.zeros((n_traj, n_events), dtype=np.float32)

                for i in range(n_events):
                    raw_event_array[:, i] = nc.variables['e' + str(i)][:, 0]

                raw_lon0_array[:] = nc.variables['lon0'][:]
                raw_lat0_array[:] = nc.variables['lat0'][:]

                # Update stats
                total_particles += n_traj
                total_encounters += np.count_nonzero(raw_event_array)
                total_full_events += np.count_nonzero(raw_event_array[:, -1])

                # Now flatten arrays
                raw_event_array = raw_event_array.flatten()
                mask = raw_event_array != 0
                raw_event_array = raw_event_array[mask]

                lon0_array = raw_lon0_array.flatten()
                lon0_array = lon0_array[mask]

                lat0_array = raw_lat0_array.flatten()
                lat0_array = lat0_array[mask]

                # Now convert events
                sink_array = np.zeros_like(raw_event_array, dtype=np.int16)
                time_at_sink_array = np.zeros_like(raw_event_array, dtype=np.int32)
                prior_tb_array = np.zeros_like(raw_event_array, dtype=np.int32)
                prior_ts_array = np.zeros_like(raw_event_array, dtype=np.int32)

                converted_arrays = translate_events_para(raw_event_array)

                sink_array[:] = converted_arrays[0]
                time_at_sink_array[:] = converted_arrays[1]*dt
                prior_tb_array[:] = converted_arrays[2]*dt
                prior_ts_array[:] = converted_arrays[3]*dt
                post_ts_array = time_at_sink_array + prior_ts_array
                post_tb_array = time_at_sink_array + prior_tb_array

                # Now calculate plastic loss
                post_mass = np.exp(-(us*post_ts_array)-(ub*post_tb_array))
                prior_mass = np.exp(-(us*prior_ts_array)-(ub*prior_tb_array))
                loss = (ub/(ub+us))*(prior_mass - post_mass)
                loss = loss.astype('float32')

                # Now form output array
                frame = pd.DataFrame(data=sink_array, columns=['sink_iso'])

                frame['plastic_flux'] = loss
                frame['days_at_sea'] = prior_ts_array
                frame['lat0'] = lat0_array
                frame['lon0'] = lon0_array
                frame['days_at_sea'] = pd.to_timedelta(frame['days_at_sea'], unit='S')
                frame['sink_date_'] = frame['days_at_sea'] + t0
                frame['sink_date'] = frame['sink_date_'].dt.year + (frame['sink_date_'].dt.month-0.5)*(1/12)
                frame['days_at_sea'] = frame['days_at_sea'].dt.days
                frame['source_date_'] = pd.to_datetime(t0)
                frame['source_date'] = frame['source_date_'].dt.year + (frame['source_date_'].dt.month-0.5)*(1/12)

                # Clean up
                frame.drop(labels=['sink_date_', 'source_date_'], axis=1, inplace=True)
                frame.reset_index(drop=True)

                # Remove unnecessary precision
                frame = frame.astype({'days_at_sea': 'int16'})

                # Append
                data_list.append(frame)

    data = pd.concat(data_list, axis=0)
    data.reset_index(drop=True)

    if (data['days_at_sea'].max() > 3660) or (data['days_at_sea'].max() < 3640):
        print('WARNING: ARE YOU SURE THAT THE TIMESTEP IS CORRECT?')
        print('max: ' + str(data['days_at_sea'].max()) + ' days')

    # Store stats
    stats = {'total_particles_reaching_seychelles': total_particles,
             'total_particles_released': total_particles_all,
             'total_encounters': total_encounters,
             'total_full_events': total_full_events}

    return data, stats

##############################################################################
# GRID PLASTIC FLUXES                                                        #
##############################################################################

# Firstly construct grid
with Dataset(fh['grid'], mode='r') as nc:
    lon = nc.variables['lon_psi'][:]
    lat = nc.variables['lat_psi'][:]
    lon_bnd = np.concatenate([nc.variables['lon_rho'][:], [180]])
    lat_bnd = nc.variables['lat_rho'][:]

    lon = lon[(lon >= param['lon_range'][0])*(lon <= param['lon_range'][1])]
    lat = lat[(lat >= param['lat_range'][0])*(lat <= param['lat_range'][1])]
    lon_bnd = lon_bnd[(lon_bnd >= param['lon_range'][0])*(lon_bnd <= param['lon_range'][1])]
    lat_bnd = lat_bnd[(lat_bnd >= param['lat_range'][0])*(lat_bnd <= param['lat_range'][1])]

sink_list = pd.read_csv(fh['sink_list']).sort_values('Sink code').iloc[:18, :]
nsink = np.shape(sink_list)[0]

# Generate time axis
if param['time'] == 'sink':
    sink_time_dec = np.arange(1993, 2021, 1/12) + 1/24
    sink_time = pd.date_range(start=datetime(year=1993, month=1, day=1),
                              end=datetime(year=2021, month=1, day=1),
                              freq='M')
    sink_ntime = len(sink_time_dec)
    sink_time_bnds = np.arange(1993, 2021 + (1/12), 1/12)

    # Flux matrix dimensions:
    # LAT | LON | TIME
    fmatrix = np.zeros((len(lat), len(lon), sink_ntime), dtype=np.float32)
    fmatrix = xr.DataArray(fmatrix, coords=[lat, lon, sink_time],
                           dims=['latitude', 'longitude', 'sink_time'])

    # Time matrix dimensions:
    # SOURCE | SINK | DRIFT_TIME
    ftmatrix = np.zeros((len(lat), len(lon), sink_ntime), dtype=np.float32)
    ftmatrix = xr.DataArray(ftmatrix, coords=[lat, lon, sink_time],
                           dims=['latitude', 'longitude', 'sink_time'])

else:
    raise NotImplementedError('Not yet implemented')

# source_time_dec = np.arange(param['y0'], param['y1']+1, 1/12) + 1/24
# source_time = pd.date_range(start=datetime(year=param['y0'], month=1, day=1),
#                             end=datetime(year=param['y1']+1, month=1, day=1),
#                             freq='M')

# source_ntime = len(source_time_dec)

# # Create bounds for source, sink, release_time, and arrival_time
# source_bnds = source_list['ISO code'].values.astype(np.float32) - 0.5
# source_bnds = np.append(source_bnds, source_bnds[-1]+1)



# drift_time_bnds = np.arange(0, 2925, 5) # Days
# drift_time = 0.5*(drift_time_bnds[:-1] + drift_time_bnds[1:])
# drift_time_bnds[-1] = 3650
# n_drift_time = len(drift_time_bnds)-1

# source_time_bnds = np.arange(param['y0'], param['y1'] + (13/12), 1/12)
#

for year in np.arange(param['y0'], param['y1']+1):
    for month in np.arange(12):
        for release in np.arange(4):
            print('Year ' + str(year) + '/' + str(param['y1']))
            print('Month ' + str(month+1) + '/12')
            print('Release' + str(release+1) + '/4')
            yearmonth_fh = sorted(glob(dirs['traj'] + str(year) + '_' + str(month+1) + '_' + str(release) + '_*.nc'))

            data, stats = convert_events(yearmonth_fh, param['dt'], param['us'], param['ub'], 25,
                                         particles_per_file=325233)

            fmatrix += np.histogramdd(np.array([data[data['sink_iso'].isin(param['sites'])]['lat0'],
                                                data[data['sink_iso'].isin(param['sites'])]['lon0'],
                                                data[data['sink_iso'].isin(param['sites'])]['sink_date']]).T,
                                      bins=(lat_bnd, lon_bnd, sink_time_bnds),
                                      weights=(data[data['sink_iso'].isin(param['sites'])]['plastic_flux']))[0]

            ftmatrix += np.histogramdd(np.array([data[data['sink_iso'].isin(param['sites'])]['lat0'],
                                                 data[data['sink_iso'].isin(param['sites'])]['lon0'],
                                                 data[data['sink_iso'].isin(param['sites'])]['sink_date']]).T,
                                      bins=(lat_bnd, lon_bnd, sink_time_bnds),
                                       weights=(data[data['sink_iso'].isin(param['sites'])]['plastic_flux']*
                                                data[data['sink_iso'].isin(param['sites'])]['days_at_sea']))[0]

            print()


save_fh_f = dirs['script'] + '/marine_flux_' + param['mode'] + '_' + np.array2string(param['sites'], separator='-') + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc'
save_fh_ft = dirs['script'] + '/marine_drift_time_' + param['mode'] + '_' + np.array2string(param['sites'], separator='-') + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '.nc'

for matrix in [fmatrix, ftmatrix]:
    matrix.attrs['us'] = param['us_d']
    matrix.attrs['ub'] = param['ub_d']

fmatrix.to_netcdf(save_fh_f)
ftmatrix.to_netcdf(save_fh_ft)
