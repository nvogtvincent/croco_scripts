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
from sys import argv

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = { # Runtime parameters
         'dt': 3600,    # Simulation timestep (s)

         # Analysis parameters
         'us_d': int(argv[1]),      # Sinking timescale (days)
         'ub_d': int(argv[2]),      # Beaching timescale (days)

         # Time range
         'y0'  : 1993,
         'y1'  : 2014,

         'r_frac': 1.0,     # Fraction of riverine plastics
         'c_frac': 0.25,     # Fraction of coastal plastics

         # Physics
         'mode': argv[3]
         }

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/../GRID_DATA/',
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/../PLASTIC_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/../TRAJ/LAND/' + param['mode'] + '/'}

# FILE HANDLES
fh = {'source_list': dirs['plastic'] + 'country_list.in',
      'sink_list': dirs['plastic'] + 'sink_list.in',
      'grid':    dirs['grid'] + 'griddata_land.nc',
      'clist':   dirs['plastic'] + 'country_list.in',
      'traj':    sorted(glob(dirs['traj'] + '*FwdLand' + param['mode'] + '.nc'))}

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
                raw_rp0_array = np.zeros((n_traj, n_events), dtype=np.float32)
                raw_cp0_array = np.zeros((n_traj, n_events), dtype=np.float32)
                raw_source_iso_array = np.zeros((n_traj, n_events), dtype=np.int16)

                for i in range(n_events):
                    raw_event_array[:, i] = nc.variables['e' + str(i)][:, 0]

                # Divide plastic fluxes by 12*4 to convert kg/yr -> kg/release (4*12 releases per year)
                raw_rp0_array[:] = nc.variables['rp0'][:]/48
                raw_cp0_array[:] = nc.variables['cp0'][:]/48
                raw_source_iso_array[:] = nc.variables['origin_iso'][:]

                # Update stats
                total_particles += n_traj
                total_encounters += np.count_nonzero(raw_event_array)
                total_full_events += np.count_nonzero(raw_event_array[:, -1])

                # Now flatten arrays
                raw_event_array = raw_event_array.flatten()
                mask = raw_event_array != 0
                raw_event_array = raw_event_array[mask]

                raw_source_iso_array = raw_source_iso_array.flatten()
                raw_source_iso_array = raw_source_iso_array[mask]

                raw_rp0_array = raw_rp0_array.flatten()
                raw_rp0_array = raw_rp0_array[mask]

                raw_cp0_array = raw_cp0_array.flatten()
                raw_cp0_array = raw_cp0_array[mask]

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
                post_tb_array = time_at_sink_array + prior_tb_array
                post_ts_array = time_at_sink_array + prior_ts_array

                # Now calculate plastic loss
                post_mass = np.exp(-(us*post_ts_array)-(ub*post_tb_array))
                prior_mass = np.exp(-(us*prior_ts_array)-(ub*prior_tb_array))
                loss = (ub/(ub+us))*(prior_mass - post_mass)
                loss = loss.astype('float32')

                # Now form output array
                frame = pd.DataFrame(data=sink_array, columns=['sink_iso'])

                frame['cplastic_flux'] = loss*(raw_cp0_array)
                frame['rplastic_flux'] = loss*(raw_rp0_array)
                frame['source_iso'] = raw_source_iso_array
                frame['days_at_sea'] = prior_ts_array
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
                frame = frame.astype({'days_at_sea': 'int16',
                                      'source_iso': 'int16'})

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
source_list = pd.read_csv(fh['source_list'])
source_list = source_list.append(pd.DataFrame([[999, 'Other']], columns=source_list.columns))
source_list = source_list.sort_values('ISO code')
sink_list = pd.read_csv(fh['sink_list']).sort_values('Sink code')
nsource = np.shape(source_list)[0]
nsink = np.shape(sink_list)[0]
source_time_dec = np.arange(param['y0'], param['y1']+1, 1/12) + 1/24
source_time = pd.date_range(start=datetime(year=param['y0'], month=1, day=1),
                            end=datetime(year=param['y1']+1, month=1, day=1),
                            freq='M')
sink_time_dec = np.arange(1993, 2020, 1/12) + 1/24
sink_time = pd.date_range(start=datetime(year=1993, month=1, day=1),
                          end=datetime(year=2020, month=1, day=1),
                          freq='M')
source_ntime = len(source_time_dec)
sink_ntime = len(sink_time_dec)

# Create bounds for source, sink, release_time, and arrival_time
source_bnds = source_list['ISO code'].values.astype(np.float32) - 0.5
source_bnds = np.append(source_bnds, source_bnds[-1]+1)

sink_bnds = sink_list['Sink code'].values.astype(np.float32) - 0.5
sink_bnds = np.append(sink_bnds, sink_bnds[-1]+1)

drift_time_bnds = np.arange(0, 2925, 5) # Days
drift_time = 0.5*(drift_time_bnds[:-1] + drift_time_bnds[1:])
drift_time_bnds[-1] = 3650
n_drift_time = len(drift_time_bnds)-1

source_time_bnds = np.arange(param['y0'], param['y1'] + (13/12), 1/12)
sink_time_bnds = np.arange(1993, 2020 + (1/12), 1/12)

# Flux matrix dimensions:
# SOURCE | SINK | RELEASE_TIME | ARRIVAL_TIME
fmatrix = np.zeros((nsource, nsink, source_ntime, sink_ntime), dtype=np.float32)
fmatrix = xr.DataArray(fmatrix, coords=[source_list['Country Name'],
                                        sink_list['Site name'],
                                        source_time,
                                        sink_time],
                       dims=['source', 'sink', 'source_time', 'sink_time'])

# Time matrix dimensions:
# SOURCE | SINK | DRIFT_TIME
tmatrix = np.zeros((nsource, nsink, n_drift_time))
tmatrix = xr.DataArray(tmatrix, coords=[source_list['Country Name'],
                                        sink_list['Site name'],
                                        drift_time],
                       dims=['source', 'sink', 'drift_time'])

total_stats = np.zeros((4,), dtype=np.int64)

for year in np.arange(param['y0'], param['y1']+1):
    for month in np.arange(12):
        for release in np.arange(4):
            print('Year ' + str(year) + '/' + str(param['y1']))
            print('Month ' + str(month+1) + '/12')
            print('Release' + str(release+1) + '/4')
            yearmonth_fh = sorted(glob(dirs['traj'] + str(year) + '_' + str(month+1) + '_' + str(release) + '_*.nc'))

            data, stats = convert_events(yearmonth_fh, param['dt'], param['us'], param['ub'], 35,
                                         particles_per_file=325233)

            fmatrix += np.histogramdd(np.array([data['source_iso'], data['sink_iso'],
                                                data['source_date'], data['sink_date']]).T,
                                      bins=(source_bnds, sink_bnds, source_time_bnds, sink_time_bnds),
                                      weights=((param['c_frac']*data['cplastic_flux'])+
                                               (param['r_frac']*data['rplastic_flux'])))[0]

            tmatrix += np.histogramdd(np.array([data['source_iso'], data['sink_iso'],
                                                data['days_at_sea']]).T,
                                      bins=(source_bnds, sink_bnds, drift_time_bnds),
                                      weights=((param['c_frac']*data['cplastic_flux'])+
                                               (param['r_frac']*data['rplastic_flux'])))[0]

            total_stats += np.array([stats['total_particles_reaching_seychelles'],
                                     stats['total_particles_released'],
                                     stats['total_encounters'],
                                     stats['total_full_events']])

print('Total particles reaching seychelles: ' + str(total_stats[0]))
print('Total particles released: ' + str(total_stats[1]))
print('Total events: ' + str(total_stats[2]))
print('Total full events: ' + str(total_stats[3]))

save_fh_f = dirs['script'] + '/land_flux_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '_c' + str(param['c_frac']) + '.nc'
save_fh_t = dirs['script'] + '/land_drift_time_' + param['mode'] + '_s' + str(param['us_d']) + '_b' + str(param['ub_d']) + '_c' + str(param['c_frac']) +'.nc'

for matrix in [fmatrix, tmatrix]:
    matrix.attrs['us'] = param['us_d']
    matrix.attrs['ub'] = param['ub_d']
    matrix.attrs['r_frac'] = param['r_frac']
    matrix.attrs['c_frac'] = param['c_frac']

fmatrix.to_netcdf(save_fh_f)
tmatrix.to_netcdf(save_fh_t)
