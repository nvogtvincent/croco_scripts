#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read event data from forward marine debris simulations
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
from glob import glob
import time

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = { # Runtime parameters
         'dt': 3600,    # Simulation timestep (s)

         # Analysis parameters
         'ls_d': 3650,    # Sinking timescale (days)
         'lb_d': 40,      # Beaching timescale (days)

         # Mode
         'mode': 'land'   # Options: land/marine
         }

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/GRID_DATA/',
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/PLASTIC_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/'}

# FILE HANDLES
fh = {'grid':    dirs['grid'] + 'griddata.nc',
      'clist':   dirs['plastic'] + 'country_list.in',
      'fig':     dirs['fig'] + 'fwd_results.png',
      'traj':    sorted(glob(dirs['traj'] + '*.nc')),
      'output':  dirs['traj']}

# Convert sinking rates and beaching rates to correct units
param['ls'] = 1/(param['ls_d']*3600*24)
param['lb'] = 1/(param['lb_d']*3600*24)

if param['mode'] == 'land':
    param['mode'] = True
else:
    param['mode'] = False

# param['ls'] = 3.17e-9 # sinking
# param['lb'] = 5.79e-7 # beaching
##############################################################################
# EXTRACT EVENT DATA                                                         #
##############################################################################


def translate_events(array):
    # Create output arrays
    sink_id = np.zeros_like(array, dtype=np.int8)
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


def convert_events(fh_list, dt, ls, lb, n_events, **kwargs):
    '''
    Parameters
    ----------
    fh_list : List of trajectory files
    dt : Simulation timestep (s)
    ls : Sinking rate (1/s)
    lb : Beaching rate (1/s)
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
    for fhi, fh in enumerate(fh_list):
        print(str(100*fhi/len(fh_list))+'%')
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
                raw_source_cell_array = np.zeros((n_traj, n_events), dtype=np.int32)

                if param['mode']:
                    raw_rp0_array = np.zeros((n_traj, n_events), dtype=np.float32)
                    raw_cp0_array = np.zeros((n_traj, n_events), dtype=np.float32)
                    raw_source_id_array = np.zeros((n_traj, n_events), dtype=np.int16)
                else:
                    raw_gfw_num_array = np.zeros((n_traj, n_events), dtype=np.int32)

                for i in range(n_events):
                    raw_event_array[:, i] = nc.variables['e' + str(i)][:, 0]

                raw_source_cell_array[:] = nc.variables['source_cell'][:]

                # Divide plastic fluxes by 12 to convert kg/yr -> kg/mo
                if param['mode']:
                    raw_source_id_array[:] = nc.variables['source_id'][:]
                    raw_rp0_array[:] = nc.variables['rp0'][:]/12
                    raw_cp0_array[:] = nc.variables['cp0'][:]/12
                else:
                    raw_gfw_num_array[:] = nc.variables['gfw_num'][:]

                # Update stats
                total_particles += n_traj
                total_encounters += np.count_nonzero(raw_event_array)
                total_full_events += np.count_nonzero(raw_event_array[:, -1])

                # Now flatten arrays
                raw_event_array = raw_event_array.flatten()
                mask = raw_event_array != 0
                raw_event_array = raw_event_array[mask]

                raw_source_cell_array = raw_source_cell_array.flatten()
                raw_source_cell_array = raw_source_cell_array[mask]

                if param['mode']:
                    raw_source_id_array = raw_source_id_array.flatten()
                    raw_source_id_array = raw_source_id_array[mask]

                    raw_rp0_array = raw_rp0_array.flatten()
                    raw_rp0_array = raw_rp0_array[mask]

                    raw_cp0_array = raw_cp0_array.flatten()
                    raw_cp0_array = raw_cp0_array[mask]
                else:
                    raw_gfw_num_array = raw_gfw_num_array.flatten()
                    raw_gfw_num_array = raw_gfw_num_array[mask]


                # Now convert events
                sink_array = np.zeros_like(raw_event_array, dtype=np.int8)
                time_at_sink_array = np.zeros_like(raw_event_array, dtype=np.int32)
                prior_tb_array = np.zeros_like(raw_event_array, dtype=np.int32)
                prior_ts_array = np.zeros_like(raw_event_array, dtype=np.int32)

                converted_arrays = translate_events(raw_event_array)

                sink_array[:] = converted_arrays[0]
                time_at_sink_array[:] = converted_arrays[1]*dt
                prior_tb_array[:] = converted_arrays[2]*dt
                prior_ts_array[:] = converted_arrays[3]*dt

                # Now calculate plastic loss
                prior_mass = np.exp((-ls*prior_ts_array)+(-lb*prior_tb_array))
                post_mass = prior_mass*np.exp(-(ls+lb)*time_at_sink_array)
                loss = prior_mass - post_mass
                loss = loss.astype('float32')

                # Now form output array
                frame = pd.DataFrame(data=raw_source_cell_array, columns=['source_cell'])

                if param['mode']:
                    frame['plastic_flux'] = loss*(raw_rp0_array + raw_cp0_array)
                    frame['source_id'] = raw_source_id_array
                else:
                    frame['plastic_flux'] = loss
                    frame['gfw_num'] = raw_gfw_num_array

                frame['sink_id'] = sink_array
                frame['days_at_sea'] = prior_ts_array
                frame['days_at_sea'] = pd.to_timedelta(frame['days_at_sea'], unit='S')
                frame['sink_date'] = frame['days_at_sea'] + t0
                frame['sink_year'] = frame['sink_date'].dt.year
                frame['sink_month'] = frame['sink_date'].dt.month
                frame['days_at_sea'] = frame['days_at_sea'].dt.days
                frame['source_date'] = pd.to_datetime(t0)
                frame['source_year'] = frame['source_date'].dt.year
                frame['source_month'] = frame['source_date'].dt.month

                # Clean up
                frame.drop(labels=['sink_date', 'source_date'], axis=1, inplace=True)
                frame.reset_index(drop=True)

                # Remove unnecessary precision
                frame = frame.astype({'days_at_sea': 'int16',
                                      'sink_year': 'int16',
                                      'sink_month': 'int8',
                                      'source_year': 'int16',
                                      'source_month': 'int8'})

                # Append
                data_list.append(frame)

    data = pd.concat(data_list, axis=0)
    data.reset_index(drop=True)

    if (data['days_at_sea'].max() > 3660) or (data['days_at_sea'].max() < 3600):
        print('WARNING: ARE YOU SURE THAT THE TIMESTEP IS CORRECT?')
        print('max: ' + str(data['days_at_sea'].max()) + ' days')

    # Store stats
    stats = {'total_particles_reaching_seychelles': total_particles,
             'total_particles_released': total_particles_all,
             'total_encounters': total_encounters,
             'total_full_events': total_full_events}

    if 'save_data' in kwargs:
        if 'save_fh' in kwargs:
            save_fh = kwargs['save_fh']
        else:
            save_fh = 'data.nc'

        data.to_pickle(save_fh)

    return data, stats



t0 = time.time()
for year in np.arange(1993, 2010):
    year_fh = sorted(glob(dirs['traj'] + str(year) + '*.nc'))
    save_fh = dirs['traj'] + 'data_s' + str(param['ls_d']) + '_b' + str(param['lb_d']) + '_' + str(year) + '.pkl'
    data, stats = convert_events(year_fh, param['dt'], param['ls'], param['lb'], 20,
                                 particles_per_file=646456, save_data=True,
                                 save_fh=save_fh)
    print(stats)
print('Run time: ' + str(time.time() - t0) + 's')