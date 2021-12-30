#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read event data from forward marine debris simulations
@author: Noam Vogt-Vincent
"""

import os
import MD_Methods as mdm
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
import pandas as pd
import cmasher as cmr
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar, Variable,
                     DiffusionUniformKh)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob
from sys import argv
from numba import njit
from datetime import timedelta, datetime
import time

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = { # Runtime parameters
         'dt': 2700,    # Simulation timestep (s)

         # Analysis parameters
         'ls': 3650,    # Sinking timescale (days)
         'lb': 40,}     # Beaching timescale (days)

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
      'traj':    sorted(glob(dirs['traj'] + '*.nc'))}

# Convert sinking rates and beaching rates to correct units
param['ls'] = 1/(param['ls']*3600*24)
param['lb'] = 1/(param['lb']*3600*24)

##############################################################################
# EXTRACT EVENT DATA                                                         #
##############################################################################

# def extract(code_list, dt):
#     sink_id = np.zeros_like(code_list, dtype=np.int64)
#     time_at_sink = np.zeros_like(code_list, dtype=np.int64)
#     prior_tb = np.zeros_like(code_list, dtype=np.int64)
#     prior_ts = np.zeros_like(code_list, dtype=np.int64)

#     p0 = 2**20
#     p1 = 2**40
#     p2 = 2**52

#     for row in range(np.shape(code_list)[0]):
#         val = code_list[row]
#         val3 = int(val%p0)
#         val2 = int(((val-val3)%p1)/p0)
#         val1 = int(((val-val2)%p2)/p1)
#         val0 = int((val-val2)/p2)

#         sink_id[row] = val0
#         time_at_sink[row] = val1*dt
#         prior_tb[row] = val2*dt
#         prior_ts[row] = val3*dt

#     return [sink_id, time_at_sink, prior_tb, prior_ts]

# @njit
def translate_events(array):
    # Create output arrays
    sink_id = np.zeros_like(array, dtype=np.int32)
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





def convert_events(fh_list, dt, ls, lb, n_events):
    # Keep track of some basic data
    total_particles = 0
    total_encounters = 0
    total_full_events = 0

    output_array_made = False

    # Open all data
    for fhi, fh in enumerate(fh_list):
        with Dataset(fh, mode='r') as nc:
            e_num = nc.variables['e_num'][:]
            n_traj = np.shape(e_num)[0] # Number of trajectories in file

            if n_traj:
                # Extract origin date
                y0 = int(fh.split('/')[-1].split('_')[0])
                # m0 = int(fh.split('/')[-1].split('_')[1]) # PUT BACK IN!!!
                m0 = 1                                 # TEMPORARY FOR TESTING FILES ONLY!!!
                t0 = datetime(year=y0, month=m0, day=1, hour=0)

                # Firstly load primary variables into memory
                raw_event_array = np.zeros((n_traj, n_events), dtype=np.int64)
                raw_source_array = np.zeros((n_traj, n_events), dtype=np.int64)
                raw_rp0_array = np.zeros((n_traj, n_events), dtype=np.float32)
                raw_cp0_array = np.zeros((n_traj, n_events), dtype=np.float32)

                for i in range(n_events):
                    raw_event_array[:, i] = nc.variables['e' + str(i)][:, 0]

                raw_source_array[:] = nc.variables['source_id'][:]
                raw_rp0_array[:] = nc.variables['rp0'][:]
                raw_cp0_array[:] = nc.variables['cp0'][:]

                # Update stats
                total_particles += n_traj
                total_encounters += np.count_nonzero(raw_event_array)
                total_full_events += np.count_nonzero(raw_event_array[:, -1])

                # Now flatten arrays
                raw_event_array = raw_event_array.flatten()
                mask = raw_event_array != 0
                raw_event_array = raw_event_array[mask]

                raw_source_array = raw_source_array.flatten()
                raw_source_array = raw_source_array[mask]

                raw_rp0_array = raw_rp0_array.flatten()
                raw_rp0_array = raw_rp0_array[mask]

                raw_cp0_array = raw_cp0_array.flatten()
                raw_cp0_array = raw_cp0_array[mask]

                # Now convert events
                sink_array = np.zeros_like(raw_event_array, dtype=np.int64)
                time_at_sink_array = np.zeros_like(raw_event_array, dtype=np.int64)
                prior_tb_array = np.zeros_like(raw_event_array, dtype=np.int64)
                prior_ts_array = np.zeros_like(raw_event_array, dtype=np.int64)

                converted_arrays = translate_events(raw_event_array)

                sink_array[:] = converted_arrays[0]
                time_at_sink_array[:] = converted_arrays[1]*dt
                prior_tb_array[:] = converted_arrays[2]*dt
                prior_ts_array[:] = converted_arrays[3]*dt

                # Now calculate plastic loss
                prior_mass = np.exp((-ls*prior_ts_array)+(-lb*prior_tb_array))
                post_mass = prior_mass*np.exp(-(ls+lb)*time_at_sink_array)
                loss = prior_mass - post_mass

                # Now form output array
                frame = pd.DataFrame(data=raw_source_array, columns=['source_id'])
                frame['sink_id'] = sink_array
                frame['plastic_flux'] = loss*(raw_rp0_array + raw_cp0_array)

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

                # Append
                if not output_array_made:
                    data = frame.copy()
                    output_array_made = True
                else:
                    data = data.append(frame, ignore_index=True)

    # Store stats
    stats = {'total_particles': total_particles,
             'total_encounters': total_encounters,
             'total_full_events': total_full_events}

    return data, stats



t0 = time.time()
data, stats = convert_events(fh['traj'], param['dt'], param['ls'], param['lb'], 10)
print(time.time() - t0)