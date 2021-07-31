#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles arriving at a country backward in time using the
CMEMS GLORYS12V1 reanalysis (current and Stokes drift)
@author: Noam Vogt-Vincent
"""

import marinedebrismethods as mdm
import numpy as np
import os
import matplotlib.pyplot as plt
import cmocean.cm as cm
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar, Variable)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARAMETERS
param = {# Release timing
         'Ymin'              : 2019,          # First release year
         'Ymax'              : 2019,          # Last release year
         'Mmin'              : 2   ,          # First release month
         'Mmax'              : 12  ,          # Last release month
         'RPM'               : 1   ,          # Releases per month
         'mode'              :'START',        # Release at END or START

         # Release location
         'id'                : [690],         # ISO IDs of release countries
         'pn'                : 2    ,         # Particles to release per cell

         # Simulation parameters
         'stokes'            : True,          # Toggle to use Stokes drift
         'windage'           : False,         # Toggle to use windage
         'fw'                : 0.0,           # Windage fraction

         # Runtime parameters
         'Yend'              : 2019,          # Last year of simulation
         'Mend'              : 1   ,          # Last month
         'Dend'              : 2   ,          # Last day (00:00, start)
         'dt_RK4'            : timedelta(minutes=-15),  # RK4 time-step

         # Output parameters
         'dt_out'            : timedelta(hours=6),     # Output frequency
         'fn_out'            : 'test',                 # Output filename

         # Other parameters
         'update'            : True,                   # Update grid files
         'plastic'           : True,                   # Write plastic data
         'p_param'           : {'l'  : 50.,            # Plastic length scale
                                'cr' : 0.15}}          # Fraction entering sea

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'model': os.path.dirname(os.path.realpath(__file__)) + '/MODEL_DATA/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/GRID_DATA/',
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/PLASTIC_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/'}

# FILE HANDLES
fh = {'ocean':   sorted(glob(dirs['model'] + 'OCEAN_*.nc')),
      'wave':    sorted(glob(dirs['model'] + 'WAVE_*.nc')),
      'grid':    dirs['grid'] + 'griddata.nc',
      'traj':    dirs['traj'] + 'backtrack_test.nc',
      'sid':     dirs['traj'] + 'sid.nc'}

##############################################################################
# SET UP PARTICLE RELEASEs                                                   #
##############################################################################

# Calculate the starting time (t0) for the model data
with Dataset(fh['ocean'][0], 'r') as nc:
    param['t0'] = nc.variables['time']
    param['t1'] = nc.variables['time']
    param['t0'] = num2date(param['t0'][0], param['t0'].units)
    param['t1'] = num2date(param['t1'][-1], param['t1'].units)

    # Check that the provided simulation end time is consistent with the
    # provided files
    if (datetime(year=param['Yend'],
                 month=param['Mend'],
                 day=param['Dend']) - param['t0']).total_seconds() < 0:
        raise FileNotFoundError('File input does not span simulation time')

    # Calculate the offset in seconds for the first ocean frame
    # Ocean time = seconds from start of first year
    # Parcels time = seconds from start of first frame
    # t0 = Ocean time - Parcels time

    param['t0'] = (param['t0'] - datetime(year=param['Ymin'],
                                          month=param['Mmin'],
                                          day=1)).total_seconds()

grid = mdm.gridgen(fh, dirs, param, plastic=True)

# Calculate the times for particle releases
particles = {'time_array' : mdm.release_time(param, mode=param['mode'])}

# Calculate the locations for particle releases
particles['loc_array'] = mdm.release_loc(param, fh)

# Calculate a final array of release positions, IDs, and times
particles['pos'] = mdm.add_times(particles)

print()



##############################################################################
# PARAMETERS                                                                 #
##############################################################################

# # Release timing
# Years  = [2019, 2019]  # Minimum and maximum release year
# Months = [1, 11]        # Minimum and maximum release month
# RPM    = 1             # Particle releases per calender month
# mode   = 'end'         # Release at start or end of month

# # Release locations
# CountryIDs = [690]     # ISO country codes for starting locations
# PN         = 2       # Sqrt of number of particles per cell (must be even!)

# Runtime parameters
# sim_T      = timedelta(days=10)
# sim_dt     = timedelta(minutes=-15)
# out_dt     = timedelta(hours=1)

# # Debug/Checking tools
# debug      = False
# viz_lim    = {'lonW': 46,
#               'lonE': 47,
#               'latS': -10,
#               'latN': -9}
