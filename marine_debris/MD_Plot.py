#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script plots output from marine debris simulations
@author: noam
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
# MODE                                                                       #
##############################################################################

# OPTIONS:
# 'CLOSE' : Low, cell-level view with options to view fluid velocity,
#           particle properties, etc.
# 'FAR'   : High-level view of bulk particle trajectories, options to
#           sub-sample trajectories.

mode = 'CLOSE'

# PARAMETERS
param = {# View region
         'lon_min' : +45.0,
         'lon_max' : +46.0,
         'lat_min' : -10.0,
         'lat_max' : -09.0,

         # Display options
         'show_particle_var'   : True,
         'particle_var'        : 'lsm_rho',

         'show_fluid_velocity' : True,
         'fluid_time'          : 0}


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