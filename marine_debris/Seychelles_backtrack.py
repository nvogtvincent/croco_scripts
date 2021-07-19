#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles arriving at Seychelles backward in time using the
CMEMS GLORYS12V1 reanalysis (current and Stokes drift)
@author: Noam Vogt-Vincent
"""
import marinedebrismethods as mdm
import numpy as np
import os
import matplotlib.pyplot as plt
import cmocean.cm as cm
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime

##############################################################################
# KERNELS                                                                    #
##############################################################################

##############################################################################
# DIRECTORIES                                                                #
##############################################################################

dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'model': os.path.dirname(os.path.realpath(__file__)) + '/CMEMS/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/'}

fh = {'ocean': dirs['model'] + 'OCEAN_1993.nc',
      'wave': dirs['model'] + 'WAVE_1993.nc',
      'grid': dirs['model'] + 'globmask.nc',
      'plastic': dirs['model'] + 'plastic_flux.nc',
      'traj': dirs['traj'] + 'backtrack_test.nc'}

##############################################################################
# PARAMETERS                                                                 #
##############################################################################

# Release timing
Years  = [1993, 1994]  # Minimum and maximum release year
Months = [1, 1]        # Minimum and maximum release month
RPM    = 1             # Particle releases per calender month

# Release locations
CountryIDs = [690]     # ISO country codes for starting locations
PN         = 3       # Sqrt of number of particles per cell (must be even!)

##############################################################################
# SET UP RUN                                                                 #
##############################################################################

# Calculate the starting time (t0) for the model data
with Dataset(fh['ocean'], 'r') as nc:
    t0 = nc.variables['time']
    t0 = num2date(t0[0], t0.units)

t0 = (t0 - datetime(year=Years[0], month=Months[0], day=1)).total_seconds()

# Calculate the starting times for particle releases
rtime = mdm.release_time(Years, Months, RPM, int(t0))

# Calculate the starting locations for particle releases
pos0 = mdm.release_loc(fh, CountryIDs, PN)

# Add the times
pos0 = mdm.add_times(pos0, rtime)
print()

f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(pos0['lon'], pos0['lat'], s=1)
ax.set_xlim(46,47)
ax.set_ylim(-10,-9)