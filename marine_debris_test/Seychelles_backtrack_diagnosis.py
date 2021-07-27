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
                     ErrorCode, Geographic, GeographicPolar, Variable)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob

##############################################################################
# DIRECTORIES                                                                #
##############################################################################

dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'model': os.path.dirname(os.path.realpath(__file__)) + '/MODEL_DATA/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/GRID_DATA/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/'}

fh = {'ocean':   sorted(glob(dirs['model'] + 'OCEAN_*.nc')),
      'grid':    dirs['grid'] + 'globmask.nc',
      'plastic': dirs['grid'] + 'plastic_flux.nc',
      'traj':    dirs['traj'] + 'backtrack_test.nc'}

##############################################################################
# PARAMETERS                                                                 #
##############################################################################

# Release timing
Years  = [1993, 1993]  # Minimum and maximum release year
Months = [6, 6]        # Minimum and maximum release month
RPM    = 1             # Particle releases per calender month
mode   = 'end'         # Release at start or end of month

# Release locations
CountryIDs = [690]     # ISO country codes for starting locations
PN         = 2       # Sqrt of number of particles per cell (must be even!)

# Runtime parameters
sim_T      = timedelta(days=2)
sim_dt     = timedelta(minutes=-15)
out_dt     = timedelta(hours=1)

##############################################################################
# SET UP PARTICLE RELEASE                                                    #
##############################################################################

# Calculate the starting time (t0) for the model data
with Dataset(fh['ocean'][0], 'r') as nc:
    t0 = nc.variables['time']
    t0 = num2date(t0[0], t0.units)

t0 = (t0 - datetime(year=Years[0], month=Months[0], day=1)).total_seconds()

# Calculate the starting times for particle releases
rtime = mdm.release_time(Years, Months, RPM, int(t0), mode=mode)

# Calculate the starting locations for particle releases
pos0 = mdm.release_loc(fh, CountryIDs, PN)

# Add the times
pos0 = mdm.add_times(pos0, rtime)

##############################################################################
# SET UP FIELDSETS                                                           #
##############################################################################

# Chunksize for parallel execution
cs_OCEAN = {'time': ('time', 2),
            'lat': ('latitude', 512),
            'lon': ('longitude', 512)}

# OCEAN (CMEMS GLORYS12V1)
filenames = fh['ocean']

variables = {'U': 'uo',
              'V': 'vo'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, chunksize=cs_OCEAN)

##############################################################################
# KERNELS                                                                    #
##############################################################################


##############################################################################
# INITIALISE SIMULATION AND RUN                                              #
##############################################################################
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=JITParticle,
                             lon  = pos0['lon'],
                             lat  = pos0['lat'],
                             time = pos0['time'])

print(str(len(pos0['lon'])) + ' particles released!')

traj = pset.ParticleFile(name=fh['traj'],
                         outputdt=out_dt)

kernels = (pset.Kernel(AdvectionRK4))

pset.execute(kernels,
             runtime=sim_T,
             dt=sim_dt,
             output_file=traj)

traj.export()

