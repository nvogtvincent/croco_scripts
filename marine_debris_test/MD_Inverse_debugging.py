#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles arriving at a country backward in time using the
CMEMS GLORYS12V1 reanalysis (current and Stokes drift)
@author: Noam Vogt-Vincent
"""

import os
from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode)
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
         'Mmin'              : 2  ,          # First release month
         'Mmax'              : 12  ,          # Last release month
         'RPM'               : 1   ,          # Releases per month
         'mode'              :'START',        # Release at END or START

         # Release location
         'id'                : [690],         # ISO IDs of release countries
         'pn'                : 400  ,         # Particles to release per cell

         # Simulation parameters
         'stokes'            : True,          # Toggle to use Stokes drift
         'windage'           : False,         # Toggle to use windage
         'fw'                : 0.0,           # Windage fraction

         # Runtime parameters
         'Yend'              : 2019,         # Last year of simulation
         'Mend'              : 6   ,          # Last month
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
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/'}

# FILE HANDLES
fh = {'ocean':   sorted(glob(dirs['model'] + 'OCEAN_2019*crop.nc')),
      'wave':    sorted(glob(dirs['model'] + 'WAVE_2019*crop.nc')),
      'traj':    dirs['traj'] + 'backtrack_test.nc'}

##############################################################################
# SET UP PARTICLE RELEASES                                                   #
##############################################################################

# Just writing in the known parameters for a particle that triggers the error
lon0 = 55.8354190826416
lat0 = -4.272917652130127
t0   = 7905600

param['endtime'] = datetime(year=param['Yend'],
                            month=param['Mend'],
                            day=param['Dend'])

##############################################################################
# SET UP FIELDSETS                                                           #
##############################################################################

# OCEAN (CMEMS GLORYS12V1)
filenames = fh['ocean']

variables = {'U': 'uo',
              'V': 'vo'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset_ocean = FieldSet.from_netcdf(filenames, variables, dimensions)

# WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
filenames = fh['wave']

variables = {'U': 'VSDX',
             'V': 'VSDY'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset_wave = FieldSet.from_netcdf(filenames, variables, dimensions)

fieldset = FieldSet(U=fieldset_ocean.U+fieldset_wave.U,
                    V=fieldset_ocean.V+fieldset_wave.V)

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    #  (possible in certain configurations)
    particle.delete()

##############################################################################
# INITIALISE SIMULATION AND RUN                                              #
##############################################################################
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=JITParticle,
                             lon  = lon0,
                             lat  = lat0,
                             time = t0)


traj = pset.ParticleFile(name=fh['traj'],
                         outputdt=param['dt_out'])

kernels = (pset.Kernel(AdvectionRK4))

pset.execute(kernels,
             endtime=param['endtime'],
             dt=param['dt_RK4'],
             recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
             output_file=traj)

traj.export()

from parcels import plotTrajectoriesFile

plotTrajectoriesFile(fh['traj'])
