#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles arriving at a country backward in time using the
CMEMS GLORYS12V1 reanalysis (current and Stokes drift)
@author: Noam Vogt-Vincent
"""

import MD_Methods as mdm
import numpy as np
import os
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
         'Mmin'              : 12   ,          # First release month
         'Mmax'              : 12  ,          # Last release month
         'RPM'               : 1   ,          # Releases per month
         'mode'              :'START',        # Release at END or START

         # Release location
         'id'                : [690],         # ISO IDs of release countries
         'pn'                : 100    ,         # Particles to release per cell

         # Simulation parameters
         'stokes'            : True,          # Toggle to use Stokes drift
         'windage'           : False,         # Toggle to use windage
         'fw'                : 0.0,           # Windage fraction
         'max_age'           : 1.,           # Max age (years). 0 == inf.

         # Runtime parameters
         'Yend'              : 2019,          # Last year of simulation
         'Mend'              : 1   ,          # Last month
         'Dend'              : 2   ,          # Last day (00:00, start)
         'dt_RK4'            : timedelta(minutes=-15),  # RK4 time-step

         # Output parameters
         'dt_out'            : timedelta(hours=6),     # Output frequency
         'fn_out'            : 'test.nc',                 # Output filename

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
      'traj':    dirs['traj'] + param['fn_out'],
      'sid':     dirs['traj'] + 'sid.nc'}

##############################################################################
# SET UP PARTICLE RELEASES                                                   #
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

    # Add the end time
    param['endtime'] = datetime(year=param['Yend'],
                                month=param['Mend'],
                                day=param['Dend'])

grid = mdm.gridgen(fh, dirs, param, plastic=True)

# Calculate the times for particle releases
particles = {'time_array' : mdm.release_time(param, mode=param['mode'])}

# Calculate the locations for particle releases
particles['loc_array'] = mdm.release_loc(param, fh)

# Calculate a final array of release positions, IDs, and times
particles['pos'] = mdm.add_times(particles)

##############################################################################
# SET UP FIELDSETS                                                           #
##############################################################################

# Chunksize for parallel execution
cs_OCEAN = {'time': ('time', 2),
            'lat': ('latitude', 512),
            'lon': ('longitude', 512)}

cs_WAVE  = {'time': ('time', 2),
            'lat': ('latitude', 512),
            'lon': ('longitude', 512)}

# OCEAN (CMEMS GLORYS12V1)
filenames = fh['ocean']

variables = {'U': 'uo',
              'V': 'vo'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset_ocean = FieldSet.from_netcdf(filenames, variables, dimensions,
                                      chunksize=cs_OCEAN)

# WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
filenames = fh['wave']

variables = {'U': 'VSDX',
             'V': 'VSDY'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset_wave = FieldSet.from_netcdf(filenames, variables, dimensions,
                                      chunksize=cs_WAVE)

fieldset = FieldSet(U=fieldset_ocean.U+fieldset_wave.U,
                    V=fieldset_ocean.V+fieldset_wave.V)

# ADD THE LSM, ID, CDIST, AND CNORM FIELDS
lsm_rho = Field.from_netcdf(fh['grid'],
                            variable='lsm_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            allow_time_extrapolation=True)

id_psi  = Field.from_netcdf(fh['grid'],
                            variable='id_psi',
                            dimensions={'lon': 'lon_psi',
                                        'lat': 'lat_psi'},
                            interp_method='nearest',
                            allow_time_extrapolation=True)

cdist   = Field.from_netcdf(fh['grid'],
                            variable='cdist_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            allow_time_extrapolation=True)

cnormx  = Field.from_netcdf(fh['grid'],
                            variable='cnormx_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            mesh='spherical',
                            allow_time_extrapolation=True)

cnormy  = Field.from_netcdf(fh['grid'],
                            variable='cnormy_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            mesh='spherical',
                            allow_time_extrapolation=True)

fieldset.add_field(cdist)
fieldset.add_field(id_psi)
fieldset.add_field(cnormx)
fieldset.add_field(cnormy)
fieldset.add_field(lsm_rho)

fieldset.cnormx_rho.units = GeographicPolar()
fieldset.cnormy_rho.units = Geographic()

# ADD THE PERIODIC BOUNDARY
fieldset.add_constant('halo_west', -180.)
fieldset.add_constant('halo_east', 180.)
fieldset.add_periodic_halo(zonal=True)

# ADD MAXIMUM PARTICLE AGE (IF LIMITED AGE)
if param['max_age']:
    fieldset.add_constant('max_age', param['max_age']*3600*24*365.25)

##############################################################################
# KERNELS                                                                    #
##############################################################################

class debris(JITParticle):
    # Land-sea mask (if particle has beached)
    lsm = Variable('lsm',
                   dtype=np.float32,
                   initial=0,
                   to_write=False)

    # Source ID
    sid = Variable('sid',
                   dtype=np.int32,
                   initial=fieldset.id_psi,
                   to_write='once')

    # Particle distance from land
    cd  = Variable('cd',
                   dtype=np.float32,
                   initial=0.,
                   to_write=False)

    # Time at sea (ocean time)
    ot  = Variable('ot',
                   dtype=np.int32,
                   initial=0,
                   to_write=False)

    # Velocity away from coast (to prevent beaching)
    uc  = Variable('uc',
                   dtype=np.float32,
                   initial=0.,
                   to_write=False)

    vc  = Variable('vc',
                   dtype=np.float32,
                   initial=0.,
                   to_write=False)


def beach(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it is beached
    particle.lsm = fieldset.lsm_rho[particle]

    if particle.lsm >= 0.999:
        particle.delete()

def antibeach(particle, fieldset, time):
    #  Kernel to repel particles from the coast
    particle.cd = fieldset.cdist_rho[particle]

    if particle.cd < 0.5:

        particle.uc = fieldset.cnormx_rho[particle]
        particle.vc = fieldset.cnormy_rho[particle]

        particle.uc *= -1*(particle.cd - 0.5)**2
        particle.vc *= -1*(particle.cd - 0.5)**2

        particle.lon += particle.uc*particle.dt
        particle.lat += particle.vc*particle.dt

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    #  (possible in certain configurations)
    particle.delete()

def oldParticle(particle, fieldset, time):
    # Remove particles older than given age
    particle.ot -= particle.dt

    if particle.ot > fieldset.max_age:
        particle.delete()

def periodicBC(particle, fieldset, time):
    # Move the particle across the periodic boundary
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west

##############################################################################
# INITIALISE SIMULATION AND RUN                                              #
##############################################################################
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=debris,
                             lon  = particles['pos']['lon'],
                             lat  = particles['pos']['lat'],
                             time = particles['pos']['time'])
print(str(len(particles['pos']['time'])) + ' particles released!')

traj = pset.ParticleFile(name=fh['traj'],
                         outputdt=param['dt_out'])

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(antibeach) +
           pset.Kernel(beach) +
           pset.Kernel(periodicBC))

pset.execute(kernels,
             endtime=param['endtime'],
             dt=param['dt_RK4'],
             recovery={ErrorCode.ErrorOutOfBounds: deleteParticle,
                       ErrorCode.ErrorInterpolation: deleteParticle},
             output_file=traj)

traj.export()