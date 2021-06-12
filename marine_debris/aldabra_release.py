#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:58:56 2021
The basis script for Arctic tern releases
@author: noam
"""

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import Variable, plotTrajectoriesFile, DiffusionUniformKh, Field
from parcels.tools.converters import Geographic
import numpy as np
from datetime import timedelta
import os
from geographiclib.geodesic import Geodesic
from marinedebrismethods import cmems_proc, one_release, time_stagger


##############################################################################
# Particle defs  #############################################################
##############################################################################

class marineDebris(JITParticle):
    ocean_time = Variable('ocean_time', dtype=np.float32, initial=0.)
    sink_time = Variable('sink_time', dtype=np.float32, initial=0.)
    lon0 = Variable('lon0', dtype=np.float32, initial=0.)
    lat0 = Variable('lat0', dtype=np.float32, initial=0.)

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    particle.delete()

def periodicBC(particle, fieldset, time):
    # Move the particle across the periodic boundary
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west

def drift(particle, fieldset, time):
    # Record the sink time (i.e. simulation 'start' time) of the particle
    if particle.ocean_time == 0.:
        particle.sink_time = time

    # Update the particle age
    particle.ocean_time += particle.dt


##############################################################################
# File locations #############################################################
##############################################################################

script_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = script_dir + '/CMEMS/'
trajectory_dir = script_dir + '/TRAJ/'

cmems_fh = model_dir + '2019.nc'
cmems_proc_fh = model_dir + 'masks.nc'
traj_fh = trajectory_dir + '/Aldabra_test.nc'

##############################################################################
# Parameters #################################################################
##############################################################################

# The number of particles to be seeded per cell = pn^2
pn = 10

# Simulation length
sim_time = 30 # days

# Bounds for sink region of interest
lon0 = 46
lon1 = 47
lat0 = -10
lat1 = -9

# Release interval
last_release     = 362  # days
release_number   = 1
release_interval = 30   # days


##############################################################################
# Set up particle tracking ###################################################
##############################################################################

# Import the mask and labelled cells
coast, lsm, lon, lat, lon_bnd, lat_bnd = cmems_proc(cmems_fh, cmems_proc_fh)

# Designate the source locations
pos0 = one_release(lon0, lon1, lat0, lat1,
                   coast,
                   lon, lat, lon_bnd, lat_bnd,
                   pn)

# Add release times
pos0 = time_stagger(pos0, last_release, release_number, release_interval)


# Set up the fieldset
filenames = {'U': {'lon': cmems_fh, 'lat': cmems_fh,
                   'time': cmems_fh, 'data': cmems_fh},
             'V': {'lon': cmems_fh, 'lat': cmems_fh,
                   'time': cmems_fh, 'data': cmems_fh}}

variables = {'U': 'uo',
             'V': 'vo'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset = FieldSet.from_netcdf(filenames,
                                variables,
                                dimensions)

# Add the periodic boundary
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_periodic_halo(zonal=True)


# Add diffusion
# fieldset.add_constant_field('Kh_zonal', Kh_zonal, mesh='spherical')
# fieldset.add_constant_field('Kh_meridional', Kh_meridional, mesh='spherical')

# Set up the particle set
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=marineDebris,
                             lon = pos0[:, 0],
                             lat = pos0[:, 1],
                             time = pos0[:, 2])

# Set up the simulation
traj = pset.ParticleFile(name=traj_fh,
                         outputdt=timedelta(hours=1))

pset.execute((pset.Kernel(AdvectionRK4) +
              pset.Kernel(periodicBC) +
              pset.Kernel(drift)),
              runtime=timedelta(days=sim_time),
              dt = -timedelta(minutes=30),
              output_file=traj)

traj.export()
plotTrajectoriesFile(traj_fh)