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
    # Ocean time: age of particle
    ocean_time = Variable('ocean_time', dtype=np.float32, initial=0.)

    # Sink time: time particle ends up at sink site
    sink_time = Variable('sink_time', dtype=np.float32, initial=0.)

    # [lon0, lat0] = sink coordinates
    # lon0 = Variable('lon0', dtype=np.float32, initial=0.)
    # lat0 = Variable('lat0', dtype=np.float32, initial=0.)

    # Events: number of events encountered
    events = Variable('events', dtype=np.int8, initial=0)

    # Variables to store beaching events
    lon0 = Variable('lon0', dtype=np.float32, initial=0.)
    lat0 = Variable('lat0', dtype=np.float32, initial=0.)

    lon1 = Variable('lon1', dtype=np.float32, initial=0.)
    lat1 = Variable('lat1', dtype=np.float32, initial=0.)

    lon2 = Variable('lon2', dtype=np.float32, initial=0.)
    lat2 = Variable('lat2', dtype=np.float32, initial=0.)


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

    # Record if currently in a coast cell
    location_status = fieldset.coast[time,
                                     particle.depth,
                                     particle.lat,
                                     particle.lon]

    if (location_status == 1 and particle.ocean_time > 259200.):
        lon = particle.lon
        lat = particle.lat

        if particle.events == 0:
            particle.lon0 = lon
            particle.lat0 = lat
        elif particle.events == 1:
            particle.lon1 = lon
            particle.lat1 = lat
        elif particle.events == 2:
            particle.lon2 = lon
            particle.lat2 = lat

        particle.events += 1

        if particle.events == 3:
            particle.delete()


    # if location_status == 1:
    #     particle.coast_status = 1
    # else:
    #     particle.coast_status = 0

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
pn = 3

# Simulation length
sim_time = 100 # days

# Bounds for sink region of interest
lon0 = 46
lon1 = 47
lat0 = -10
lat1 = -9

# Release interval
last_release     = 200  # days
release_number   = 1
release_interval = 30   # days


##############################################################################
# Set up particle tracking ###################################################
##############################################################################

# Import the mask and labelled cells (coast)
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

# Add the coast mask
coastgrid = Field.from_netcdf(cmems_proc_fh,
                              variable='coast',
                              dimensions={'lon': 'longitude',
                                         'lat': 'latitude'},
                              interp_method='nearest',
                              allow_time_extrapolation=True)
fieldset.add_field(coastgrid)

# Add diffusion
# fieldset.add_constant_field('Kh_zonal', Kh_zonal, mesh='spherical')
# fieldset.add_constant_field('Kh_meridional', Kh_meridional, mesh='spherical')

# Set up the particle set
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=marineDebris,
                             lon = pos0[:, 0],
                             lat = pos0[:, 1],
                             time = pos0[:, 2])

# Prevent writing unnecessary information (depth)
# for v in pset.ptype.variables:
#     if v.name == 'depth':
#         v.to_write = False

# Set up the simulation
traj = pset.ParticleFile(name=traj_fh,
                         outputdt=timedelta(hours=1),
                         write_ondelete=True)

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(periodicBC) +
           pset.Kernel(drift))

pset.execute(kernels,
             runtime=timedelta(days=sim_time),
             dt = timedelta(minutes=30),
             output_file=traj)

traj.export()
plotTrajectoriesFile(traj_fh)