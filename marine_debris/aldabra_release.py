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
    ocean_time = Variable('ocean_time', dtype=np.int32, initial=0)

    # Sink time: time particle ends up at sink site
    sink_time = Variable('sink_time', dtype=np.int32, initial=0)

    # Records of coastal encounters:
    # s[x] = grid cell number of encounter
    # t[x] = ocean time of encounter
    s0 = Variable('s0', dtype=np.int32, initial=0)
    t0 = Variable('t0', dtype=np.int32, initial=0)

    s1 = Variable('s1', dtype=np.int32, initial=0)
    t1 = Variable('t1', dtype=np.int32, initial=0)

    s2 = Variable('s2', dtype=np.int32, initial=0)
    t2 = Variable('t2', dtype=np.int32, initial=0)

    s3 = Variable('s3', dtype=np.int32, initial=0)
    t3 = Variable('t3', dtype=np.int32, initial=0)

    s4 = Variable('s4', dtype=np.int32, initial=0)
    t4 = Variable('t4', dtype=np.int32, initial=0)

    s5 = Variable('s5', dtype=np.int32, initial=0)
    t5 = Variable('t5', dtype=np.int32, initial=0)

    s6 = Variable('s6', dtype=np.int32, initial=0)
    t6 = Variable('t6', dtype=np.int32, initial=0)

    s7 = Variable('s7', dtype=np.int32, initial=0)
    t7 = Variable('t7', dtype=np.int32, initial=0)

    s8 = Variable('s8', dtype=np.int32, initial=0)
    t8 = Variable('t8', dtype=np.int32, initial=0)

    s9 = Variable('s9', dtype=np.int32, initial=0)
    t9 = Variable('t9', dtype=np.int32, initial=0)

    # Events: number of events encountered
    events = Variable('events', dtype=np.int8, initial=0)


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
    location_status = fieldset.groups[time,
                                      particle.depth,
                                      particle.lat,
                                      particle.lon]

    # Minimum time at sea (days)
    time_at_sea_min = 7

    if (location_status > 0 and particle.ocean_time > 86400*time_at_sea_min):
        # Yes, I cringed writing this but I'm not sure a more elegant solution
        # currently exists within Parcels...

        if particle.events == 0:
            particle.t0 = particle.ocean_time
            particle.s0 = location_status
        elif particle.events == 1:
            particle.t1 = particle.ocean_time
            particle.s1 = location_status
        elif particle.events == 2:
            particle.t2 = particle.ocean_time
            particle.s2 = location_status
        elif particle.events == 3:
            particle.t3 = particle.ocean_time
            particle.s3 = location_status
        elif particle.events == 4:
            particle.t4 = particle.ocean_time
            particle.s4 = location_status
        elif particle.events == 5:
            particle.t5 = particle.ocean_time
            particle.s5 = location_status
        elif particle.events == 6:
            particle.t6 = particle.ocean_time
            particle.s6 = location_status
        elif particle.events == 7:
            particle.t7 = particle.ocean_time
            particle.s7 = location_status
        elif particle.events == 8:
            particle.t8 = particle.ocean_time
            particle.s8 = location_status
        elif particle.events == 9:
            particle.t9 = particle.ocean_time
            particle.s9 = location_status

        particle.events += 1

        if particle.events == 10:
            particle.delete()

    # Update the particle age
    particle.ocean_time -= particle.dt

    # Delete all at the end of the simulation
    if (particle.ocean_time > 360*86400 and particle.events > 0):
        particle.delete()


##############################################################################
# File locations #############################################################
##############################################################################

script_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = script_dir + '/CMEMS/'
trajectory_dir = script_dir + '/TRAJ/'

cmems_fh = model_dir + 'CMEMS_2019.nc'
cmems_proc_fh = model_dir + 'masks.nc'
traj_fh = trajectory_dir + '/Aldabra_test.nc'

##############################################################################
# Parameters #################################################################
##############################################################################

# The number of particles to be seeded per cell = pn^2
pn = 20

# Simulation length
sim_time = 364 # days

# Bounds for sink region of interest
lon0 = 46
lon1 = 47
lat0 = -10
lat1 = -9

# Release interval
# Remember: t=0 is 12pm (not midnight!!!)
last_release     = 364  # days
release_number   = 1
release_interval = 30   # days


##############################################################################
# Set up particle tracking ###################################################
##############################################################################

# Import the mask and labelled cells (coast)
coast, groups, lsm, lon, lat, lon_bnd, lat_bnd = cmems_proc(cmems_fh, cmems_proc_fh)

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

# Add the groups field (and implicitly the coast)
coastgrid = Field.from_netcdf(cmems_proc_fh,
                              variable='groups',
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

# Set up the simulation
traj = pset.ParticleFile(name=traj_fh,
                         outputdt=timedelta(hours=1),
                         write_ondelete=True)

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(periodicBC) +
           pset.Kernel(drift))

pset.execute(kernels,
             runtime=timedelta(days=sim_time),
             dt = -timedelta(minutes=30),
             output_file=traj)

traj.export()
plotTrajectoriesFile(traj_fh)