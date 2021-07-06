#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:58:56 2021
Script to test particle tracking on the CMEMS A-grid
@author: noam
"""

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import Variable, plotTrajectoriesFile, DiffusionUniformKh, Field
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta
import os
from marinedebrismethods import cmems_globproc, one_release, time_stagger
import matplotlib.pyplot as plt
import cmocean.cm as cm

##############################################################################
# Particle defs  #############################################################
##############################################################################


def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    particle.delete()

def periodicBC(particle, fieldset, time):
    # Move the particle across the periodic boundary
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west

##############################################################################
# File locations #############################################################
##############################################################################

script_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = script_dir + '/CMEMS/'
trajectory_dir = script_dir + '/TRAJ/'

cmems_fh = model_dir + 'CMEMS_1993.nc'
cmems_proc_fh = model_dir + 'globmask.nc'
traj_fh = trajectory_dir + '/Aldabra_test.nc'

##############################################################################
# Parameters #################################################################
##############################################################################

# The number of particles to be seeded per cell = pn^2
pn = 30

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

# Display region
(lon_min, lon_max, lat_min, lat_max) = (139, 141, 34, 36)
fig_fh = 'flow_test.png'

##############################################################################
# Set up particle tracking ###################################################
##############################################################################

# Import the mask and labelled cells (coast)
masks = cmems_globproc(cmems_fh, cmems_proc_fh)

lat_psi = masks['lat_psi']
lon_psi = masks['lon_psi']
lat_rho = masks['lat_rho']
lon_rho = masks['lon_rho']

lsm_psi     = masks['lsm_psi']
lsm_rho     = masks['lsm_rho']
coast_psi   = masks['coast_psi']

# Calculate grid indices for graphing
jmin_psi = np.searchsorted(lon_psi, lon_min) - 1
jmin_rho = jmin_psi
jmax_psi = np.searchsorted(lon_psi, lon_max)
jmax_rho = jmax_psi + 1

imin_psi = np.searchsorted(lat_psi, lat_min) - 1
imin_rho = imin_psi
imax_psi = np.searchsorted(lat_psi, lat_max)
imax_rho = imax_psi + 1

disp_lon_rho = lon_rho[jmin_rho:jmax_rho]
disp_lat_rho = lat_rho[imin_rho:imax_rho]

disp_lon_psi = lon_psi[jmin_psi:jmax_psi]
disp_lat_psi = lat_psi[imin_psi:imax_psi]

disp_lsm_psi   = lsm_psi[imin_psi:imax_psi, jmin_psi:jmax_psi]
disp_lsm_rho   = lsm_rho[imin_rho:imax_rho, jmin_rho:jmax_rho]
disp_coast_psi = coast_psi[imin_psi:imax_psi, jmin_psi:jmax_psi]

with Dataset(cmems_fh, mode='r') as nc:
    disp_u_rho   = nc.variables['uo'][0, 0,
                                      imin_rho:imax_rho,
                                      jmin_rho:jmax_rho]

    disp_v_rho   = nc.variables['vo'][0, 0,
                                      imin_rho:imax_rho,
                                      jmin_rho:jmax_rho]

# Plot the map
f, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Plot the rho grid
for i in range(len(disp_lat_rho)):
    ax.plot([lon_min, lon_max], [disp_lat_rho[i], disp_lat_rho[i]],
            'k--', linewidth=0.5)

for j in range(len(disp_lon_rho)):
    ax.plot([disp_lon_rho[j], disp_lon_rho[j]], [lat_min, lat_max],
            'k--', linewidth=0.5)

# Plot the lsm_psi mask
disp_lon_rho_, disp_lat_rho_ = np.meshgrid(disp_lon_rho, disp_lat_rho)
disp_lon_psi_, disp_lat_psi_ = np.meshgrid(disp_lon_psi, disp_lat_psi)

ax.pcolormesh(disp_lon_rho, disp_lat_rho, disp_lsm_psi, cmap=cm.topo,
              vmin=-0.5, vmax=1.5)

# Plot the lsm_psi nodes
ax.scatter(disp_lon_rho_, disp_lat_rho_, c=disp_lsm_rho, s=10, marker='o',
           cmap=cm.gray_r)

# Plot the original lsm (on rho)
ax.scatter(disp_lon_psi_, disp_lat_psi_, s=20, marker='+', c=disp_lsm_psi,
           cmap=cm.gray_r, linewidth=0.1)

# Plot the velocity field
ax.quiver(disp_lon_rho, disp_lat_rho, disp_u_rho, disp_v_rho)

# Save
plt.savefig(script_dir + '/' + fig_fh, dpi=300)











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
print(str(len(pos0[:, 0])) + ' particles released!')

# Set up the simulation
traj = pset.ParticleFile(name=traj_fh,
                         outputdt=timedelta(hours=1),
                         write_ondelete=True)

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(periodicBC))

pset.execute(kernels,
             runtime=timedelta(days=sim_time),
             dt = -timedelta(minutes=30),
             output_file=traj)

traj.export()
plotTrajectoriesFile(traj_fh)