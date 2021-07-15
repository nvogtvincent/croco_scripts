#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:58:56 2021
Script to test particle tracking on the CMEMS A-grid
@author: noam
"""

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
from parcels import Variable, plotTrajectoriesFile, DiffusionUniformKh, Field
from parcels import Geographic, GeographicPolar
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

class debris(JITParticle):
    # Source cell is defined at particle initiation and is the origin reef cell
    lsm_state = Variable('lsm_state',
                         dtype=np.int8,
                         initial=0)

    cdist_state = Variable('cdist_state',
                           dtype=np.float32,
                           initial=0.)

    cnormx_state = Variable('cnormx_state',
                            dtype=np.float32,
                            initial=0.)

    cnormy_state = Variable('cnormy_state',
                            dtype=np.float32,
                            initial=0.)

# def beaching(particle, fieldset, time):
#     #  Recovery kernel to delete a particle if it is beached
#     particle.lsm_state = fieldset.lsm_psi[time, particle.depth, particle.lat, particle.lon]

#     if particle.lsm_state == 1:
#         particle.delete()

def antibeach(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it is beached
    particle.cdist_state = fieldset.cdist_rho[time, particle.depth, particle.lat, particle.lon]

    if particle.cdist_state < 1:
        particle.cnormx_state = fieldset.cnormx_rho[time, particle.depth, particle.lat, particle.lon]
        particle.cnormy_state = fieldset.cnormy_rho[time, particle.depth, particle.lat, particle.lon]

        particle.lon += particle.cnormx_state*particle.dt
        particle.lat += particle.cnormy_state*particle.dt

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
traj_fh = trajectory_dir + '/land_test.nc'

##############################################################################
# Parameters #################################################################
##############################################################################

# The number of particles to be seeded per cell = pn^2
pn = 50

# Simulation length
sim_time = 1.5 # days

# Bounds for release zone (linear)
(lon_0, lon_1, lat_0, lat_1) = (139.0, 139.0, 34.0, 35.00)

# Diffusion parameters
Kh_meridional = 1.
Kh_zonal      = 1.

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

cnormx_rho  = masks['cnormx_rho']
cnormy_rho  = masks['cnormy_rho']

# Designate the source locations
posx = np.linspace(lon_0, lon_1, num=pn)
posy = np.linspace(lat_0, lat_1, num=pn)
posx = np.array([139.])
posy = np.array([34.8])
post = np.zeros_like(posx)


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
lsm = Field.from_netcdf(cmems_proc_fh,
                        variable='lsm_psi',
                        dimensions={'lon': 'lon_psi',
                                    'lat': 'lat_psi'},
                        interp_method='nearest',
                        allow_time_extrapolation=True)

fieldset.add_field(lsm)

# Add the cdist and cnorm fields
cdist = Field.from_netcdf(cmems_proc_fh,
                          variable='cdist_rho',
                          dimensions={'lon': 'lon_rho',
                                      'lat': 'lat_rho'},
                          interp_method='linear',
                          allow_time_extrapolation=True)

cnormx = Field.from_netcdf(cmems_proc_fh,
                           variable='cnormx_rho',
                           dimensions={'lon': 'lon_rho',
                                       'lat': 'lat_rho'},
                           interp_method='linear',
                           mesh='spherical',
                           allow_time_extrapolation=True)

cnormy = Field.from_netcdf(cmems_proc_fh,
                           variable='cnormy_rho',
                           dimensions={'lon': 'lon_rho',
                                       'lat': 'lat_rho'},
                           interp_method='linear',
                           mesh='spherical',
                           allow_time_extrapolation=True)

fieldset.add_field(cdist)
fieldset.add_field(cnormx)
fieldset.add_field(cnormy)

fieldset.cnormx_rho.units = GeographicPolar()
fieldset.cnormy_rho.units = Geographic()

# Add diffusion
fieldset.add_constant_field('Kh_zonal', Kh_zonal, mesh='spherical')
fieldset.add_constant_field('Kh_meridional', Kh_meridional, mesh='spherical')

# Set up the particle set
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=debris,
                             lon = posx,
                             lat = posy,
                             time = post)
print(str(len(posx)) + ' particles released!')

# Set up the simulation
traj = pset.ParticleFile(name=traj_fh,
                         outputdt=timedelta(minutes=60))

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(periodicBC) +
           pset.Kernel(antibeach))

pset.execute(kernels,
             runtime=timedelta(days=sim_time),
             dt = timedelta(minutes=5),
             recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
             output_file=traj)

traj.export()

##############################################################################
# Plotting ###################################################################
##############################################################################

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

cnormx   = cnormx_rho[imin_rho:imax_rho,jmin_rho:jmax_rho]

cnormy   = cnormy_rho[imin_rho:imax_rho,jmin_rho:jmax_rho]

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

# Plot the coast_psi mask
ax.pcolormesh(disp_lon_rho, disp_lat_rho,
              np.ma.masked_values(disp_coast_psi, 0), cmap=cm.topo,
              vmin=0, vmax=3)

# Plot the lsm_rho nodes
ax.scatter(disp_lon_rho_, disp_lat_rho_, c=disp_lsm_rho, s=10, marker='o',
           cmap=cm.gray_r)

# # Plot the  lsm_psi nodes
# ax.scatter(disp_lon_psi_, disp_lat_psi_, s=20, marker='+', c=disp_lsm_psi,
#             cmap=cm.gray_r, linewidth=0.3)

# Plot the velocity field and BCs
ax.quiver(disp_lon_rho, disp_lat_rho, disp_u_rho, disp_v_rho)
ax.quiver(disp_lon_rho, disp_lat_rho, cnormx, cnormy, units='inches', scale=5, color='w')

# Load the trajectories
with Dataset(traj_fh, mode='r') as nc:
    plat  = nc.variables['lat'][:]
    plon  = nc.variables['lon'][:]
    pstate = nc.variables['lsm_state'][:]

pnum = np.shape(plat)[0]
pt   = np.shape(plat)[1]

for particle in range(pnum):
    # if plon[particle, 0] == plon[particle, 1]:
    #     ax.scatter(plon[particle, 0], plat[particle, 0], c='r', s=15, marker='o')
    # else:
    #     ax.scatter(plon[particle, 0], plat[particle, 0], c='b', s=15, marker='o')
    ax.plot(plon[particle, :], plat[particle, :], 'w-', linewidth=0.5)
    # ax.scatter(plon[particle, :], plat[particle, :],
    #            c = pstate[particle, :],
    #            cmap = cm.gray_r,
    #            s = 10,
    #            marker = 's')




# Save
plt.savefig(script_dir + '/' + fig_fh, dpi=300)




