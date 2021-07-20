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
Years  = [1993, 1993]  # Minimum and maximum release year
Months = [2, 11]        # Minimum and maximum release month
RPM    = 1             # Particle releases per calender month

# Release locations
CountryIDs = [690]     # ISO country codes for starting locations
PN         = 8       # Sqrt of number of particles per cell (must be even!)

# Runtime parameters
sim_T      = timedelta(days=30*10)
sim_dt     = timedelta(minutes=-5)
out_dt     = timedelta(hours=1)

# Debug/Checking tools
debug      = False
viz_lim    = {'lonW': 46,
              'lonE': 47,
              'latS': -10,
              'latN': -9}

##############################################################################
# SET UP PARTICLE RELEASE                                                    #
##############################################################################

# Run the mask script if necessary (but plastics must be updated manually)
update_mask = False
if update_mask:
    masks = mdm.cmems_globproc(fh['ocean'], fh['grid'], add_seychelles=True)

# Calculate the starting time (t0) for the model data
with Dataset(fh['ocean'], 'r') as nc:
    t0 = nc.variables['time']
    t0 = num2date(t0[0], t0.units)

t0 = (t0 - datetime(year=Years[0], month=Months[0], day=1)).total_seconds()

# Calculate the starting times for particle releases
rtime = mdm.release_time(Years, Months, RPM, int(t0), mode='end')

# Calculate the starting locations for particle releases
pos0 = mdm.release_loc(fh, CountryIDs, PN)

# Add the times
pos0 = mdm.add_times(pos0, rtime)

##############################################################################
# SET UP FIELDSETS                                                           #
##############################################################################

# Chunksize for parallel execution
cs_OCEAN = {'time': ('time', 2),
            'lat': ('lat', 2048),
            'lon': ('lon', 2048)}

cs_WAVE  = {'time': ('time', 2),
            'lat': ('lat', 2048),
            'lon': ('lon', 2048)}

# OCEAN (CMEMS GLORYS12V1)
filenames = {'U': {'lon': fh['ocean'], 'lat': fh['ocean'],
                    'time': fh['ocean'], 'data': fh['ocean']},
              'V': {'lon': fh['ocean'], 'lat': fh['ocean'],
                    'time': fh['ocean'], 'data': fh['ocean']}}

variables = {'U': 'uo',
              'V': 'vo'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset_ocean = FieldSet.from_netcdf(filenames, variables, dimensions,
                                      chunksize=cs_OCEAN)

# WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
filenames = {'U': {'lon': fh['wave'], 'lat': fh['wave'],
                   'time': fh['wave'], 'data': fh['wave']},
             'V': {'lon': fh['wave'], 'lat': fh['wave'],
                   'time': fh['wave'], 'data': fh['wave']}}

variables = {'U': 'VSDX',
             'V': 'VSDY'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

fieldset_wave = FieldSet.from_netcdf(filenames, variables, dimensions,
                                     chunksize=cs_WAVE)

fieldset = FieldSet(U=fieldset_ocean.U+fieldset_wave.U,
                    V=fieldset_ocean.V+fieldset_wave.V)


# ADD THE LSM, ID, CDIST, AND CNORM FIELDS
lsm   = Field.from_netcdf(fh['grid'],
                          variable='lsm_psi',
                          dimensions={'lon': 'lon_psi',
                                      'lat': 'lat_psi'},
                          interp_method='nearest',
                          allow_time_extrapolation=True)

id_psi = Field.from_netcdf(fh['grid'],
                           variable='id_psi',
                           dimensions={'lon': 'lon_psi',
                                       'lat': 'lat_psi'},
                           interp_method='nearest',
                           allow_time_extrapolation=True)

cdist = Field.from_netcdf(fh['grid'],
                          variable='cdist_rho',
                          dimensions={'lon': 'lon_rho',
                                      'lat': 'lat_rho'},
                          interp_method='linear',
                          allow_time_extrapolation=True)

cnormx = Field.from_netcdf(fh['grid'],
                           variable='cnormx_rho',
                           dimensions={'lon': 'lon_rho',
                                       'lat': 'lat_rho'},
                           interp_method='linear',
                           mesh='spherical',
                           allow_time_extrapolation=True)

cnormy = Field.from_netcdf(fh['grid'],
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
fieldset.add_field(lsm)

fieldset.cnormx_rho.units = GeographicPolar()
fieldset.cnormy_rho.units = Geographic()

# ADD THE PERIODIC BOUNDARY
fieldset.add_constant('halo_west', -180.)
fieldset.add_constant('halo_east', 180.)
fieldset.add_periodic_halo(zonal=True)

##############################################################################
# KERNELS                                                                    #
##############################################################################

class debris(JITParticle):
    # Land-sea mask (if particle has beached)
    lsm = Variable('lsm',
                   dtype=np.int8,
                   initial=0,
                   to_write=False)

    # Source ID
    # sid = Variable('sid',
    #                dtype=np.int32,
    #                initial=fieldset.id_psi,
    #                to_write='once')

    # Particle distance from land
    cd = Variable('cd',
                  dtype=np.float32,
                  initial=0.,
                  to_write=False)

    # Time at sea (ocean time)
    ot = Variable('ot',
                  dtype=np.int32,
                  initial=0,
                  to_write=False)

    # Velocity away from coast (to prevent beaching)
    uc = Variable('uc',
                  dtype=np.float32,
                  initial=0.,
                  to_write=False)

    vc = Variable('vc',
                  dtype=np.float32,
                  initial=0.,
                  to_write=False)

    # # Advection velocity (ocean + wave)
    uo = Variable('uo',
                  dtype=np.float32,
                  initial=0.,
                  to_write=False)

    vo = Variable('vo',
                  dtype=np.float32,
                  initial=0.,
                  to_write=False)

def beach(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it is beached
    particle.lsm = fieldset.lsm_psi[time,
                                    particle.depth,
                                    particle.lat,
                                    particle.lon]

    if particle.lsm == 1:
        particle.delete()

def antibeach(particle, fieldset, time):
    #  Kernel to repel particles from the coast
    particle.cd = fieldset.cdist_rho[time,
                                     particle.depth,
                                     particle.lat,
                                     particle.lon]

    if particle.cd < 0.5:

        particle.uc = fieldset.cnormx_rho[time,
                                          particle.depth,
                                          particle.lat,
                                          particle.lon]
        particle.vc = fieldset.cnormy_rho[time,
                                          particle.depth,
                                          particle.lat,
                                          particle.lon]

        particle.uc *= (particle.cd - 0.5)**2
        particle.vc *= (particle.cd - 0.5)**2

        particle.lon += 1*particle.uc*particle.dt
        particle.lat += 1*particle.vc*particle.dt

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    #  (unlikely but possible at Antarctica)
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
                             lon  = pos0['lon'],
                             lat  = pos0['lat'],
                             time = pos0['time'])
print(str(len(pos0['lon'])) + ' particles released!')

traj = pset.ParticleFile(name=fh['traj'],
                         outputdt=out_dt)

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(antibeach) +
           pset.Kernel(beach) +
           pset.Kernel(periodicBC))

pset.execute(kernels,
             runtime=sim_T,
             dt = timedelta(minutes=-5),
             recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
             output_file=traj)

traj.export()

##############################################################################
# VISUALISE IF REQUESTED                                                     #
##############################################################################


viz_lim    = {'lonW': 46,
              'lonE': 47,
              'latS': -10,
              'latN': -9}

if debug:
    if not update_mask:
        masks = mdm.cmems_globproc(fh['ocean'],
                                   fh['grid'],
                                   add_seychelles=True)

    lat_psi = masks['lat_psi']
    lon_psi = masks['lon_psi']
    lat_rho = masks['lat_rho']
    lon_rho = masks['lon_rho']

    lsm_psi     = masks['lsm_psi']
    lsm_rho     = masks['lsm_rho']
    coast_psi   = masks['coast_psi']

    cnormx_rho  = masks['cnormx_rho']
    cnormy_rho  = masks['cnormy_rho']

    # Calculate grid indices for graphing
    jmin_psi = np.searchsorted(lon_psi, viz_lim['lonW']) - 1
    if jmin_psi < 0:
        jmin_psi = 0
    jmin_rho = jmin_psi
    jmax_psi = np.searchsorted(lon_psi, viz_lim['lonE'])
    jmax_rho = jmax_psi + 1

    imin_psi = np.searchsorted(lat_psi, viz_lim['latS']) - 1
    imin_rho = imin_psi
    imax_psi = np.searchsorted(lat_psi, viz_lim['latN'])
    imax_rho = imax_psi + 1

    disp_lon_rho = lon_rho[jmin_rho:jmax_rho]
    disp_lat_rho = lat_rho[imin_rho:imax_rho]

    disp_lon_psi = lon_psi[jmin_psi:jmax_psi]
    disp_lat_psi = lat_psi[imin_psi:imax_psi]

    disp_lsm_psi   = lsm_psi[imin_psi:imax_psi, jmin_psi:jmax_psi]
    disp_lsm_rho   = lsm_rho[imin_rho:imax_rho, jmin_rho:jmax_rho]

    disp_coast_psi = coast_psi[imin_psi:imax_psi, jmin_psi:jmax_psi]

    with Dataset(fh['ocean'], mode='r') as nc:
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

    ax.set_xlim(viz_lim['lonW'], viz_lim['lonE'])
    ax.set_ylim(viz_lim['latS'], viz_lim['latN'])

    # Plot the rho grid
    for i in range(len(disp_lat_rho)):
        ax.plot([viz_lim['lonW'], viz_lim['lonE']],
                [disp_lat_rho[i], disp_lat_rho[i]],
                'k--', linewidth=0.5)

    for j in range(len(disp_lon_rho)):
        ax.plot([disp_lon_rho[j], disp_lon_rho[j]],
                [viz_lim['lonW'], viz_lim['lonE']],
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

    # Plot the velocity field and BCs
    ax.quiver(disp_lon_rho, disp_lat_rho, disp_u_rho, disp_v_rho)
    ax.quiver(disp_lon_rho, disp_lat_rho, cnormx, cnormy, units='inches', scale=3, color='w')

    # Load the trajectories
    with Dataset(fh['traj'], mode='r') as nc:
        plat  = nc.variables['lat'][:]
        plon  = nc.variables['lon'][:]

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
    plt.savefig(dirs['script'] + '/test.png', dpi=300)
