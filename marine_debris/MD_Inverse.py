#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles arriving at a country backward in time using the
CMEMS GLORYS12V1 reanalysis (current and Stokes drift)
@author: Noam Vogt-Vincent
"""

import os
import MD_Methods as mdm
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
from parcels import (Field, FieldSet, ParticleSet, JITParticle, AdvectionRK4,
                     ErrorCode, Geographic, GeographicPolar, Variable,
                     ParcelsRandom)
from netCDF4 import Dataset, num2date
from datetime import timedelta, datetime
from glob import glob
from mpi4py import MPI

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
         'pn'                :  2500,           # Particles to release per cell

         # Simulation parameters
         'stokes'            : True,          # Toggle to use Stokes drift
         'windage'           : False,         # Toggle to use windage
         'fw'                : 0.0,           # Windage fraction
         'max_age'           : 0,             # Max age (years). 0 == inf.

         # Runtime parameters
         'Yend'              : 2019,                   # Last year of simulation
         'Mend'              : 11  ,                   # Last month
         'Dend'              : 25   ,                  # Last day (00:00, start)
         'dt_RK4'            : timedelta(minutes=-30), # RK4 time-step

         # Output parameters
         'dt_out'            : timedelta(hours=1),     # Output frequency
         'fn_out'            : '1M_f8_test.nc',        # Output filename

         # Other parameters
         'update'            : True,                   # Update grid files
         'plastic'           : True,                   # Write plastic data
         'add_sey'           : True,                   # Add extra Sey islands
         'p_param'           : {'l'  : 50.,            # Plastic length scale
                                'cr' : 0.15},          # Fraction entering sea

         'test'              : False,                   # Activate test mode
         'line_rel'          : False,}                 # Release particles in line

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'model': os.path.dirname(os.path.realpath(__file__)) + '/MODEL_DATA_NEW/',
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

grid = mdm.gridgen(fh, dirs, param, plastic=True, add_seychelles=True)

# Calculate the times for particle releases
particles = {'time_array' : mdm.release_time(param, mode=param['mode'])}

# Calculate number of processes
param['nproc'] = MPI.COMM_WORLD.Get_size()

# Calculate the locations, ids, procs for particle releases
if not param['test']:
    particles['loc_array'] = mdm.release_loc(param, fh)
else:
    if param['line_rel']:
        particles['loc_array'] = {}
        particles['loc_array']['lon0'] = 49.35
        particles['loc_array']['lon1'] = 49.35
        particles['loc_array']['lat0'] = -12.05
        particles['loc_array']['lat1'] = -12.2

        particles['loc_array']['ll'] = [np.linspace(particles['loc_array']['lon0'],
                                                    particles['loc_array']['lon1'],
                                                    num=10),
                                        np.linspace(particles['loc_array']['lat0'],
                                                    particles['loc_array']['lat1'],
                                                    num=10),]


        particles['loc_array']['lon'] = particles['loc_array']['ll'][0].flatten()
        particles['loc_array']['lat'] = particles['loc_array']['ll'][1].flatten()

        particles['loc_array']['iso'] = np.zeros_like(particles['loc_array']['lon'])
        particles['loc_array']['id'] = np.zeros_like(particles['loc_array']['lon'])
    else:
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

interp_method = {'U' : 'freeslip',
                 'V' : 'freeslip'}

fieldset_ocean = FieldSet.from_netcdf(filenames, variables, dimensions,
                                      interp_method=interp_method,
                                      chunksize=False)

# WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
filenames = fh['wave']

variables = {'U': 'VSDX',
             'V': 'VSDY'}

dimensions = {'U': {'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'time'},
              'V': {'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'time'}}

interp_method = {'U' : 'freeslip',
                 'V' : 'freeslip'}

fieldset_wave = FieldSet.from_netcdf(filenames, variables, dimensions,
                                     interp_method=interp_method,
                                     chunksize=False)

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

fieldset.add_field(id_psi)
fieldset.add_field(lsm_rho)

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

    # Time at sea (ocean time)
    ot  = Variable('ot',
                   dtype=np.int32,
                   initial=0,
                   to_write=False)


def beach(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it is beached
    particle.lsm = fieldset.lsm_rho[particle]
    # particle.lsm = ParcelsRandom.random()

    if particle.lsm == 1.0:
        particle.delete()

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if it leaves the domain
    #  (possible in certain configurations)
    particle.delete()

def shiftParticle(particle, fieldset, time):
    #  Recovery kernel to shift a particle if it returns an interpolation error
    #  Shift particle by 1m
    particle.lon += 1e-5
    particle.lat += 1e-5

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
                             lonlatdepth_dtype=np.float64,
                             lon  = particles['pos']['lon'],
                             lat  = particles['pos']['lat'],
                             time = particles['pos']['time'],
                             partitions = particles['pos']['partitions']
                             )
print(str(len(particles['pos']['time'])) + ' particles released!')

traj = pset.ParticleFile(name=fh['traj'],
                         outputdt=param['dt_out'])

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(beach) +
           pset.Kernel(periodicBC))

pset.execute(kernels,
             endtime=param['endtime'],
             dt=param['dt_RK4'],
             recovery={ErrorCode.ErrorOutOfBounds: deleteParticle,
                       ErrorCode.ErrorInterpolation: deleteParticle},
             output_file=traj)

traj.export()

##############################################################################
# PLOT TEST CASE                                                             #
##############################################################################

if param['test']:
    # Set display region
    (lon_min, lon_max, lat_min, lat_max) = (46.0, 56, -12, -2)

    # Import grids
    with Dataset(fh['grid'], mode='r') as nc:
        lon_psi = grid['lon_psi']
        lat_psi = grid['lat_psi']
        lon_rho = grid['lon_rho']
        lat_rho = grid['lat_rho']

        lsm_psi = grid['lsm_psi']
        lsm_rho = grid['lsm_rho']
        coast_psi = grid['coast_psi']

        cnormx_rho = grid['cnormx_rho']
        cnormy_rho = grid['cnormy_rho']

        cdist_rho = grid['cdist_rho']

    jmin_psi = np.searchsorted(lon_psi, lon_min) - 1
    if jmin_psi < 0:
        jmin_psi = 0
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
    disp_cdist_rho = cdist_rho[imin_rho:imax_rho, jmin_rho:jmax_rho]

    with Dataset(fh['ocean'][-1], mode='r') as nc:
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
    ax.quiver(disp_lon_rho, disp_lat_rho, cnormx, cnormy, units='inches', scale=3, color='w')

    # Load the trajectories
    with Dataset(fh['traj'], mode='r') as nc:
        plat  = nc.variables['lat'][:]
        plon  = nc.variables['lon'][:]

    pnum = np.shape(plat)[0]
    pt   = np.shape(plat)[1]

    for particle in range(pnum):
        ax.plot(plon[particle, :], plat[particle, :], 'w-', linewidth=0.5)

    # Save
    plt.savefig(dirs['script'] + '/' + 'test_fig.png', dpi=300)