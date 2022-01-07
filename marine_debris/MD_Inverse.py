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
from sys import argv

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

# PARTITIONS & STARTING YEAR
try:
    y_in = int(argv[1])
    m_in = int(argv[2])
    tot_part = int(argv[3])
    part = int(argv[4])
except:
    y_in = 2019
    m_in = 9
    tot_part = 1
    part = 0

# PARAMETERS
param = {# Release timing
         'Ymin'              : y_in,          # First release year
         'Ymax'              : y_in,          # Last release year
         'Mmin'              : 12   ,          # First release month
         'Mmax'              : 12  ,          # Last release month
         'RPM'               : 1   ,          # Releases per month
         'mode'              :'START',        # Release at END or START

         # Release location
         'id'                : [690],         # ISO IDs of release countries
         'pn'                : 4096,          # Particles to release per cell

         # Simulation parameters
         'stokes'            : True,          # Toggle to use Stokes drift
         'windage'           : True,         # Toggle to use windage
         'fw'                : 1.0,           # Windage fraction
         'max_age'           : 0,             # Max age (years). 0 == inf.

         # Runtime parameters
         'Yend'              : 2019,                 # Last year of simulation
         'Mend'              : 11   ,                    # Last month
         'Dend'              : 1   ,                    # Last day (00:00, start)
         'dt_RK4'            : timedelta(minutes=-30),  # RK4 time-step

         # Output parameters
         'dt_out'            : timedelta(hours=1),    # Output frequency
         'fn_out'            : str(y_in) + '_' + str(part) + '_SeyBwdWind.nc',  # Output filename

         # Partitioning
         'total_partitions'  : tot_part,
         'partition'         : part,

         # Other parameters
         'update'            : True,                   # Update grid files
         'plastic'           : True,                   # Write plastic data
         'add_sey'           : True,                   # Add extra Sey islands
         'p_param'           : {'l'  : 50.,            # Plastic length scale
                                'cr' : 0.15},          # Fraction entering sea

         'test'              : True,                  # Activate test mode
         'line_rel'          : True,}                 # Release particles in line

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

# MODIFICATION TO PREVENT CRASHING IF END_YR=1993
if param['Yend'] <= 1993:
    param['Yend'] = 1993
    param['Mend'] = 1
    param['Dend'] = 2

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

# Calculate the locations, ids, procs for particle releases
if not param['test']:
    particles['loc_array'] = mdm.release_loc_sey(param, fh)
else:
    if param['line_rel']:
        particles['loc_array'] = {}
        particles['loc_array']['lon0'] = 49.22
        particles['loc_array']['lon1'] = 49.22
        particles['loc_array']['lat0'] = -12.3
        particles['loc_array']['lat1'] = -11.8

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
particles['pos'] = mdm.add_times(particles, param)

##############################################################################
# SET UP FIELDSETS                                                           #
##############################################################################

# OCEAN (CMEMS GLORYS12V1)
filenames = fh['ocean']

variables = {'U': 'uo',
              'V': 'vo'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

interp_method = {'U' : 'linear',
                 'V' : 'linear'}

fieldset_ocean = FieldSet.from_netcdf(filenames, variables, dimensions,
                                      interp_method=interp_method)

if param['stokes']:
    # WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
    filenames = fh['wave']

    variables = {'U': 'VSDX',
                 'V': 'VSDY'}

    dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                  'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

    interp_method = {'U' : 'linear',
                     'V' : 'linear'}

    fieldset_wave = FieldSet.from_netcdf(filenames, variables, dimensions,
                                         interp_method=interp_method)

    fieldset = FieldSet(U=fieldset_ocean.U+fieldset_wave.U,
                        V=fieldset_ocean.V+fieldset_wave.V)
else:
    fieldset = fieldset_ocean

# ADD ADDITIONAL FIELDS
# Country identifier grid (on psi grid, nearest)
iso_psi  = Field.from_netcdf(fh['grid'],
                             variable='iso_psi',
                             dimensions={'lon': 'lon_psi',
                                         'lat': 'lat_psi'},
                             interp_method='nearest',
                             allow_time_extrapolation=True)

# Distance from nearest land point
cdist   = Field.from_netcdf(fh['grid'],
                            variable='cdist_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            allow_time_extrapolation=True)

# Normal away from coast (x component)
cnormx  = Field.from_netcdf(fh['grid'],
                            variable='cnormx_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            mesh='spherical',
                            allow_time_extrapolation=True)

# Normal away from coast (y component)
cnormy  = Field.from_netcdf(fh['grid'],
                            variable='cnormy_rho',
                            dimensions={'lon': 'lon_rho',
                                        'lat': 'lat_rho'},
                            interp_method='linear',
                            mesh='spherical',
                            allow_time_extrapolation=True)

fieldset.add_field(iso_psi)
fieldset.add_field(cdist)
fieldset.add_field(cnormx)
fieldset.add_field(cnormy)

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
    # Coast mask (if particle is in a coastal cell)
    cm = Variable('cm',
                  dtype=np.float64,
                  initial=0,
                  to_write=False)

    # Time at sea (ocean time)
    ot  = Variable('ot',
                   dtype=np.int32,
                   initial=0,
                   to_write=False)

    # Time at coast (coast time)
    ct = Variable('ct',
                  dtype=np.int32,
                  initial=0,
                  to_write=True)

    # Particle distance from land
    cd  = Variable('cd',
                   dtype=np.float32,
                   initial=0.,
                   to_write=False)

    # Velocity away from coast (to prevent beaching) - x component
    uc  = Variable('uc',
                   dtype=np.float32,
                   initial=0.,
                   to_write=False)

    # Velocity away from coast (to prevent beaching) - y component
    vc  = Variable('vc',
                   dtype=np.float32,
                   initial=0.,
                   to_write=False)

def coast(particle, fieldset, time):
    # Keep track of the amount of time spent within a coastal cell
    particle.cm = fieldset.iso_psi[particle]

    if particle.cm > 0:
        particle.ct -= particle.dt

def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if an error occurs
    particle.delete()

def oldParticle(particle, fieldset, time):
    # Remove particles older than given age
    particle.ot -= particle.dt

    if particle.ot > fieldset.max_age:
        particle.delete()

def antibeach(particle, fieldset, time):
    #  Kernel to repel particles from the coast
    particle.cd = fieldset.cdist_rho[particle]

    if particle.cd < 0.5:

        particle.uc = fieldset.cnormx_rho[particle]
        particle.vc = fieldset.cnormy_rho[particle]

        if particle.cd < 0.1:
            particle.uc *= -1*(particle.cd - 0.5)**2 -75*(particle.cd - 0.1)**2
            particle.vc *= -1*(particle.cd - 0.5)**2 -75*(particle.cd - 0.1)**2
        else:
            particle.uc *= -1*(particle.cd - 0.5)**2
            particle.vc *= -1*(particle.cd - 0.5)**2

        particle.lon += particle.uc*particle.dt
        particle.lat += particle.vc*particle.dt

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
                             time = particles['pos']['time'])

print(str(len(particles['pos']['time'])) + ' particles released!')

traj = pset.ParticleFile(name=fh['traj'],
                         outputdt=param['dt_out'])

if param['max_age']:
    kernels = (pset.Kernel(AdvectionRK4) +
               pset.Kernel(coast) +
               pset.Kernel(antibeach) +
               pset.Kernel(oldParticle) +
               pset.Kernel(periodicBC))
else:
    kernels = (pset.Kernel(AdvectionRK4) +
               pset.Kernel(coast) +
               pset.Kernel(antibeach) +
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
    (lon_min, lon_max, lat_min, lat_max) = (49, 50, -12.4, -11.4)

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
