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

# START YEAR
# y_in = int(argv[1])

# # # PARTITIONS
# try:
#     tot_part = int(argv[2])
#     part = int(argv[3])
# except:
#     tot_part = 1
#     part = 0

y_in = 2019
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
         'windage'           : False,         # Toggle to use windage
         'fw'                : 0.0,           # Windage fraction
         'max_age'           : 10,            # Max age (years). 0 == inf.

         # Runtime parameters
         'Yend'              : y_in,                 # Last year of simulation
         'Mend'              : 12   ,                    # Last month
         'Dend'              : 5   ,                    # Last day (00:00, start)
         'dt_RK4'            : timedelta(minutes=30),  # RK4 time-step

         # Output parameters
         'dt_out'            : timedelta(minutes=30),    # Output frequency
         'fn_out'            : str(y_in) + '_' + str(part) + '_SeyBwd.nc',  # Output filename

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
    particles['loc_array'] = mdm.release_loc(param, fh)
else:
    if param['line_rel']:
        particles['loc_array'] = {}
        particles['loc_array']['lon0'] = 46.65
        particles['loc_array']['lon1'] = 46.65
        particles['loc_array']['lat0'] = -10.0
        particles['loc_array']['lat1'] = -9.0

        particles['loc_array']['ll'] = [np.linspace(particles['loc_array']['lon0'],
                                                    particles['loc_array']['lon1'],
                                                    num=20),
                                        np.linspace(particles['loc_array']['lat0'],
                                                    particles['loc_array']['lat1'],
                                                    num=20),]


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

interp_method = {'U' : 'freeslip',
                 'V' : 'freeslip'}

fieldset_ocean = FieldSet.from_netcdf(filenames, variables, dimensions,
                                      interp_method=interp_method)

if param['stokes']:
    # WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
    filenames = fh['wave']

    variables = {'U': 'VSDX',
                 'V': 'VSDY'}

    dimensions = {'U': {'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'time'},
                  'V': {'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'time'}}

    interp_method = {'U' : 'freeslip',
                     'V' : 'freeslip'}

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

# Source cell ID (on psi grid, nearest)
source_id_psi  = Field.from_netcdf(fh['grid'],
                                   variable='source_id_psi',
                                   dimensions={'lon': 'lon_psi',
                                               'lat': 'lat_psi'},
                                   interp_method='nearest',
                                   allow_time_extrapolation=True)

# Sink cell ID (on psi grid, nearest)
sink_id_psi  = Field.from_netcdf(fh['grid'],
                                 variable='sink_id_psi',
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
fieldset.add_field(source_id_psi)
fieldset.add_field(sink_id_psi)
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
# PARTICLE CLASS #############################################################
##############################################################################

class debris(JITParticle):

    ##########################################################################
    # TEMPORARY VARIABLES FOR TRACKING PARTICLE POSITION/STATUS ##############
    ##########################################################################

    # ISO code of current cell (>0 if in any coastal cell)
    iso = Variable('iso',
                   dtype=np.int32,
                   initial=0,
                   to_write=True)

    # Sink ID of current cell (>0 if in specific sink cell)
    sink_id = Variable('sink_id',
                   dtype=np.int32,
                   initial=0,
                   to_write=True)

    # Source cell ID (specifically identifying source cell)
    source_id = Variable('source_id',
                         dtype=np.int32,
                         initial=fieldset.source_id_psi,
                         to_write='once')

    # Time at sea (Total time since release)
    ot  = Variable('ot',
                   dtype=np.int32,
                   initial=0,
                   to_write=False)

    # Time at coast (Time time spent in coastal cells)
    ct = Variable('ct',
                  dtype=np.int32,
                  initial=0,
                  to_write=False)

    ##########################################################################
    # ANTIBEACHING VARIABLES #################################################
    ##########################################################################

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

    ##########################################################################
    # TEMPORARY VARIABLES FOR TRACKING BEACHING AT SPECIFIED SINK SITES ######
    ##########################################################################

    # Sink status (memory of time in current sink cell - in time-steps)
    actual_sink_status = Variable('actual_sink_status',
                                  dtype=np.int32,
                                  initial=0,
                                  to_write=False)

    # Sink status (memory of ID of current sink cell)
    actual_sink_id = Variable('actual_sink_id',
                              dtype=np.int16,
                              initial=0,
                              to_write=False)

    # Sink t0 (memory of time when current sink cell was reached - in time-steps)
    actual_sink_t0 = Variable('actual_sink_t0',
                              dtype=np.int16,
                              initial=0,
                              to_write=False)

    # Sink ct (memory of time spent in coastal cells when arriving at current sink cell - in time-steps)
    actual_sink_ct = Variable('actual_sink_ct',
                              dtype=np.int32,
                              initial=0,
                              to_write=False)

    ##########################################################################
    # RECORD OF ALL EVENTS ###################################################
    ##########################################################################

    # Number of events
    e_num = Variable('e_num',
                     dtype=np.int16,
                     initial=0,
                     to_write=True)

    # Event 0
    e0 = Variable('e0',
                  dtype=np.int64,
                  initial=0,
                  to_write=True)

    # Event 0
    e1 = Variable('e1',
                  dtype=np.int64,
                  initial=0,
                  to_write=True)

    # Event 0
    e2 = Variable('e2',
                  dtype=np.int64,
                  initial=0,
                  to_write=True)

    # Event 0
    e3 = Variable('e3',
                  dtype=np.int64,
                  initial=0,
                  to_write=True)

    # Event 0
    e4 = Variable('e4',
                  dtype=np.int64,
                  initial=0,
                  to_write=True)

    ##########################################################################
    # TEMPORARY VARIABLES FOR TESTING ########################################
    ##########################################################################

    # Particle mass
    mass = Variable('mass',
                    dtype=np.float64,
                    initial=1.,
                    to_write=True)

    # Particle mass lost to event loc 0
    m0 = Variable('m0',
                  dtype=np.float64,
                  initial=0.,
                  to_write=True)

    # Particle mass lost to event loc 1
    m1 = Variable('m1',
                  dtype=np.float64,
                  initial=0.,
                  to_write=True)

    # Particle mass lost to event loc 2
    m2 = Variable('m2',
                  dtype=np.float64,
                  initial=0.,
                  to_write=True)

    # Particle mass lost to event loc 3
    m3 = Variable('m3',
                  dtype=np.float64,
                  initial=0.,
                  to_write=True)

    # Particle mass lost to event loc 4
    m4 = Variable('m4',
                  dtype=np.float64,
                  initial=0.,
                  to_write=True)

    # Test variables
    test1 = Variable('test1',
                    dtype=np.float64,
                    initial=0.,
                    to_write=True)

    test2 = Variable('test2',
                    dtype=np.float64,
                    initial=0.,
                    to_write=True)

    test3 = Variable('test3',
                    dtype=np.float64,
                    initial=0.,
                    to_write=True)


##############################################################################
# KERNELS ####################################################################
##############################################################################

def testing_mass(particle, fieldset, time):
    # Testing kernel to ensure that post-run calculations for mass fluxes are correct
    ls = 3.17e-9 # sinking
    lb = 5.79e-7 # beaching

    # Subtract mass due to losses from sinking
    particle.mass -= particle.mass*particle.dt*ls

    # Subtract mass due to losses from beaching
    if particle.sink_id > 0:
        if particle.e_num == 0:
            particle.m0 += particle.mass*particle.dt*lb
            particle.mass -= particle.mass*particle.dt*lb
        elif particle.e_num == 1:
            particle.m1 += particle.mass*particle.dt*lb
            particle.mass -= particle.mass*particle.dt*lb
        elif particle.e_num == 2:
            particle.m2 += particle.mass*particle.dt*lb
            particle.mass -= particle.mass*particle.dt*lb
        elif particle.e_num == 3:
            particle.m3 += particle.mass*particle.dt*lb
            particle.mass -= particle.mass*particle.dt*lb
        elif particle.e_num == 4:
            particle.m4 += particle.mass*particle.dt*lb
            particle.mass -= particle.mass*particle.dt*lb
    elif particle.iso > 0:
        particle.mass -= particle.mass*particle.dt*lb

    # if particle.sink_id > 0:
    #     if particle.e_num == 0:
    #         particle.m0 += particle.mass*(1/100)
    #         particle.mass -= particle.mass*(1/100)
    #     elif particle.e_num == 1:
    #         particle.m1 += particle.mass*(1/100)
    #         particle.mass -= particle.mass*(1/100)
    #     elif particle.e_num == 2:
    #         particle.m2 += particle.mass*(1/100)
    #         particle.mass -= particle.mass*(1/100)
    #     elif particle.e_num == 3:
    #         particle.m3 += particle.mass*(1/100)
    #         particle.mass -= particle.mass*(1/100)
    #     elif particle.e_num == 4:
    #         particle.m4 += particle.mass*(1/100)
    #         particle.mass -= particle.mass*(1/100)
    # elif particle.iso > 0:
    #     particle.mass -= particle.mass*(1/100)


def time_at_coast(particle, fieldset, time):
    # Keep track of the amount of time spent within a coastal cell
    particle.iso = fieldset.iso_psi[particle]

    # ct is updated whenever the particle is in any coastal cell
    if particle.iso > 0:
        particle.ct += particle.dt


def time_at_sea(particle, fieldset, time):
    # Keep track of the amount of time spent at sea
    particle.ot += particle.dt

    # Delete particle if the maximum age is exceeded
    if particle.ot > fieldset.max_age:
        particle.delete()


def event(particle, fieldset, time):
    # Controller for managing particle events
    particle.sink_id = fieldset.sink_id_psi[particle]

    save_event = False
    new_event = False

    # Trigger event if particle is within selected sink site
    if particle.sink_id > 0:

        # Check if event has already been triggered
        if particle.actual_sink_status > 0:

            # Check if we are in the same sink cell as the current event
            if particle.sink_id == particle.actual_sink_id:

                # If contiguous event, just add time
                particle.actual_sink_status += particle.dt

                # But also check that the particle isn't about to expire (save if so)
                # Otherwise particles hanging around coastal regions forever won't get saved
                if particle.ot > fieldset.max_age - 3600:
                    save_event = True

            else:

                # Otherwise, we need to save the old event and create a new event
                save_event = True
                new_event = True

        else:

            # If event has not been triggered, create a new event
            new_event = True

    else:

        # Otherwise, check if ongoing event has just ended
        if particle.actual_sink_status > 0:

            save_event = True

    if save_event:
        # Save actual values
        # Unfortunately, due to the limited functions allowed in parcels, this
        # required a horrendous if-else chain

        if particle.e_num == 0:
            particle.e0 += (particle.actual_sink_t0/particle.dt)
            particle.e0 += (particle.actual_sink_ct/particle.dt)*2**20
            particle.e0 += (particle.actual_sink_status/particle.dt)*2**40
            particle.e0 += (particle.actual_sink_id)*2**52

        elif particle.e_num == 1:
            particle.e1 += (particle.actual_sink_t0/particle.dt)
            particle.e1 += (particle.actual_sink_ct/particle.dt)*2**20
            particle.e1 += (particle.actual_sink_status/particle.dt)*2**40
            particle.e1 += (particle.actual_sink_id)*2**52

        elif particle.e_num == 2:
            particle.e2 += (particle.actual_sink_t0/particle.dt)
            particle.e2 += (particle.actual_sink_ct/particle.dt)*2**20
            particle.e2 += (particle.actual_sink_status/particle.dt)*2**40
            particle.e2 += (particle.actual_sink_id)*2**52

        elif particle.e_num == 3:
            particle.e3 += (particle.actual_sink_t0/particle.dt)
            particle.e3 += (particle.actual_sink_ct/particle.dt)*2**20
            particle.e3 += (particle.actual_sink_status/particle.dt)*2**40
            particle.e3 += (particle.actual_sink_id)*2**52

        elif particle.e_num == 4:
            particle.e4 += (particle.actual_sink_t0/particle.dt)
            particle.e4 += (particle.actual_sink_ct/particle.dt)*2**20
            particle.e4 += (particle.actual_sink_status/particle.dt)*2**40
            particle.e4 += (particle.actual_sink_id)*2**52

            particle.delete() # Delete particle, since no more sinks can be saved

        else:
            particle.delete()

        # Then reset actual values to zero
        particle.actual_sink_t0 = 0
        particle.actual_sink_ct = 0
        particle.actual_sink_status = 0
        particle.actual_sink_id = 0

        # Add to event number counter
        particle.e_num += 1

    if new_event:
        # Add status to actual values
        particle.actual_sink_status += particle.dt
        particle.actual_sink_t0 = particle.ot
        particle.actual_sink_ct = particle.ct
        particle.actual_sink_id = particle.sink_id


def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if an error occurs
    particle.delete()


def antibeach(particle, fieldset, time):
    #  Kernel to repel particles from the coast
    particle.cd = fieldset.cdist_rho[particle]

    if particle.cd < 0.2:

        particle.uc = fieldset.cnormx_rho[particle]
        particle.vc = fieldset.cnormy_rho[particle]

        particle.uc *= 5*(particle.cd - 0.2)**2
        particle.vc *= 5*(particle.cd - 0.2)**2

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
               pset.Kernel(time_at_coast) +
               pset.Kernel(event) +
               pset.Kernel(testing_mass) +
               pset.Kernel(antibeach) +
               pset.Kernel(time_at_sea) +
               pset.Kernel(periodicBC))
else:
    kernels = (pset.Kernel(AdvectionRK4) +
               pset.Kernel(time_at_coast) +
               pset.Kernel(event) +
               pset.Kernel(testing_mass) +
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
    (lon_min, lon_max, lat_min, lat_max) = (45.7, 46.7, -10.0, -9.0)

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
        # ax.scatter(plon[particle, :], plat[particle, :])

    # Save
    plt.savefig(dirs['script'] + '/' + 'test_fig.png', dpi=300)
