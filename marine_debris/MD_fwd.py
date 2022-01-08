#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tracks particles arriving at a country forward in time using the
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
                     DiffusionUniformKh)
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
         'Mmin'              : m_in,          # First release month
         'Mmax'              : m_in,           # Last release month
         'mode'              :'START',        # Release at END or START
         'RPM'               : 1,             # Releases per month

         # Seeding strategy
         'threshold'         : 1e2,           # Minimum plastic threshold (kg)
         'log_mult'          : 12,            # Log-multiplier

         # Sink locations
         'id'                : [690],         # ISO IDs of sink countries

         # Simulation parameters
         'stokes'            : True,          # Toggle to use Stokes drift
         'windage'           : False,         # Toggle to use windage
         'fw'                : 2.0,           # Windage fraction (0.5/1.0/2.0/3.0)
         'Kh'                : 10.,           # Horizontal diffusion coefficient (m2/s, 0 = off)
         'max_age'           : 10.,           # Max age (years). 0 == inf.

         # Runtime parameters
         'Yend'              : y_in+10,                # Last year of simulation
         'Mend'              : m_in   ,                # Last month
         'Dend'              : 2   ,                   # Last day (00:00, start)
         'dt_RK4'            : timedelta(minutes=60),  # RK4 time-step

         # Output parameters
         'fn_out'            : str(y_in) + '_' + str(m_in) + '_' + str(part) + '_Fwd.nc',  # Output filename

         # Partitioning
         'total_partitions'  : tot_part,
         'partition'         : part,

         # Plastic parameters
         'plastic'           : True,                   # Write plastic source data
         'add_sey'           : True,                   # Add extra Sey islands
         'plot_input'        : False,                  # Plot plastic input

         'p_param'           : {'l'  : 25.,            # Plastic length scale
                                'cr' : 0.25},          # Fraction entering sea

         # Testing parameters
         'test'              : False,                  # Activate test mode
         'line_rel'          : False,                  # Release particles in line
         'dt_out'            : timedelta(minutes=60),} # Output frequency (testing only)

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'model': os.path.dirname(os.path.realpath(__file__)) + '/MODEL_DATA/',
        'grid': os.path.dirname(os.path.realpath(__file__)) + '/GRID_DATA/',
        'plastic': os.path.dirname(os.path.realpath(__file__)) + '/PLASTIC_DATA/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/FIGURES/',
        'traj': os.path.dirname(os.path.realpath(__file__)) + '/TRAJ/'}

# FILE HANDLES
fh = {'ocean':   sorted(glob(dirs['model'] + 'OCEAN_*.nc')),
      'wave':    sorted(glob(dirs['model'] + 'WAVE_*.nc')),
      'grid':    dirs['grid'] + 'griddata.nc',
      'clist':   dirs['plastic'] + 'country_list.in',
      'fig':     dirs['fig'] + 'plastic_input',
      'traj':    dirs['traj'] + param['fn_out'],}

# MODIFICATION TO PREVENT CRASHING IF END_YR=1993
if param['Yend'] <= 1993:
    param['Yend'] = 1993
    param['Mend'] = 1
    param['Dend'] = 2

if param['Ymin'] == 1993:
    if param['Mmin'] == 1:
        param['delay_start'] = True

if param['max_age'] == 0:
    param['max_age'] = 1e20

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
    param['iso_list'] = np.loadtxt(fh['clist'],
                                   delimiter=',',
                                   usecols=0,
                                   skiprows=1)
    particles['loc_array'] = mdm.release_loc_land(param, fh)
else:
    if param['line_rel']:
        particles['loc_array'] = {}
        particles['loc_array']['lon0'] = 80.084
        particles['loc_array']['lon1'] = 80.084
        particles['loc_array']['lat0'] = 5.8
        particles['loc_array']['lat1'] = 6.57

        particles['loc_array']['ll'] = [np.linspace(particles['loc_array']['lon0'],
                                                    particles['loc_array']['lon1'],
                                                    num=200),
                                        np.linspace(particles['loc_array']['lat0'],
                                                    particles['loc_array']['lat1'],
                                                    num=200),]


        particles['loc_array']['lon'] = particles['loc_array']['ll'][0].flatten()
        particles['loc_array']['lat'] = particles['loc_array']['ll'][1].flatten()

        particles['loc_array']['iso'] = np.zeros_like(particles['loc_array']['lon'])
        particles['loc_array']['id'] = np.zeros_like(particles['loc_array']['lon'])
        particles['loc_array']['cp0'] = np.zeros_like(particles['loc_array']['lon'])
        particles['loc_array']['rp0'] = np.zeros_like(particles['loc_array']['lon'])
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

if param['windage']:
    # WINDAGE (FROM ERA5) + WAVE (STOKES FROM WAVERYS W/ GLORYS12V1)
    if param['fw'] not in [0.5, 1.0, 2.0, 3.0]:
        raise NotImplementedError('Windage fraction not available!')

    wind_fh = 'WINDWAVE' + format(int(param['fw']*10), '04') + '*'
    vsuffix = '_windwave' + format(int(param['fw']*10), '02')
    fh['wind'] = sorted(glob(dirs['model'] + wind_fh))

    filenames = fh['wind']

    variables = {'U': 'u' + vsuffix,
                 'V': 'v' + vsuffix}

    dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                  'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}

    interp_method = {'U' : 'linear',
                     'V' : 'linear'}

    fieldset_windwave = FieldSet.from_netcdf(filenames, variables, dimensions,
                                             interp_method=interp_method)
elif param['stokes']:
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



if param['windage']:
    fieldset = FieldSet(U=fieldset_ocean.U+fieldset_windwave.U,
                        V=fieldset_ocean.V+fieldset_windwave.V)
elif param['stokes']:
    fieldset = FieldSet(U=fieldset_ocean.U+fieldset_wave.U,
                        V=fieldset_ocean.V+fieldset_wave.V)
else:
    fieldset = fieldset_ocean


# ADD ADDITIONAL FIELDS
# Country identifier grid (on psi grid, nearest)
iso_psi_all  = Field.from_netcdf(fh['grid'],
                                 variable='iso_psi_all',
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

fieldset.add_field(iso_psi_all)
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

# ADD DIFFUSION
if param['Kh']:
    fieldset.add_constant_field('Kh_zonal', param['Kh'], mesh='spherical')
    fieldset.add_constant_field('Kh_meridional', param['Kh'], mesh='spherical')

# ADD MAXIMUM PARTICLE AGE (IF LIMITED AGE)
fieldset.add_constant('max_age', param['max_age']*3600*24*365)


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
                   to_write=False)

    # Sink ID of current cell (>0 if in specific sink cell)
    sink_id = Variable('sink_id',
                   dtype=np.int16,
                   initial=0,
                   to_write=False)

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
    # PROVENANCE IDENTIFIERS #################################################
    ##########################################################################

    # Source cell ID (specifically identifying source cell)
    source_id = Variable('source_id',
                         dtype=np.int32,
                         to_write=True)

    # Source cell ID - specifically identifying source cell
    source_cell = Variable('source_cell',
                           dtype=np.int32,
                           to_write=True)

    # Source cell ID (Initial mass of plastic from direct coastal input)
    cp0 = Variable('cp0',
                   dtype=np.float32,
                   to_write=True)

    # Source cell ID (Initial mass of plastic from riverine input)
    rp0 = Variable('rp0',
                   dtype=np.float32,
                   to_write=True)

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
                              dtype=np.int32,
                              initial=0,
                              to_write=False)

    # Sink t0 (memory of time when current sink cell was reached - in time-steps)
    actual_sink_t0 = Variable('actual_sink_t0',
                              dtype=np.int32,
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
    e_num = Variable('e_num', dtype=np.int16, initial=0, to_write=True)

    # Events
    e0 = Variable('e0', dtype=np.int64, initial=0, to_write=True)
    e1 = Variable('e1', dtype=np.int64, initial=0, to_write=True)
    e2 = Variable('e2', dtype=np.int64, initial=0, to_write=True)
    e3 = Variable('e3', dtype=np.int64, initial=0, to_write=True)
    e4 = Variable('e4', dtype=np.int64, initial=0, to_write=True)
    e5 = Variable('e5', dtype=np.int64, initial=0, to_write=True)
    e6 = Variable('e6', dtype=np.int64, initial=0, to_write=True)
    e7 = Variable('e7', dtype=np.int64, initial=0, to_write=True)
    e8 = Variable('e8', dtype=np.int64, initial=0, to_write=True)
    e9 = Variable('e9', dtype=np.int64, initial=0, to_write=True)
    e10 = Variable('e10', dtype=np.int64, initial=0, to_write=True)
    e11 = Variable('e11', dtype=np.int64, initial=0, to_write=True)
    e12 = Variable('e12', dtype=np.int64, initial=0, to_write=True)
    e13 = Variable('e13', dtype=np.int64, initial=0, to_write=True)
    e14 = Variable('e14', dtype=np.int64, initial=0, to_write=True)
    e15 = Variable('e15', dtype=np.int64, initial=0, to_write=True)
    e16 = Variable('e16', dtype=np.int64, initial=0, to_write=True)
    e17 = Variable('e17', dtype=np.int64, initial=0, to_write=True)
    e18 = Variable('e18', dtype=np.int64, initial=0, to_write=True)
    e19 = Variable('e19', dtype=np.int64, initial=0, to_write=True)

##############################################################################
# KERNELS ####################################################################
##############################################################################

# Controller for managing particle events
def event(particle, fieldset, time):

    # 1 Keep track of the amount of time spent at sea
    particle.ot += particle.dt

    # 2 Assess coastal status
    particle.iso = fieldset.iso_psi_all[particle]

    if particle.iso > 0:

        # If in coastal cell, keep track of time spent in coastal cell
        particle.ct += particle.dt

        # Only need to check sink_id if we know we are in a coastal cell
        particle.sink_id = fieldset.sink_id_psi[particle]

    else:
        particle.sink_id = 0

    # 3 Manage particle event if relevant
    save_event = False
    new_event = False

    # Trigger event if particle is within selected sink site
    if particle.sink_id > 0:

        # Check if event has already been triggered
        if particle.actual_sink_status > 0:

            # Check if we are in the same sink cell as the current event
            if particle.sink_id == particle.actual_sink_id:

                # If contiguous event, just add time
                particle.actual_sink_status += 1

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
        # required an horrendous if-else chain

        if particle.e_num == 0:
            particle.e0 += (particle.actual_sink_t0)
            particle.e0 += (particle.actual_sink_ct)*2**20
            particle.e0 += (particle.actual_sink_status)*2**40
            particle.e0 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 1:
            particle.e1 += (particle.actual_sink_t0)
            particle.e1 += (particle.actual_sink_ct)*2**20
            particle.e1 += (particle.actual_sink_status)*2**40
            particle.e1 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 2:
            particle.e2 += (particle.actual_sink_t0)
            particle.e2 += (particle.actual_sink_ct)*2**20
            particle.e2 += (particle.actual_sink_status)*2**40
            particle.e2 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 3:
            particle.e3 += (particle.actual_sink_t0)
            particle.e3 += (particle.actual_sink_ct)*2**20
            particle.e3 += (particle.actual_sink_status)*2**40
            particle.e3 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 4:
            particle.e4 += (particle.actual_sink_t0)
            particle.e4 += (particle.actual_sink_ct)*2**20
            particle.e4 += (particle.actual_sink_status)*2**40
            particle.e4 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 5:
            particle.e5 += (particle.actual_sink_t0)
            particle.e5 += (particle.actual_sink_ct)*2**20
            particle.e5 += (particle.actual_sink_status)*2**40
            particle.e5 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 6:
            particle.e6 += (particle.actual_sink_t0)
            particle.e6 += (particle.actual_sink_ct)*2**20
            particle.e6 += (particle.actual_sink_status)*2**40
            particle.e6 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 7:
            particle.e7 += (particle.actual_sink_t0)
            particle.e7 += (particle.actual_sink_ct)*2**20
            particle.e7 += (particle.actual_sink_status)*2**40
            particle.e7 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 8:
            particle.e8 += (particle.actual_sink_t0)
            particle.e8 += (particle.actual_sink_ct)*2**20
            particle.e8 += (particle.actual_sink_status)*2**40
            particle.e8 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 9:
            particle.e9 += (particle.actual_sink_t0)
            particle.e9 += (particle.actual_sink_ct)*2**20
            particle.e9 += (particle.actual_sink_status)*2**40
            particle.e9 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 10:
            particle.e10 += (particle.actual_sink_t0)
            particle.e10 += (particle.actual_sink_ct)*2**20
            particle.e10 += (particle.actual_sink_status)*2**40
            particle.e10 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 11:
            particle.e11 += (particle.actual_sink_t0)
            particle.e11 += (particle.actual_sink_ct)*2**20
            particle.e11 += (particle.actual_sink_status)*2**40
            particle.e11 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 12:
            particle.e12 += (particle.actual_sink_t0)
            particle.e12 += (particle.actual_sink_ct)*2**20
            particle.e12 += (particle.actual_sink_status)*2**40
            particle.e12 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 13:
            particle.e13 += (particle.actual_sink_t0)
            particle.e13 += (particle.actual_sink_ct)*2**20
            particle.e13 += (particle.actual_sink_status)*2**40
            particle.e13 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 14:
            particle.e14 += (particle.actual_sink_t0)
            particle.e14 += (particle.actual_sink_ct)*2**20
            particle.e14 += (particle.actual_sink_status)*2**40
            particle.e14 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 15:
            particle.e15 += (particle.actual_sink_t0)
            particle.e15 += (particle.actual_sink_ct)*2**20
            particle.e15 += (particle.actual_sink_status)*2**40
            particle.e15 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 16:
            particle.e16 += (particle.actual_sink_t0)
            particle.e16 += (particle.actual_sink_ct)*2**20
            particle.e16 += (particle.actual_sink_status)*2**40
            particle.e16 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 17:
            particle.e17 += (particle.actual_sink_t0)
            particle.e17 += (particle.actual_sink_ct)*2**20
            particle.e17 += (particle.actual_sink_status)*2**40
            particle.e17 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 18:
            particle.e18 += (particle.actual_sink_t0)
            particle.e18 += (particle.actual_sink_ct)*2**20
            particle.e18 += (particle.actual_sink_status)*2**40
            particle.e18 += (particle.actual_sink_id)*2**52
        elif particle.e_num == 19:
            particle.e19 += (particle.actual_sink_t0)
            particle.e19 += (particle.actual_sink_ct)*2**20
            particle.e19 += (particle.actual_sink_status)*2**40
            particle.e19 += (particle.actual_sink_id)*2**52

            particle.delete() # Delete particle, since no more sinks can be saved

        # Then reset actual values to zero
        particle.actual_sink_t0 = 0
        particle.actual_sink_ct = 0
        particle.actual_sink_status = 0
        particle.actual_sink_id = 0

        # Add to event number counter
        particle.e_num += 1

    if new_event:
        # Add status to actual (for current event) values
        # Timesteps at current sink
        particle.actual_sink_status = 1

        # Timesteps spent in the ocean overall (minus one, before this step)
        particle.actual_sink_t0 = (particle.ot/particle.dt) - 1

        # Timesteps spent in the coast overall (minus one, before this step)
        particle.actual_sink_ct = (particle.ct/particle.dt) - 1

        # ID of current sink
        particle.actual_sink_id = particle.sink_id

    # Finally, check if particle needs to be deleted
    if particle.ot > fieldset.max_age - 3600:

        # Only delete particles where at least 1 event has been recorded
        if particle.e_num > 0:
            particle.delete()


def antibeach(particle, fieldset, time):
    #  Kernel to repel particles from the coast
    particle.cd = fieldset.cdist_rho[particle]

    if particle.cd < 0.5:

        particle.uc = fieldset.cnormx_rho[particle]
        particle.vc = fieldset.cnormy_rho[particle]

        if particle.cd <= 0:
            particle.uc *= 3 # Rapid acceleration at 3m/s away to sea (exceeds all wind + ocean)
            particle.vc *= 3 # Rapid acceleration at 3m/s away to sea (exceeds all wind + ocean)
        elif particle.cd < 0.1:
            particle.uc *= 1*(particle.cd - 0.5)**2 +75*(particle.cd - 0.1)**2 # Will prevent all normal coastward velocities (< 1m/s) from beaching
            particle.vc *= 1*(particle.cd - 0.5)**2 +75*(particle.cd - 0.1)**2 # Will prevent all normal coastward velocities (< 1m/s) from beaching
        else:
            particle.uc *= 1*(particle.cd - 0.5)**2
            particle.vc *= 1*(particle.cd - 0.5)**2

        particle.lon += particle.uc*particle.dt
        particle.lat += particle.vc*particle.dt


def periodicBC(particle, fieldset, time):
    # Move the particle across the periodic boundary
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west


def deleteParticle(particle, fieldset, time):
    #  Recovery kernel to delete a particle if an error occurs
    particle.delete()

##############################################################################
# INITIALISE SIMULATION AND RUN                                              #
##############################################################################

pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=debris,
                             lonlatdepth_dtype=np.float64,
                             lon  = particles['pos']['lon'],
                             lat  = particles['pos']['lat'],
                             time = particles['pos']['time'],
                             rp0  = particles['pos']['rp0'],
                             cp0  = particles['pos']['cp0'],
                             source_id = particles['pos']['iso'],
                             source_cell = particles['pos']['id'])

print(str(len(particles['pos']['time'])) + ' particles released!')

if param['test']:
    traj = pset.ParticleFile(name=fh['traj'], outputdt=param['dt_out'])
else:
    traj = pset.ParticleFile(name=fh['traj'], write_ondelete=True)

kernels = (pset.Kernel(AdvectionRK4) +
           pset.Kernel(periodicBC) +
           pset.Kernel(antibeach) +
           pset.Kernel(event))

if param['Kh']:
    kernels += pset.Kernel(DiffusionUniformKh)

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
    (lon_min, lon_max, lat_min, lat_max) = (80, 81, 5.75, 6.75)


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

    with Dataset(fh['ocean'][-4], mode='r') as nc:
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
