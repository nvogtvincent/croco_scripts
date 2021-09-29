#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This package converts raw parcels output in netcdf format into a single
netcdf file.

@author: Noam Vogt-Vincent
"""

# Import packages
import os
import sys
import parcels
import numpy as np
from glob import glob
from netCDF4 import Dataset
from alive_progress import alive_bar


def convert(dir_in, fh_out, **kwargs):
    '''
    Parameters
    ----------
    dir_in : STR
        TOP LEVEL DIRECTORY CONTAINING PARCELS OUTPUT.

    fh_out : STR
        FILE HANDLE FOR OUTPUT (MUST BE NETCDF).

    **kwargs : ALLOWED KWARGS:
        fwd : BOOL (default TRUE)
            IF PARTICLE TRACKING IS FORWARD OR BACKWARD IN TIME

    Returns
    -------
    None.

    '''

    # TO DO:
    # Select which variables to include with kwargs

    # BASIC METHODOLOGY:
    # 1. Scan through numpy files to find out the total number of particles and
    #    other key parameters (datatype, time, etc.)
    # 2. Create the netcdf file
    # 3. Incrementally fill the netcdf file

    ###########################################################################

    # Identify the number of parallel processes the parcels simulation was run
    # across
    dirs     = [x for x in glob(dir_in + '/*') if '.' not in x]
    nproc    = len(dirs)

    # Now scan through the processes to find out the following:
    # 1. Minimum time
    # 2. Maximum time
    # 3. Total number of particles (with particle id)

    global_tmin  = []
    global_tmax  = []
    global_idmax = []

    frame_time = dict((directory, []) for directory in range(nproc))

    for proc_num, proc_dir in enumerate(dirs):
        # Load the info file for the current process
        proc_info = np.load(proc_dir + '/pset_info.npy',
                            allow_pickle=True).item()
        proc_fhs = proc_info['file_list']

        # Record the time of every file
        frame_time[proc_num] = np.zeros([len(proc_fhs),], dtype=np.float64)

        for fh_num, fh in enumerate(proc_fhs):
            frame_time[proc_num][fh_num] = np.load(proc_dir + '/' + fh.split('/')[-1],
                                                   allow_pickle=True).item()['time'][0]

        # Add the tmin, tmax and idmax to the lists
        global_idmax.append(proc_info['maxid_written'])
        global_tmin.append(np.min(frame_time[proc_num]))
        global_tmax.append(np.max(frame_time[proc_num]))

        if proc_num == 0:
            # Record time origin
            global_torigin = np.datetime_as_string(proc_info['time_origin'].time_origin)

            # Calculate the time step
            time_dt = np.unique(np.gradient(np.array(frame_time[proc_num])),
                                return_counts=True)

            # Check that the modal time step is accurate (>10% of total)
            if np.max(time_dt[1]) > len(frame_time[proc_num])/10:
                time_dt = time_dt[0][np.argmax(time_dt[1])]
                print('Time step found.')
                print('Time step set to ' + str(int(time_dt)) + 's')
                print('')
            else:
                print('Could not find reliable time step.')
                time_dt = float(input('Please enter the time step in seconds:'))

            if time_dt > 0:
                fwd = True
            else:
                fwd = False

            # Create a dict to hold all of the variables and datatypes
            var_names = proc_info['var_names']

            # Remove id, time and depth (id and time are recorded as 1D arrays to
            # save space, and depth is not needed at the moment)

            var_names.remove('id')
            var_names.remove('time')
            var_names.remove('depth')

            var_dtype = dict((name, []) for name in var_names)

            for var_name in var_names:
                temp_fh = np.load(proc_dir + '/' + proc_fhs[0].split('/')[-1],
                                  allow_pickle=True).item()[var_name][0]
                var_dtype[var_name] = temp_fh.dtype

            # Check if any 'once' variables are included
            if 'var_names_once' in proc_info:
                if len(proc_info['var_names_once']) > 0:
                    var_names_once = proc_info['var_names_once']
                    proc_fhs_once = proc_info['file_list_once']
                    once = True

                    var_once_dtype = dict((name, []) for name in var_names_once)

                    for var_name in var_names_once:
                        temp_fh = np.load(proc_dir + '/' + proc_fhs_once[0].split('/')[-1],
                                          allow_pickle=True).item()[var_name][0]
                        var_once_dtype[var_name] = temp_fh.dtype



    # Now calculate the true global tmin, tmax, and idmax
    global_tmax = np.max(global_tmax)
    global_tmin = np.min(global_tmin)
    global_idmax = np.max(global_idmax)

    # Now generate the time series
    # Routine depends on state of fwd
    if fwd:
        array_time = np.arange(global_tmin,
                               global_tmax+time_dt,
                               time_dt)
        array_time[-1] = global_tmax
    else:
        array_time = np.arange(global_tmax,
                               global_tmin+time_dt,
                               time_dt)
        array_time[-1] = global_tmin

    # Now generate the ids
    array_id   = np.arange(0, global_idmax+1)

    global_ntime = len(array_time)
    global_npart = len(array_id)

    ###########################################################################

    # Now create the netcdf file

    # Create a dictionary to map known variables to descriptions
    long_name = {'lon' : 'longitude',
                 'lat' : 'latitude'}

    units = {'lon' : 'degrees_east',
             'lat' : 'degrees_north'}

    standard_name = {'lon' : 'longitude_degrees_east',
                     'lat' : 'latitude_degrees_north'}

    # Define output parameters
    # Missing values
    missval = -999

    # Compression level
    clevel = 4

    with Dataset(fh_out, mode='w') as nc:
        # Create the dimensions
        nc.createDimension('trajectory', global_npart)
        nc.createDimension('time', global_ntime)

        # Create the variables
        nc.createVariable('id', 'i4', ('trajectory'), zlib=True, complevel=clevel)
        nc.variables['id'].long_name = 'particle_id'
        nc.variables['id'].standard_name = 'id'
        nc.variables['id'][:] = array_id

        nc.createVariable('time', 'f8', ('time'), zlib=True, complevel=clevel)
        nc.variables['time'].long_name = 'particle_time'
        nc.variables['time'].standard_name = 'time'
        nc.variables['time'].standard_name = 'seconds since ' + global_torigin
        nc.variables['time'][:] = array_time

        for var_name in var_names:
            nc.createVariable(var_name, var_dtype[var_name].str[1:],
                              ('trajectory', 'time'), zlib=True,
                              complevel=clevel)
            nc.variables[var_name].long_name = long_name[var_name]
            nc.variables[var_name].units = units[var_name]
            nc.variables[var_name].standard_name = standard_name[var_name]
            nc.variables[var_name].missing_value = missval

        if once:
            for var_name in var_names_once:
                nc.createVariable(var_name, var_once_dtype[var_name].str[1:],
                                  ('trajectory'), zlib=True,
                                  complevel=clevel)
                nc.variables[var_name].missing_value = missval


        nc.time_origin = global_torigin

    ###########################################################################

    # netcdf file has been created, now start populating with data
    # To minimise memory use, this is done through the following method:
    # Loop through time steps
    # -> Set up a 1 x npart vector for that vector for each variable
    #    Loop through processes
    #    -> Check if time step is present in process
    #       If so, add data for each variable to relevant vector
    # Copy to netcdf

    print('Now writing netcdf file...')

    # Firstly write 'once' variables if they exist (this is quick)
    if once:
        var_data = dict((name, []) for name in var_names_once)

        for var_name in var_names_once:
            # Shouldn't really be any missing values, but anyway
            var_data[var_name] = missval*np.ones([global_npart,],
                                                 dtype=var_once_dtype[var_name])

        for proc_num, proc_dir in enumerate(dirs):
            proc_info = np.load(proc_dir + '/pset_info.npy',
                                allow_pickle=True).item()
            proc_fhs_once = proc_info['file_list_once']

            for fh in proc_fhs_once:
                data = np.load(proc_dir + '/' + fh.split('/')[-1],
                               allow_pickle=True).item()

                # Check that particle ids are sorted (method is reliant on this)
                if np.all(data['id'][:-1] < data['id'][1:]):
                    # Load indices and insert into data dictionary
                    id_index = np.isin(array_id, data['id'])
                    # id_index_not = np.nonzero(~id_index)[0]
                    id_index     = np.nonzero(id_index)[0]

                    for var_name in var_names_once:
                        # Insert into array
                        np.put(var_data[var_name][:],
                               id_index,
                               data[var_name])

        # Now write to netcdf
        with Dataset(fh_out, mode='r+') as nc:
            for var_name in var_names_once:
                nc.variables[var_name][:] = var_data[var_name]

    with alive_bar(len(array_time), bar='smooth', spinner='dots_waves') as bar:
        for ti, t in enumerate(array_time):
            # var_data: dictionary that contains the data for variables for this
            #           time slice
            # var_data_present: dictionary that contains a 1 if data exists for the
            #           variable for this time slice
            var_data = dict((name, []) for name in var_names)
            # var_data_present = dict((name, []) for name in var_names)

            for var_name in var_names:
                var_data[var_name] = missval*np.ones([global_npart,],
                                                     dtype=var_dtype[var_name])

            for proc_num, proc_dir in enumerate(dirs):
                # Check if time is present in this process
                ti_proc = np.where(frame_time[proc_num] == t)[0]

                # Only continue if time is present [len(ti_proc) >= 1].
                # I have no idea why parcels sometimes saves a frame as multiple
                # files, but it does and we have to account for this.

                for ti_proc_i in ti_proc:
                    # Load in frame
                    proc_info = np.load(proc_dir + '/pset_info.npy',
                                        allow_pickle=True).item()
                    proc_fhs = proc_info['file_list']

                    data = np.load(proc_dir + '/' + proc_fhs[ti_proc_i].split('/')[-1],
                                   allow_pickle=True).item()

                    # Check that particle ids are sorted (method is reliant on this)
                    if np.all(data['id'][:-1] < data['id'][1:]):
                        # Load indices and insert into data dictionary
                        id_index = np.isin(array_id, data['id'])
                        # id_index_not = np.nonzero(~id_index)[0]
                        id_index     = np.nonzero(id_index)[0]

                        for var_name in var_names:
                            # Insert into array
                            np.put(var_data[var_name][:],
                                   id_index,
                                   data[var_name])


            # Now set all points without data to a fill value
            # for var_name in var_names:
            #     var_data[var_name][var_data_present == 0] = -999

            # Now write to netcdf
            with Dataset(fh_out, mode='r+') as nc:
                for var_name in var_names:
                    nc.variables[var_name][:, ti] = var_data[var_name]

            bar()


if __name__ == "__main__":

    '''
    When run from the terminal, the first argument is the path to the directory
    containing parcels output (i.e. folders 0, 1, 2...). The second argument is
    the directory to the output and output name.
    '''

    this_dir = os.path.dirname(os.path.realpath(__file__)) + '/'

    try:
        parcels_output_dir = this_dir + sys.argv[1]
        netcdf_fh = this_dir + sys.argv[2]

        if parcels_output_dir[-1] != '/':
            parcels_output_dir += '/'
    except:
        parcels_output_dir = this_dir + input('Please enter path to parcels numpy directory')
        netcdf_fh = this_dir + input('Please enter path to output netcdf')

    convert(parcels_output_dir, netcdf_fh)




