#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This package converts raw parcels output in netcdf format into a single
netcdf file.

@author: Noam Vogt-Vincent
"""

# Import packages
import psutil
import os
import parcels
import numpy as np
from glob import glob
from netCDF4 import Dataset


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

    # IMPORTANT NOTE: CHUNKING IS NOT YET IMPLEMENTED!

    # BASIC METHODOLOGY:
    # 1. Scan through numpy files to find out the total number of particles and
    #    other key parameters (datatype, time, etc.)
    # 2. Create the netcdf file
    # 3. Incrementally fill the netcdf file

    ###########################################################################

    # Detect available system memory
    avail_mem = psutil.virtual_memory()[4]

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
                print('Confident time step found.')
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

            # var_data  = dict((name, []) for name in var_names)
            var_dtype = dict((name, []) for name in var_names)

            for var_name in var_names:
                temp_fh = np.load(proc_dir + '/' + proc_fhs[0].split('/')[-1],
                                  allow_pickle=True).item()[var_name][0]
                var_dtype[var_name] = temp_fh.dtype


    # Now calculate the true global tmin, tmax, and idmax
    global_tmax = np.max(global_tmax)
    global_tmin = np.min(global_tmin)
    global_idmax = np.max(global_idmax)

    # Now generate the time series
    array_time = np.arange(global_tmin,
                           global_tmax+np.abs(time_dt),
                           np.abs(time_dt))

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

    with Dataset(fh_out, mode='w') as nc:
        # Create the dimensions
        nc.createDimension('trajectory', global_npart)
        nc.createDimension('time', global_ntime)

        # Create the variables
        nc.createVariable('id', 'i4', ('trajectory'), zlib=True)
        nc.variables['id'].long_name = 'particle_id'
        nc.variables['id'].standard_name = 'id'
        nc.variables['id'][:] = array_id

        nc.createVariable('time', 'f8', ('time'), zlib=True)
        nc.variables['time'].long_name = 'particle_time'
        nc.variables['time'].standard_name = 'time'
        nc.variables['time'].standard_name = 'seconds since ' + global_torigin
        nc.variables['time'][:] = array_time

        for var_name in var_names:
            nc.createVariable(var_name, var_dtype[var_name].str[1:],
                              ('trajectory', 'time'), zlib=True)
            nc.variables[var_name].long_name = long_name[var_name]
            nc.variables[var_name].units = units[var_name]
            nc.variables[var_name].standard_name = standard_name[var_name]

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

    for ti, t in enumerate(array_time):
        var_data = dict((name, []) for name in var_names)
        for var_name in var_names:
            var_data[var_name] = np.zeros([global_npart,],
                                            dtype=var_dtype[var_name])

        for proc_num, proc_dir in enumerate(dirs):
            # Check if time is present in this process
            ti_proc = np.where(frame_time[proc_num] == t)[0]
            if len(ti_proc):
                # Load in frame
                proc_info = np.load(proc_dir + '/pset_info.npy',
                                    allow_pickle=True).item()
                proc_fhs = proc_info['file_list']

                data = np.load(proc_dir + '/' + proc_fhs[ti_proc[0]].split('/')[-1],
                               allow_pickle=True).item()

                # Load indices and isnert into data dictionary
                id_index = np.isin(array_id, data['id'])
                # id_index_not = np.nonzero(~id_index)[0]
                id_index     = np.nonzero(id_index)[0]

                for var_name in var_names:
                    # Insert into array
                    np.put(var_data[var_name][:],
                           id_index,
                           data[var_name])

                    # # Set other values to NaNs
                    # np.put(var_data[var_name][:, ti],
                    #        id_index_not,
                    #        np.nan)






    # First calculate the predicted size of the largest (64-bit) array of
    # particles and check if this fits within system memory

    # estimated_size = global_npart*global_ntime*8

    # if estimated_size*1.5 < avail_mem:
    #     mem_ok = True
    # else:
    #     mem_ok = False
    #     raise NotImplementedError('Chunking not yet implemented!')

    # if mem_ok:
        #
        # for proc_num, proc_dir in enumerate(dirs):
        #     proc_info = np.load(proc_dir + '/pset_info.npy',
        #                         allow_pickle=True).item()
        #     proc_fhs = proc_info['file_list']

        #     for fh_num, fh in enumerate(proc_fhs):
        #         for var_name in var_names:
        #             var_data = 1




        print()





    for proc_num, proc_dir in enumerate(dirs):
        # Load the info file for the current process
        proc_info = np.load(proc_dir + '/pset_info.npy',
                            allow_pickle=True).item()

        proc_fhs  = proc_info['file_list'][::-1]

        # Create a dict to hold all of the variables and datatypes
        var_names = proc_info['var_names']

        # Remove id, time and depth (id and time are recorded as 1D arrays to
        # save space, and depth is not needed at the moment)

        var_names.remove('id')
        var_names.remove('time')
        var_names.remove('depth')

        var_data  = dict((name, []) for name in var_names)
        var_dtype = dict((name, []) for name in var_names)

        for var_name in var_names:
            temp_fh = np.load(proc_dir + '/' + proc_fhs[0].split('/')[-1],
                              allow_pickle=True).item()[var_name][0]
            var_dtype[var_name] = temp_fh.dtype

        # Firstly try to intelligently figure out the output time step by
        # calculating the time step between the first 10 frames and looking
        # for the most common time step
        time_list = []

        for i in range(10):
            fh = proc_fhs[i]
            time_list.append(np.load(proc_dir + '/' + fh.split('/')[-1],
                                     allow_pickle=True).item()['time'][0])

        time_dt = np.unique(np.gradient(np.array(time_list)),
                            return_counts=True)

        # One time step must appear at least half the time to be taken as
        # reliable. Otherwise, search the entire dataset.

        if time_dt[1][-1] > 5:
            time_dt = -time_dt[0][-1]
            print('Confident time step found.')
            print('Time step set to ' + str(int(time_dt)) + 's')
            print('')
        else:
            print('Could not find reliable time step with quick search.')
            print('Searching all time frames.')
            print('')

            time_list = []

            for fh in proc_fhs:
                time_list.append(np.load(proc_dir + '/' + fh.split('/')[-1],
                                         allow_pickle=True).item()['time'][0])

            time_dt = np.unique(np.gradient(np.array(time_list)),
                                return_counts=True)

            # One time step must appear at least 10% of the time to be taken as
            # reliable. Otherwise, return an error.

            if time_dt[1][-1] > int(len(proc_fhs)/10):
                time_dt = -time_dt[0][-1]
                print('Confident time step found.')
                print('Time step set to ' + str(int(time_dt)) + 's')
                print('')
            else:
                print('Could not find reliable time step.')
                time_dt = float(input('Please enter the time step in seconds:'))

        if time_dt > 0:
            fwd = True
        else:
            fwd = False

        # Now generate the time series
        t0 = np.load(proc_dir + '/' + proc_fhs[-1].split('/')[-1],
                     allow_pickle=True).item()['time'][0]
        t1 = np.load(proc_dir + '/' + proc_fhs[0].split('/')[-1],
                     allow_pickle=True).item()['time'][0]

        array_time = np.arange(t0, t1+time_dt, time_dt)

        if not fwd:
            # Keep the time in true chronological order regardless of
            # backtracking
            array_time = array_time[::-1]

        # Now generate a complete list of particle IDs (ignoring deleted!)
        array_id = np.load(proc_dir + '/' + proc_fhs[0].split('/')[-1],
                           allow_pickle=True).item()['id']

        p_num = len(array_id)
        t_num = len(array_time)

        # Now calculate the predicted size of the largest (64-bit) array of
        # particles and check if this fits within system memory

        estimated_size = p_num*t_num*8

        if estimated_size*1.5 < avail_mem:
            mem_ok = True
        else:
            mem_ok = False
            raise NotImplementedError('Chunking not yet implemented!')

        # Now write data to one numpy array
        if mem_ok:
            for var_name in var_names:
                var_data[var_name] = np.zeros((p_num, t_num),
                                                dtype=var_dtype[var_name])

                for fh in proc_fhs:
                    data = np.load(proc_dir + '/' + fh.split('/')[-1],
                                   allow_pickle=True).item()

                    # Calculate time index of slice, and id indices (but only if
                    # time is in time array)
                    try:
                        t_index  = np.where(array_time == data['time'][0])[0][0]
                    except IndexError:
                        t_index = -1

                    if t_index >= 0:
                        id_index = np.isin(array_id, data['id'])
                        id_index_not = np.nonzero(~id_index)[0]
                        id_index     = np.nonzero(id_index)[0]

                        # Insert into array
                        np.put(var_data[var_name][:, t_index],
                               id_index,
                               data[var_name])

                        # Set other values to NaNs
                        np.put(var_data[var_name][:, t_index],
                               id_index_not,
                               np.nan)


        for fh_num, fh in enumerate(proc_fhs):

            # Append frame time to the time list
            time_list.append(np.load(proc_dir + '/' + fh.split('/')[-1],
                                     allow_pickle=True).item()['time'][0])

        # Find unique values
        time_list2 = set(time_list)


        for fh_num, fh in enumerate(proc_info['file_list'][::-1]):
            # Now construct an array for particles in this process. To speed
            # this up, this loop will work 'backward' from the last time step.

            this_file = np.load(proc_dir + '/' + fh.split('/')[-1],
                                allow_pickle=True).item()

            # this_file = np.load(fh, allow_pickle=True).item()['id']

            print()

        print()



    print()










if __name__ == "__main__":
    print()

    # example_dir = os.path.dirname(os.path.realpath(__file__)) + '/test_output'
    # fh1 = example_dir + '/6/338.npy'

    # f1  = np.load(fh1, allow_pickle=True)
    # f1_arr = f1.item()

    this_dir = os.path.dirname(os.path.realpath(__file__)) + '/test_output/'
    out_dir  = this_dir + 'example_out.nc'

    convert(this_dir, out_dir, fwd=False)



