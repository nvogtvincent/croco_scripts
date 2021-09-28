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







def convert(dir_in, fh_out, **kwargs):
    '''
    Parameters
    ----------
    dir_in : STR
        TOP LEVEL DIRECTORY CONTAINING PARCELS OUTPUT.

    fh_out : STR
        FILE HANDLE FOR OUTPUT (MUST BE NETCDF).

    **kwargs : ALLOWED KWARGG:
        fwd : BOOL (default TRUE)
            IF PARTICLE TRACKING IS FORWARD OR BACKWARD IN TIME

    Returns
    -------
    None.

    '''

    # IMPORTANT NOTE: CURRENTLY IGNORES ALL DELETED PARTICLES!

    # BASIC METHODOLOGY:
    # 1. Scan through numpy files to find out the total number of particles
    # 2. Create the netcdf file

    # Detect available system memory
    avail_mem = psutil.virtual_memory()[4]

    # Identify the number of parallel processes the parcels simulation was run
    # across
    dirs     = glob(dir_in + '/*')
    nproc    = len(dirs)

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
    out_dir  = this_dir + '/example_out.nc'

    convert(this_dir, out_dir, fwd=False)



