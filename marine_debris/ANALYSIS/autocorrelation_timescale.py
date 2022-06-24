"""
Find autocorrelation decorrelation timescale
@author: noam
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from statsmodels.tsa.stattools import acf


# PARAMETERS
param = {'n_lag': 48,
         'mode': ['NINO4', 'IOD'],
         'ls': ['-', '--']}

# DIRECTORIES
dirs = {'script': os.path.dirname(os.path.realpath(__file__)),
        'ref': os.path.dirname(os.path.realpath(__file__)) + '/../REFERENCE/',
        'fig': os.path.dirname(os.path.realpath(__file__)) + '/../FIGURES/'}

# FILE HANDLES
fh = {'IOD': dirs['ref'] + 'dmi.had.long.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/
      'NINO4': dirs['ref'] + 'nino4.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino4/
      'NINO34': dirs['ref'] + 'nino34.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino34
      'NINO3': dirs['ref'] + 'nino3.long.anom.data', # https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3
      'fig': dirs['fig'] + 'mode_autocorrelation.pdf'}

##############################################################################
# EXTRACT DATA                                                               #
##############################################################################

clim_acf = []
acf_ts = []
lag_axis = np.arange(param['n_lag']+1)

for clim_idx in param['mode']:

    # Load the chosen climate index
    clim = pd.read_table(fh[clim_idx], skiprows=1, delim_whitespace=True,
                         header=None, index_col=0, skipfooter=8, engine='python')

    clim = clim[clim.index.isin(np.arange(1900,2020))].values.flatten()
    clim = xr.DataArray(clim, coords={'time': np.arange(1900,2020,1/12)})

    # Carry out autocorrelation
    clim_acf.append(acf(clim, nlags=param['n_lag']))

    # Calculate e-folding decay scale
    acf_ts.append(lag_axis[np.argmax(clim_acf[-1] < (1/np.e))])


##############################################################################
# PLOT                                                                       #
##############################################################################
f, ax = plt.subplots(1, 1, figsize=(20, 10))

# Generate a t-axis from the datetime format
ax.set_xlim([0, param['n_lag']])

for i in range(len(param['mode'])):
    ax.plot(lag_axis, clim_acf[i], linestyle=param['ls'][i], c='k', linewidth=1,
            label=param['mode'][i] + ' autocorrelation (decay = ' + str(acf_ts[i]) + ' months)')

ax.legend(loc="upper right", frameon=False, fontsize=22)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(0, param['n_lag']+1, 6))

ax.set_ylabel('Lag (months)', fontsize=22)
ax.set_xlabel('Autocorrelation', fontsize=22)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.grid(axis='x')
plt.savefig(fh['fig'], dpi=300, bbox_inches="tight")