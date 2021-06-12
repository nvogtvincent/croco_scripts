#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the location for coral larvae releases for the WIO
model. The script firstly identifies (rho) grid cells that contain reefs. It
then uses the model mask to ensure that all reefs are in ocean tiles, and
shifts reefs to model psi nodes. The script finally calculates the area within
which larval releases occur and the release locations.
@author: Noam Vogt-Vincent (2020)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ct
import cmocean.cm as cm
import os


from scipy.interpolate import griddata
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset

from coral_grid_methods import regrid_coral_raster as rcr

##############################################################################
# File locations #############################################################
##############################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = (this_dir + '/DATA/')
preproc_coral_file1 = data_dir + 'coral_grid.tif'
preproc_coral_file2 = data_dir + 'GEBCO2019.nc'
croco_grid_file = data_dir + 'croco_grd.nc'
cmems_grid_file = data_dir + 'croco_grd.nc'
proc_coral_file = data_dir + 'coral_grid.npz'

##############################################################################
# Options ####################################################################
##############################################################################

# STAGE 1 (grid corals from raster)
S1_activate = 1  # Activate stage 1

##############################################################################
# Main script ################################################################
##############################################################################
if S1_activate == 1:
    rcr(preproc_coral_file1, preproc_coral_file2, croco_grid_file, data_dir)
else:
    print('Skipping stage 1!')



