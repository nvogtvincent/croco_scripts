#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for coral_sources.py
@author: Noam Vogt-Vincent (2020)
"""

import gdal
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import ndimage

def regrid_coral_raster(coral_in_fh, coral_grid_in_fh, model_grid_fh, dr):
    """
    This function script identifies (psi) grid cells that contain reefs from a
    raster. For my experiments, I rasterised reef locations from the UNEP-WCM
    Global Distribution of Warm-Water Coral Reefs (2018) dataset onto the 500m
    GEBCO_2019 grid, which is the input for this script.
    https://data.unep-wcmc.org/datasets/1

    raw_coral_file = rasterised UNEP-WCM data
    raw_coral_grid_file = GEBCO_2019 (in this case)
    model_grid_file = model file with coordinates of rho/psi points + masks

    Using 2D binning to ensure that all model cells including at least 1 reef
    cell from the raster are counted as reef cells

    @author: Noam Vogt-Vincent (2020)
    """

    plot_psi = 1  # Plot psi coral cell status
    save_psi = 0  # Save psi coral cells

    # PART 1: IMPORT GRID AND BIN
    # Open the coral raster and its grid
    print('')
    print('Stage 1: grid coral sites!')
    print('')
    print('Binning raster input...')
    coral_in_object = gdal.Open(coral_in_fh)
    coral_in = coral_in_object.ReadAsArray()
    coral_in = np.flip(coral_in, axis=0)

    coral_grid_in = Dataset(coral_grid_in_fh, mode='r')
    lon_in_ = np.array(coral_grid_in.variables['lon'][:])
    lat_in_ = np.array(coral_grid_in.variables['lat'][:])
    lon_in, lat_in = np.meshgrid(lon_in_, lat_in_)
    coral_grid_in.close()

    # Open the model grid
    model_grid = Dataset(model_grid_fh, mode='r')
    lon_rho = np.array(model_grid.variables['lon_rho'][:])
    lat_rho = np.array(model_grid.variables['lat_rho'][:])
    lon_psi = np.array(model_grid.variables['lon_psi'][:])
    lat_psi = np.array(model_grid.variables['lat_psi'][:])
    mask_rho = np.array(model_grid.variables['mask_rho'][:])
    mask_psi = np.array(model_grid.variables['mask_psi'][:])
    model_grid.close()

    # Bin raster coral sites to model psi grid (limited by the rho grid)
    # Firstly mask non-coral coordinates with 0s
    lat_in = lat_in*coral_in
    lon_in = lon_in*coral_in

    # Flatten array
    lat_in = lat_in.flatten()
    lon_in = lon_in.flatten()

    # Remove all masked values
    lat_in = lat_in[~(lat_in == 0)]
    lon_in = lon_in[~(lon_in == 0)]

    # Now bin on psi grid (using rho grid as bin edges)
    edges_lon = lon_rho[0, :]
    edges_lat = lat_rho[:, 0]

    coral_psi = np.histogram2d(lon_in, lat_in, bins=(edges_lon, edges_lat))
    coral_psi = np.transpose(coral_psi[0])

    # Now make a new array to store boolean coral grid
    # coral_dens = np.copy(coral_psi)  # Stored in case it becomes useful
    coral_psi[coral_psi > 0] = 1  # 1 = coral
    print('...complete!')
    print('')
    print('Number of coral cells identified: ' + str(int(np.sum(coral_psi))))

    # PART 2: PROCESS CORAL GRID
    # Firstly, mark points where the coral grid overlaps with the mask
    coral_mask_clash = coral_psi*(1-mask_psi)  # 1 = clash
    print('Number of coral cells under land mask: ' +
          str(int(np.sum(coral_mask_clash))))

    # Now surround all these with new coral cells
    struct = ndimage.generate_binary_structure(2, 2)
    new_coral = ndimage.binary_dilation(coral_mask_clash,
                                        structure=struct).astype(
                                            coral_mask_clash.dtype)
    # Remove all coral cells that are still masked
    new_coral = new_coral*mask_psi  # 1 = coral
    coral_psi = coral_psi*mask_psi  # 1 = coral
    # Remove new coral cells that already have coral
    new_coral = new_coral*(1-coral_psi)  # 1 = coral
    print('Number of new coral cells added from mask-shift: ' +
          str(int(np.sum(new_coral))))
    print('')

    # Plot status of psi cells
    if plot_psi == 1:
        print('Plotting coral cells (psi)...')
        # This plot will show ocean cells (blue), direct corals from data (red)
        # new corals from the mask shift (orange), and the land mask (light
        # brown).

        # Colour coral status
        coral_stat = np.zeros_like(coral_psi)
        coral_stat[coral_psi == 1] = 0.325
        coral_stat[new_coral == 1] = 0.125

        # Mask out all non-coral tiles
        coral_stat = np.ma.masked_array(coral_stat, coral_stat == 0)

        # Now, set up the grid to plot the rho mask (i.e. psi points)
        edges_lon_psi = lon_psi[0, :]
        edges_lat_psi = lat_psi[:, 0]

        # Now expand this grid to be able to plot all rho points
        edges_lat_psi = np.append((2*edges_lat_psi[0] - edges_lat_psi[1]),
                                  edges_lat_psi)
        edges_lat_psi = np.append(edges_lat_psi,
                                  (2*edges_lat_psi[-1] - edges_lat_psi[-2]))

        edges_lon_psi = np.append((2*edges_lon_psi[0] - edges_lon_psi[1]),
                                  edges_lon_psi)
        edges_lon_psi = np.append(edges_lon_psi,
                                  (2*edges_lon_psi[-1] - edges_lon_psi[-2]))

        # Set up mask colours
        disp_rho_mask = np.copy(mask_rho)
        disp_rho_mask[mask_rho == 1] = 0.075  # ocean
        disp_rho_mask[mask_rho == 0] = 0.575  # land

        # Set up figure
        f0 = plt.figure(figsize=(21, 12))
        a0 = f0.add_subplot(1, 1, 1)
        a0.pcolormesh(edges_lon_psi, edges_lat_psi, disp_rho_mask,
                      cmap='tab20', vmin=0, vmax=1)
        a0.pcolormesh(edges_lon, edges_lat, coral_stat, cmap='tab20',
                      vmin=0, vmax=1)
        a0.set_aspect('auto')
        a0.set_xlabel('Longitude ($^\circ$E)')
        a0.set_ylabel('Latitude ($^\circ$N)')
        a0.set_title('Coral cell status (psi points)')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-',
                 alpha=0.25)

        figure_loc = dr + '../FIGURES/coral_cells_psi.jpg'
        plt.savefig(figure_loc, dpi=600)
        plt.close()
        print('...plotting complete!')
        print('')

    coral_psi += new_coral
    print('Total number of coral cells: ' + str(int(np.sum(coral_psi))))

    if save_psi == 1:
        print('')
        print('Saving coral grids...')

        # Save all of the information required to evaluate the coral grids:
        # coral_psi (actual coral cells on the psi grid)
        # edges_lat/edges_lon (boundaries for coral cells on psi grid)
        # original version of edges_lat_psi/edges_lon_psi (centroids of coral
        # cells on the psi grid)

        # Saved in order coral_psi/edges_lon/edges_lat/edges_lon_p/edges_lat_p

        edges_lon_psi = lon_psi[0, :]
        edges_lat_psi = lat_psi[:, 0]

        out_file = dr + 'coral_grid.npz'

        np.savez(out_file, coral_psi, edges_lon, edges_lat, edges_lon_psi,
                 edges_lat_psi)

