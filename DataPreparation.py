# This script imports base data from the Quijote simulations and formats them
# ready for the main SR-GAN. The 3D density fields are imported, then downscaled
# to the various grid resolutions to form our LR and HR pairs. 2D slices are 
# then taken from the downscaled grids along all 3 axes, and interpolated
# accordingly to make them 1 pixel wide. These arrays are then sorted into the
# required data format: a 2x2 grid with one of each resolution image. The images
# are saved as PNG files and exported to a folder ready to be read into the SR
# model.

import numpy as np
import plotting_library as PL
import sys,os

import readgadget
import MAS_library as MASL
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm

import Pk_library as PKL


# %%
################################# INPUT #######################################

Resolution = 'LR'
Redshift = 4

Grids = [32, 64, 128, 256]

# Setting up the empty arrays to store out density fields.

# For Snapshots
DF32 = []
DF64 = []
DF128 = []
DF256 = []


for Instance in range(0, 3):

    snapshot = 'DataFiles/%s/%d/snapdir_00%d/snap_00%d' % (Resolution, Instance, Redshift, Redshift)
    
    MAS     = 'CIC'  #mass-assigment scheme
    verbose = True   #print information on progress
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
    
    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    # Nall     = header.nall         #Total number of particles
    # Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    # Omega_m  = header.omega_m      #value of Omega_m
    # Omega_l  = header.omega_l      #value of Omega_l
    # h        = header.hubble       #value of h
    # redshift = header.redshift     #redshift of the snapshot
    # Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l) #Value of H(z) in km/s/(Mpc/h)
    
    # particle positions in 3D, creates an array of size 3 by (grid)^3
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    
    
    for Grid in Grids:

        # density field parameters
        grid    = Grid # 128    #the 3D field will have grid x grid x grid voxels

        # define 3D density field
        delta = np.zeros((grid,grid,grid), dtype=np.float32)
        
        # construct 3D density field
        MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)
        
        # at this point, delta contains the effective number of particles in each voxel
        # now compute overdensity and density constrast
        delta /= np.mean(delta, dtype=np.float64)
        delta -= 1.0
        
        if Grid == 32:
            # For Density Fields:
            DF32.append(delta)

        elif Grid == 64:
            DF64.append(delta)
       
        elif Grid == 128:
            DF128.append(delta)
        
        elif Grid == 256:
            DF256.append(delta)


# %% Taking 2D slices of data:
    
DF32 = np.array(DF32)
DF64 = np.array(DF64)
DF128 = np.array(DF128)
DF256 = np.array(DF256)
    
# %% Taking 2D slices:

import tensorflow as tf
from tensorflow import keras
    
# Taking 32x32 slices and upscaling into 256x256 slices:

train_images_32 = []    
train_images_64 = []
train_images_128 = []
train_images_256 = []

for i in range(0, 32):
    
    Slice = DF32[0,:,:,i:i+1]
    
    Slice = Slice.repeat(8, axis=0).repeat(8, axis=1)
    
    # Normalising
    Slice = Slice - np.min(Slice)
    Slice = (Slice - (np.max(Slice)/2))/(np.max(Slice)/2)
    
    # train_images_LR is now a list, 32 items long, of 32x32 slices upscaled to 256x256
    train_images_32.append(Slice)
    
    
# And now doing the same with the 64 grid DF
# We require the first 2 slices of the 64 3D DF as that equates to the same
# percentage of data as the 32 slices.

for i in range(0, 64, 2):
    
    Slice = DF64[0,:,:,i:i+2]
    Slice = np.sum(Slice, axis = 2, keepdims=True)
    
    Slice = Slice.repeat(4, axis=0).repeat(4, axis=1)

    # Normalising
    Slice = Slice - np.min(Slice)
    Slice = (Slice - (np.max(Slice)/2))/(np.max(Slice)/2)
    
    train_images_64.append(Slice)
    

# And now doing the same with the 128 grid DF
# We require the first 4 slices of the 128 3D DF as that equates to the same
# percentage of data as the 32 slices.

for i in range(0, 128, 4):
    
    Slice = DF128[0,:,:,i:i+4]
    Slice = np.sum(Slice, axis = 2, keepdims=True)
    
    Slice = Slice.repeat(2, axis=0).repeat(2, axis=1)

    # Normalising
    Slice = Slice - np.min(Slice)
    Slice = (Slice - (np.max(Slice)/2))/(np.max(Slice)/2)
    
    train_images_128.append(Slice)


# And now doing the same with the 256 grid DF
# We require the first 8 slices of the 256 3D DF as that equates to the same
# percentage of data as the 32 slices.

for i in range(0, 256, 8):
    
    Slice = DF256[0,:,:,i:i+8]
    Slice = np.sum(Slice, axis = 2, keepdims=True)

    # Normalising
    Slice = Slice - np.min(Slice)
    Slice = (Slice - (np.max(Slice)/2))/(np.max(Slice)/2)
    
    train_images_256.append(Slice)
    
# All of the above train_images_number are now lists, 32 items long, of 256x256x1 
# slices of data, upscaled from their relevant grid size, and all with an effective
# width of 1/32 of their original 3D DF.

# %% Putting them into a single array of all 4 slices, 512x512, starting with 
# 32x32 in the top left and ending with 256x256 in the bottom right.

train_images_32to256 = []
train_image_total = np.zeros((512, 512))

for n in range (0, 32): # Starting off with 10 sets of 4 images

    input32 = train_images_32[n]
    input64 = train_images_64[n]
    input128 = train_images_128[n]
    input256 = train_images_256[n]
    
    input32 = input32[:,:,0]
    input64 = input64[:,:,0]
    input128 = input128[:,:,0]
    input256 = input256[:,:,0]
    
    train_image_total[0:256, 0:256] = input32
    train_image_total[0:256, 256:512] = input64
    train_image_total[256:512, 0:256] = input128
    train_image_total[256:512, 256:512] = input256
    
    
    train_set = train_image_total
    train_set = np.array(train_set)
    
    train_images_32to256.append(train_set)


# %% Trying to save as PNG:

# Need pip install pypng
import png

for n in range (30, 32): # Starting off with 32 sets of 4 images

    #train_set = [train_images_32[n], train_images_64[n], train_images_128[n], train_images_256[n]]
    train_set = train_images_32to256[n]
    train_set = np.array(train_set)
    
    x = train_set
    zgray = (65535*((x - x.min())/x.ptp())).astype(np.uint16)
    

    Path = 'test_set_%d.png' % (n)
    # np.save(Path+Path2, train_set)
    
    with open('DataFiles/NewCosmo/New_Cosmo_Test_Set/'+Path, 'wb') as f:
        writer = png.Writer(width=zgray.shape[1], height=zgray.shape[0], bitdepth=16) #, greyscale=True)
        zgray2list = zgray.tolist()
        writer.write(f, zgray2list)
