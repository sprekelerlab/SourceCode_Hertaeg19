# -*- coding: utf-8 -*-
"""
@author: L. Hertaeg
"""

# %% import packages and functions

import sys
sys.path.append("./Functions")
import numpy as np

import PlotFuncs as PltFcs
import SimNets_FullMC as SimNet
import SimNets_INnet as SimNet_IN

# %% Fig 1B

flg = 1                                 # simulate full network
NeuPar = np.array([-0.7,0.0,0.2,0.4])   # array of neuron parameters (mutual inhibition, recurrence, adaptation, initial synaptic efficacy) 
SimDur = 4000.0                         # simulation time (ms)

# run simulations
x_full, r_full = SimNet.RunSig(flg,NeuPar,SimDur)

# plot data
PltFcs.Plot_1B(x_full,r_full)

# %% Fig 1D

flg = 0                                 # simulate reference network
NeuPar = np.array([-0.7,0.0,0.2,0.4])   # array of neuron parameters (mutual inhibition, recurrence, adaptation, initial synaptic efficacy) 
SimDur = 4000.0                         # simulation time (ms)

# run simulations
x_ref, r_ref = SimNet.RunSig(flg,NeuPar,SimDur)

# plot data
PltFcs.Plot_1D(x_ref,r_ref,x_full,r_full)

# %% Fig 1F

SimDur = 2000.0                                    # simulation time (ms)

# run simulations
x_ref, r_ref = SimNet_IN.RunSig(0,0.0,SimDur)         # reference network
x_full, r_full = SimNet_IN.RunSig(1,-0.9,SimDur)      # full interneuron network

# plot data
PltFcs.Plot_1F(x_ref,r_ref,x_full,r_full)

# %% Fig 1G

SimDur = 2000.0                                         # simulation time (ms)
w = np.arange(-0.1,-1.2,-0.05)                          # range of 

# run simulations
x_KO, r_KO = SimNet_IN.RunAmpFac_w(w,SimDur,Fix=[2,0.0])   # K0 interneuron network
x_full, r_full = SimNet_IN.RunAmpFac_w(w,SimDur)           # full interneuron network
x_ref, r_ref = SimNet_IN.RunSig(0,0.0,SimDur)              # reference enetwork

# plot data
PltFcs.Plot_1G(w,x_ref,r_ref,x_full,r_full,x_KO,r_KO,fig_x=7.0)
