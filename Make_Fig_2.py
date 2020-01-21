# -*- coding: utf-8 -*-
"""
@author: L. Hertaeg
"""

# %% import packages and functions

import sys
sys.path.append("./Functions")
import numpy as np

import PlotFuncs as PltFcs
import SimNets_INnet as SimNet

# %% Fig 2A

SimDur = 2000.0                                                 # simulation time (ms)
w = np.arange(-0.1,-1.3,-0.1)                                   # range of mutual inhibition values

# run simulations
STP = [0.1, 100.0] # u, tf
x_strong, r_strong = SimNet.RunAmpFac_STF(w,STP,SimDur)

STP = [0.5, 100.0] # u, tf
x_weak, r_weak = SimNet.RunAmpFac_STF(w,STP,SimDur)

x_no, r_no = SimNet.RunAmpFac_w(w,SimDur)                       # full interneuron network
x_ref, r_ref = SimNet.RunSig(0,0.0,SimDur)                      # reference network

# plot data
PltFcs.Plot_2A(w,x_ref,r_ref,x_no,r_no,x_weak,r_weak,x_strong,r_strong,fig_x=6.5)

# %% Fig 2B

SimDur = 2000.0                                         # simulation time (ms)
wr = np.arange(-0.1,-1.0,-0.05)                         # recurrence strength tested
w = -0.8                                                # fixed mutual inhibition strength

# run simulations
x_full, r_full = SimNet.RunAmpFac_Rec(1,wr,w,SimDur)    # full interneuron network
x_ref, r_ref = SimNet.RunAmpFac_Rec(0,wr,w,SimDur)      # reference enetwork

# plot data
PltFcs.Plot_2B(wr,x_ref,r_ref,x_full,r_full,fig_x=6.0)

