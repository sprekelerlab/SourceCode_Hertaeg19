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

# %% Fig 6 A

SimDur = 4000.0                             # simulation time (ms)
x_VIP = np.linspace(-3.0,2.0,41)            # modulatory input to VIPs
x_SOM = np.linspace(-2.0,0.0,41)            # modulatory input to SOMs
w0 = -0.8                                   # fixed mutual inhibition
    
# run simulations
rates = SimNet.RunSig_2Inp(x_SOM,x_VIP,w0,SimDur)

# plot data
PltFcs.Plot_6A(x_SOM,x_VIP,rates,fig_x=6.0,fig_y=10.0)

# %% Fig 6 C & D

SimDur = 10000.0                            # simulation time (ms)
w = -0.9                                    # mutual inhibition: w = wSV, wVS = w - 0.2

# run simulations
t, R_mis, Xv_mis, Xm_mis = SimNet.Run_MismatchDetection(0,w,SimDur)
t, R_play, Xv_play, Xm_play = SimNet.Run_MismatchDetection(1,w,SimDur)

# plot data
PltFcs.Plot_6CD(t,R_mis,Xv_mis,Xm_mis,R_play,Xv_play,Xm_play,SimDur,fig_x=10.0,fig_y=4.0)

# %% Fig 6 E

SimDur = 2500                               # simulation time (ms)

# run simulations
w = -0.1 # w = wSV, wVS = w - 0.2
t, R_weak, Xv, Xm = SimNet.Run_MismatchDetection(2,w,SimDur)

w = -0.9 # w = wSV, wVS = w - 0.2
t, R_strong, Xv, Xm = SimNet.Run_MismatchDetection(2,w,SimDur)

# plot data
PltFcs.Plot_6E(t,R_weak,R_strong,SimDur,fig_x=6.0)
