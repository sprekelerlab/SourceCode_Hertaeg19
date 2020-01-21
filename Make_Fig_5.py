# -*- coding: utf-8 -*-
"""
@author: L. Hertaeg

"""

# %% import packages and functions

import sys
sys.path.append("./Functions")
import numpy as np

import PlotFuncs as PltFcs
#import SimNets_INnet as SimNet_1
import SimNets_FullMC as SimNet_2

# %% Fig 5 B

SimDur = 4000                           # simulation time (ms)
w = -1.1                                # fixed mutual inhibition
x_mod = np.linspace(-1,1,20)            # range of modulatory inputs tested

# run simulations
A, B = SimNet_2.RelTransInpStr_Perm(x_mod,w,SimDur)

# plot data
PltFcs.Plot_5B(x_mod,A,B,fig_x=5.5)

# %% Fig 5 C

SimDur = 12000                          # simulation time (ms)
w = -1.2                                # fixed mutual inhibition

# run simulations
t, R, x, y, A, B = SimNet_2.RelTransInpStr_Imp(w,SimDur)

# plot data
PltFcs.Plot_5C(t,R,x,y,A,B,SimDur,fig_x=12.0,fig_y=8.0)

