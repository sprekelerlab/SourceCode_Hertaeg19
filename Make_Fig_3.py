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

# %% Fig 3 A

SimDur = 2000.0                                         # simulation time (ms)
T_in = np.arange(50.0,650.0,50.0)                       # oscillation period tested
rec = np.arange(-0.1,-1.0,-0.2)                         # recurrence strength
w = -0.8                                                # fixed mutual inhibition 
A0 = 1.0                                                # amplidtude of input oscillation

# run simulations
Aout, Aout_ref = SimNet.FreqRespAna_Rec(rec, T_in, w, SimDur)

# plot data
PltFcs.Plot_3A(T_in,Aout,Aout_ref,fig_x=6.0)

# %% Fig 3 B and C

SimDur = 3000.0                                         # simulation time (ms)
T_in = np.arange(50.0,650.0,50.0)                       # oscillation period tested
ada = np.arange(0.1,1.0,0.2)                            # adaptation strength tested 
w = -0.8                                                # fixed mutual inhibition 
A0 = 1.0                                                # amplidtude of input oscillation

# run simulations
Aout, Aout_ref = SimNet.FreqRespAna_Ada(ada, T_in, w, SimDur)

# plot data
PltFcs.Plot_3B(T_in,Aout,Aout_ref,fig_x=6.0)
PltFcs.Plot_3C(ada,T_in,Aout,Aout_ref,fig_x=6.0)

# %% Fig 3 D

SimDur = 3000.0                                         # simulation time (ms)
ada = np.arange(0.0,2.0,0.1)                            # adaptation strength tested 
rec = np.arange(-0.0,-2.0,-0.1)                         # recurrence strength tested
w = -0.8                                                # fixed mutual inhibition 
ix = 10                                                 # index for which the correlation structure is shown

# run simulations
Corr_rec, SEM_rec, CMtx_rec = SimNet.CompCorr_Rec(rec, w, SimDur, ix)
Corr_ada, SEM_ada, CMtx_ada = SimNet.CompCorr_Ada(ada, w, SimDur, ix)

# plot data
PltFcs.Plot_3D(ada,rec,Corr_ada,SEM_ada,Corr_rec,SEM_rec,CMtx_ada,CMtx_rec,fig_x=10.5) 

# %% Fig 3 E and F

SimDur = 3000.0                                         # simulation time (ms)
T_in = np.arange(50.0,650.0,50.0)                       # oscillation period tested

# run simulations
w = -0.8                            # fixed mutual inhibition 
ada = np.arange(0.4,1.0,0.05)       # adaptation strength tested 
Aout_a, Aout_ref_a = SimNet.FreqRespAna_Ada(ada, T_in, w, SimDur)

b = 0.8                             # fixed adaptation strength
w_all = -np.arange(0.4,1.0,0.05)    # mutual inhibition tested
Aout, Aout_ref = SimNet.FreqRespAna_MutInhAda(w_all, T_in, b, SimDur)

# plot data
PltFcs.Plot_3EF(ada,w_all,Aout_a,Aout_ref_a,Aout,Aout_ref,T_in,fig_x=10.5,fig_y=7.0)
