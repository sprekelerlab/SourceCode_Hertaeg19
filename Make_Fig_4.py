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

# %% Fig 4 A - left

SimDur = 5000.0                                         # simulation time (ms)
w = np.arange(-0.1,-1.5,-0.02)                          # mutual inhibition tested
ada = np.arange(0.0,1.0,0.02)                           # adaptation strength tested
Para = [10.0, 50.0]                                     # tau, tau_w

# run simulations
Osc, WTA = SimNet.BifAna_Ada(ada,w,SimDur)

# plot data
PltFcs.Plot_4A_L(w,ada,Osc,WTA,Para,fig_x=7.0,fig_y=6.0)

# %% Fig 4 A - right

SimDur = 5000.0                                         # simulation time (ms)
tcw = 50.0                                              # adaptation time constant     

# run simulations
w, ada = -0.7, 0.5 # mutual inhibition, adaptation strength 
t, rS1, rV1 = SimNet.RateTrace(ada,tcw,w,SimDur)

w, ada = -1.3, 0.2 # mutual inhibition, adaptation strength 
t, rS2, rV2 = SimNet.RateTrace(ada,tcw,w,SimDur)

w, ada = -1.3, 0.35 # mutual inhibition, adaptation strength 
t, rS3, rV3 = SimNet.RateTrace(ada,tcw,w,SimDur)

w, ada = -1.3, 0.7 # mutual inhibition, adaptation strength 
t, rS4, rV4 = SimNet.RateTrace(ada,tcw,w,SimDur)

# plot data
PltFcs.Plot_4A_R(t, rS1,rV1,rS2,rV2,rS3,rV3,rS4,rV4,fig_x=7.0,fig_y=6.0)

# %% Fig 4 B

SimDur = 10000.0                                        # simulation time (ms)
w = np.array([-1.4,-1.3])                               # mutual inhibition tested

# run simulations
b = np.arange(0.3,2.1,0.1) # adaptation strength
fout_b_1 = SimNet.OscFreq(0,w[0],b,SimDur)

tcw = np.linspace(30,150,13) # adaptation time constant
fout_tcw_1 = SimNet.OscFreq(1,w[0],tcw,SimDur)

b = np.arange(0.3,2.1,0.1) # adaptation strength
fout_b_2 = SimNet.OscFreq(0,w[1],b,SimDur)

tcw = np.linspace(30,150,13) # adaptation time constant
fout_tcw_2 = SimNet.OscFreq(1,w[1],tcw,SimDur)   

# plot data
PltFcs.Plot_4B(w,b,tcw,fout_b_1,fout_b_2,fout_tcw_1,fout_tcw_2,fig_x=7.0)

# %% Fig 4 C

SimDur = 5000.0                                         # simulation time (ms)
u, tf = 0.1, 100                                        # STF parameter: Us, facilitation time constant
w = np.arange(-0.1,-1.5,-0.02)                          # mutual inhibition tested
ada = np.arange(0.0,1.0,0.02)                           # adaptation strength tested
Para = [25.0, 10.0, 50.0]                               # mean input, tau, tau_w

# run simulations
Osc, WTA, RS, RV = SimNet.BifAna_AdaSTF(ada,w,u,tf,SimDur)

# plot data
PltFcs.Plot_4C(w,ada,u,tf,Para,Osc,WTA,RS,RV,fig_x=7.0,fig_y=6.0)


