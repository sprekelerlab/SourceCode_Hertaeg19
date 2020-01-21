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
import SimNets_FullMC as SimNet_MC

# %% Fig S1

SimDur = 2000.0     
wsv, wvs = -1.0, -1.0
w = np.arange(-0.1,-1.2,-0.1)

# run simulations
x_ref, r_ref = SimNet.RunSig(0,0.0,SimDur)                      # reference enetwork
x_sv, r_sv = SimNet.RunAmpFac_w(w,SimDur,Fix=[1,wsv])           # full interneuron network, wsv fixed
x_vs, r_vs = SimNet.RunAmpFac_w(w,SimDur,Fix=[2,wvs])           # full interneuron network, wvs fixed

# plot data
PltFcs.Plot_S1(w,x_ref,r_ref,x_sv,r_sv,x_vs,r_vs,fig_x=6.0)

# %% Fig S2

SimDur = 2000.0                                                         # simulation time (ms) 
wvv, wss = -0.5, -0.5                                                   # recurrence strength when fixed
wr = np.arange(-0.1,-1.0,-0.1)                                          # recurrence strength tested
w = -0.8                                                                # mutual inhibition strength

# run simulations
x0, r0 = SimNet.RunAmpFac_Rec(0,np.array([wss]),w,SimDur)               # reference network - SOM-SOM fixed
x_ref, r_ref = SimNet.RunAmpFac_Rec(0,wr,w,SimDur)                      # reference network
x_vv, r_vv = SimNet.RunAmpFac_Rec(1,wr,w,SimDur,Fix=[1,wvv])            # full interneuron network, wvv fixed
x_ss, r_ss = SimNet.RunAmpFac_Rec(1,wr,w,SimDur,Fix=[2,wss])            # full interneuron network, wss fixed

# plot data
PltFcs.Plot_S2(wr,x0,r0,x_ref,r_ref,x_vv,r_vv,x_ss,r_ss,fig_x=6.0)

# %% Fig S3

SimDur = 5000.0                                         # simulation time (ms)
tcw = 50.0                                              # fixed adaptation time constant
w = -1.3                                                # fixed mutual inhibition

# run simulations
RS_all, RV_all = {}, {}

aS = np.linspace(0.4,1.0,4)
aV = np.linspace(0.4,1.0,4)

for i in range(len(aS)):
    for j in range(len(aV)):
        ada = aV[j]
        Fix = [1,1,aS[i]]
        t, rS, rV = SimNet.RateTrace(ada,tcw,w,SimDur,Fix)
        RS_all[str(len(aV)*i+j)] = rS
        RV_all[str(len(aV)*i+j)] = rV
        
# plot data
PltFcs.Plot_S3(aS,aV,t,RS_all,RV_all,fig_x=12,fig_y=10)

# %% Fig S4

SimDur = 5000.0                                         # simulation time (ms)
ada = 0.5                                               # fixed adaptation strength
w = -1.3                                                # fixed mutual inhibition

# run simulations
RS_all, RV_all = {}, {}

TS = np.array([50.0,100.0,200.0,400.0])
TV = np.array([50.0,100.0,200.0,400.0])

for i in range(len(TS)):
    for j in range(len(TV)):
        tcw = TV[j]
        Fix = [2,1,TS[i]]
        t, rS, rV = SimNet.RateTrace(ada,tcw,w,SimDur,Fix)
        RS_all[str(len(TV)*i+j)] = rS
        RV_all[str(len(TV)*i+j)] = rV
        
# plot data
PltFcs.Plot_S4(TS,TV,t,RS_all,RV_all,fig_x=12,fig_y=10)

# %% Fig S5 - left

SimDur = 5000.0                                     # simulation time (ms)
w = np.concatenate((np.linspace(-0.1,-0.6,3),   
                    np.linspace(-0.61,-4.95,80),
                    np.linspace(-5.0,-6.0,3)))      # mutual inhibition tested
rec = np.concatenate((np.linspace(0.0,-3.85,50),
                      np.linspace(-3.9,-4.1,6),
                      np.linspace(-4.15,-5.0,3)))   # recurrence tested
N = 5                                               # 5 neurons per IN population    

# run simulations
NoAC_SOM, NoAC_VIP = SimNet.BifAna_Rec(rec,w,SimDur)

# plot data
PltFcs.Plot_S5_L(rec, w, N, NoAC_SOM, NoAC_VIP,fig_x=7.0,fig_y=6.0)

# %% Fig S5 - right

SimDur = 5000.0                                     # simulation time (ms)

# run simulations
w, rec = -1.0, -2.5 # mutual inhibition, recurrence
t, rS1, rV1 = SimNet.RateTrace_Rec(rec,w,SimDur)

w, rec = -4.5, -1.5 # mutual inhibition, recurrence
t, rS2, rV2 = SimNet.RateTrace_Rec(rec,w,SimDur)

w, rec = -2.0, -4.5 # mutual inhibition, recurrence
t, rS3, rV3 = SimNet.RateTrace_Rec(rec,w,SimDur)

w, rec = -5.5, -4.5 # mutual inhibition, recurrence
t, rS4, rV4 = SimNet.RateTrace_Rec(rec,w,SimDur)

# plot data
PltFcs.Plot_S5_R(t,rS1,rV1,rS2,rV2,rS3,rV3,rS4,rV4,fig_x=7.0,fig_y=7.0)

# %% Fig S6 A

flg = 1                                             # flag to indicate if reference (0) or full (1) network is simulated
SimDur = 4000.0                                     # simulation time (ms)

# run simulations
NeuPar = np.array([-0.7,0.0,0.2,0.4])               # array of neuron parameters (mutual inhibition, recurrence, adaptation, initial synaptic efficacy) 
x_ctr, r_ctr = SimNet_MC.RunSig(flg,NeuPar,SimDur)

NeuPar = np.array([-0.9,0.0,0.2,0.4])              
x_w, r_w = SimNet_MC.RunSig(flg,NeuPar,SimDur)

NeuPar = np.array([-0.7,-0.3,0.2,0.4])               
x_wr, r_wr = SimNet_MC.RunSig(flg,NeuPar,SimDur)

NeuPar = np.array([-0.7,0.0,0.2,0.25])               
x_STF, r_STF = SimNet_MC.RunSig(flg,NeuPar,SimDur)

NeuPar = np.array([-0.7,0.0,0.5,0.4])               
x_ada, r_ada = SimNet_MC.RunSig(flg,NeuPar,SimDur)

# plot data
PltFcs.Plot_S6A(x_ctr,r_ctr,x_w,r_w,x_wr,r_wr,x_STF,r_STF,x_ada,r_ada,fig_x=10.0,fig_y=9.0)

# %% Fig S6 B

SimDur = 3000.0                                     # simulation time (ms)
ada = np.arange(0.1,1.0,0.2)                        # adaptation strength tested
T_in = np.arange(50.0,650.0,50.0)                   # oscillation period tested
w0 = -0.8                                           # fixed mutual inhibition

# run simulations
Aout, Aout_ref = SimNet_MC.FreqRespAna_Ada(ada, T_in, w0, SimDur)

# plot data
PltFcs.Plot_S6B(ada,T_in,Aout,Aout_ref,fig_x=6.0)

# %% Fig S6 C

SimDur = 3000.0                                     # simulation time (ms)
FB = np.arange(0.0,2.0,0.1)                         # feedback strength (adaptation or recurrence)
w0 = -0.8                                           # fixed mutual inhibition

# run simulations
CorrPop_ada, SEM_ada = SimNet_MC.CompCorr(0, FB, w0, SimDur)
CorrPop_rec, SEM_rec = SimNet_MC.CompCorr(1, FB, w0, SimDur)

# plot data
PltFcs.Plot_S6C(FB, CorrPop_ada, SEM_ada, CorrPop_rec, SEM_rec, fig_x=6.0)

# %% Fig S7

SimDur = 2000.0                         # simulation time (ms)
w = -1.05                               # fixed mutual inhibition
x_IN = np.linspace(-0.5,0.5,50)         # range of modulatory inputs tested in IN net
x_full = np.linspace(-0.5,0.5,50)       # range of modulatory inputs tested in full net

# run simulations
SI, DI = SimNet.Hysteresis(x_IN,w,SimDur)
rates = SimNet_MC.Hysteresis(x_full,w,SimDur+1000.0)
     
# plot data
PltFcs.Plot_S7(w,x_IN,SI,DI,x_full,rates,fig_x=12,fig_y=12)