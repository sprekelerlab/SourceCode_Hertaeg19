# -*- coding: utf-8 -*-
"""
@author: L. Hertaeg
"""

# %% import packages and functions

import os
import numpy as np
import pandas as pd
from detect_peaks import detect_peaks 

import NumInt_LinRecNeuMod_N as LinNetMod
import NumInt_LinRecNeuModSTP_N as LinNetMod_STF
import NumInt_LinRecAdaNeuMod_N as LinNetMod_Ada
import NumInt_LinRecAdaNeuModSTP_N as LinNetMod_AdaSTF
from Functions_INRateModel import  SetConnectivity, SetPlasticityParameters, SetDefaultPara

# %%

def RunSig(flg,w,SimDur):
    
    #  flg: flag to indicate if reference (0) or full (1) network is simulated
    #  w: mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define modulatory input onto VIP neurons
    inp0 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(20),num=30),2))
    inp1 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(40),num=50),2))
    I_V = np.concatenate((-inp0,np.array([0.0]),inp1))
    I_V = np.sort(I_V)
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon)
    ModPar = {'T': T}
 
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    if flg==0:
        IS = r0[2]
        IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]
        Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
        Ir[sum(NC[0:3]):N,:] = -20.0 # to ensure that VIPs are silent (knocked-out)
    else:
        IV = r0[3] - w*r0[2]
        IS = r0[2] - w*r0[3]
        IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1] 
        Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
        
    ### Run network        
    rates = np.zeros((len(I_V),4))
    print 'Run network'
    
    for j in xrange(len(I_V)):
        
        print 'Processing: ' + str(np.round(100.0*j/len(I_V),1)) + '%'         
       
        # set modulatory input
        if flg==1: 
            Ir[sum(NC[0:3]):N,:] = IV + I_V[j]
        elif flg==0:
            Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS - I_V[j]
        InpPar = {'It':It, 'Ir':Ir}
        
        # run simulation
        LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
        R = arr[:,1:N+1]
  
        rates[j,0] = np.mean(R[-1,0:NC[0]]) # rE
        rates[j,1] = np.mean(R[-1,NC[0]:sum(NC[0:2])]) # rP
        rates[j,2] = np.mean(R[-1,sum(NC[0:2]):sum(NC[0:3])]) # rS
        rates[j,3] = np.mean(R[-1,sum(NC[0:3]):sum(NC[0:4])]) # rV
    
    os.remove('Data_LinRecNeuMod.dat')
    return I_V, rates
    

def RunAmpFac_w(w,SimDur,Fix=None):
    
    #  w: array of mutual inhibition weights
    #  SimDur: simulation time (ms)
    #  Fix: [Index, fixed weight]; Index: 1=wSV, 2=wVS
    
    print 'Set parameters' 
    
    ### Define modulatory input onto VIP neurons
    inp0 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(20),num=30),2))
    inp1 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(40),num=50),2))
    I_V = np.concatenate((-inp0,np.array([0.0]),inp1))
    I_V = np.sort(I_V)
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    ModPar = {'T': T}
 
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1] 
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
        
    ### Run network        
    rates = np.zeros((len(w),len(I_V),4))
    print 'Run network'
    
    for i in xrange(len(w)):
        
        w0 = w[i]
        if Fix==None:
            W = SetConnectivity(w0,w0,0.0,0.0,M,NC,NCon)
            IV = r0[3] - w0*r0[2]
            IS = r0[2] - w0*r0[3]
        else:
            w_fix = Fix[1]
            if Fix[0]==1:
                W = SetConnectivity(w_fix,w0,0.0,0.0,M,NC,NCon)
                IV = r0[3] - w0*r0[2]
                IS = r0[2] - w_fix*r0[3]
            elif Fix[0]==2:
                W = SetConnectivity(w0,w_fix,0.0,0.0,M,NC,NCon)
                IV = r0[3] - w_fix*r0[2]
                IS = r0[2] - w0*r0[3]
                
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
        
        for j in xrange(len(I_V)):
            
            print 'Processing: ' + str(np.round(100.0*(i*len(I_V)+j)/(len(I_V)*len(w)),2)) + '%'         
           
            # set modulatory input 
            Ir[sum(NC[0:3]):N,:] = IV + I_V[j]
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation
            LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
            R = arr[:,1:N+1]
      
            rates[i,j,0] = np.mean(R[-1,0:NC[0]]) # rE
            rates[i,j,1] = np.mean(R[-1,NC[0]:sum(NC[0:2])]) # rP
            rates[i,j,2] = np.mean(R[-1,sum(NC[0:2]):sum(NC[0:3])]) # rS
            rates[i,j,3] = np.mean(R[-1,sum(NC[0:3]):sum(NC[0:4])]) # rV
    
    os.remove('Data_LinRecNeuMod.dat')    
    return I_V, rates
    
    
def RunAmpFac_Rec(flg,wr,w,SimDur,Fix=None):
    
    #  flg: flag to indicate if reference (0) or full (1) network is simulated
    #  wr: array of recurrence values
    #  w: mutual inhibition
    #  SimDur: simulation time (ms)
    #  Fix: [Index, fixed weight]; Index: 1=wVV, 2=wSS
    
    print 'Set parameters' 
    
    ### Define modulatory input onto VIP neurons
    inp0 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(20),num=30),2))
    inp1 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(40),num=50),2))
    I_V = np.concatenate((-inp0,np.array([0.0]),inp1))
    I_V = np.sort(I_V)
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(1,NC)    
    ModPar = {'T': T}
 
    ### Define background (external) input
    InpPar = {}
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1] 
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
        
    ### Run network        
    rates = np.zeros((len(wr),len(I_V),4))
    print 'Run network'
    
    for i in xrange(len(wr)):
        
        rec = wr[i]
        if Fix==None:
            W = SetConnectivity(w,w,rec,rec,M,NC,NCon)        
            if flg==0:
                Ir[sum(NC[0:3]):N,:] = -20.0 
                IS = r0[2] - rec*r0[2]
            else:
                IV = r0[3] - w*r0[2] - rec*r0[3]
                IS = r0[2] - w*r0[3] - rec*r0[2]
                Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
        else:
            w_fix = Fix[1]
            if Fix[0]==1:
                W = SetConnectivity(w,w,rec,w_fix,M,NC,NCon)
                IV = r0[3] - w*r0[2] - w_fix*r0[3]
                IS = r0[2] - w*r0[3] - rec*r0[2]
            elif Fix[0]==2:
                W = SetConnectivity(w,w,w_fix,rec,M,NC,NCon)
                IV = r0[3] - w*r0[2] - rec*r0[3]
                IS = r0[2] - w*r0[3] - w_fix*r0[2]
            Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
                                                 
        for j in xrange(len(I_V)):
            
            print 'Processing: ' + str(np.round(100.0*(i*len(I_V)+j)/(len(I_V)*len(wr)),2)) + '%'         
           
            # set modulatory input 
            if flg==0:
                Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS - I_V[j] 
            else:
                Ir[sum(NC[0:3]):N,:] = IV + I_V[j]
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation
            LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
            R = arr[:,1:N+1]
      
            rates[i,j,0] = np.mean(R[-1,0:NC[0]]) # rE
            rates[i,j,1] = np.mean(R[-1,NC[0]:sum(NC[0:2])]) # rP
            rates[i,j,2] = np.mean(R[-1,sum(NC[0:2]):sum(NC[0:3])]) # rS
            rates[i,j,3] = np.mean(R[-1,sum(NC[0:3]):sum(NC[0:4])]) # rV
    
    os.remove('Data_LinRecNeuMod.dat')      
    return I_V, rates
    
    
def RunAmpFac_STF(w,STP,SimDur):
    
    #  w: array of mutual inhibition values
    #  STP: STF parameter for SOM->VIP and VIP->SOM (Us, tau_f)
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define modulatory input onto VIP neurons
    inp0 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(20),num=30),2))
    inp1 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(40),num=50),2))
    I_V = np.concatenate((-inp0,np.array([0.0]),inp1))
    I_V = np.sort(I_V)
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    ModPar = {'T': T}
    
    u, tf = STP[0], STP[1]
    Us, Tf = SetPlasticityParameters(u,tf,NC)
    STPar = {'Us':Us, 'Tf':Tf}
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1] 
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    u23 = (1+tf/1000.0*r0[3]) / (1+u*tf/1000.0*r0[3])
    u32 = (1+tf/1000.0*r0[2]) / (1+u*tf/1000.0*r0[2])
    
    ### Run network        
    rates = np.zeros((len(w),len(I_V),4))
    print 'Run network'
    
    for i in xrange(len(w)):
                 
        w0 = w[i]
        W = SetConnectivity(w0,w0,0.0,0.0,M,NC,NCon) 
        W = W/Us
         
        IV = r0[3] - w0*u32*r0[2]
        IS = r0[2] - w0*u23*r0[3]
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
        
        for j in xrange(len(I_V)):
            
            print 'Processing: ' + str(np.round(100.0*(i*len(I_V)+j)/(len(I_V)*len(w)),2)) + '%' 
            
            Ir[sum(NC[0:3]):N,:] = IV + I_V[j]
            InpPar = {'It':It, 'Ir':Ir}
            
            LinNetMod_STF.run(W, ModPar, STPar, InpPar, SimPar, np.zeros(N))
            arr = np.loadtxt('Data_LinRecNeuModSTP.dat',delimiter=' ')
            R = arr[:,1:N+1]
            
            rates[i,j,0] = np.mean(R[-1,0:NC[0]]) # rE
            rates[i,j,1] = np.mean(R[-1,NC[0]:sum(NC[0:2])]) # rP
            rates[i,j,2] = np.mean(R[-1,sum(NC[0:2]):sum(NC[0:3])]) # rS
            rates[i,j,3] = np.mean(R[-1,sum(NC[0:3]):sum(NC[0:4])]) # rV
    
    os.remove('Data_LinRecNeuModSTP.dat')          
    return I_V, rates

                
def FreqRespAna_Rec(rec, T_in, w, SimDur):
    
    #  rec: recurrence tested
    #  T_in: oscillation periods tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters'
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(1,NC)    
    ModPar = {'T': T}
           
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    T_trans = 750.0  
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]         
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    ### Run full IN network  
    Aout = np.zeros((len(rec),len(T_in)))
    print 'Run full IN network'
    
    for j in xrange(len(rec)):
        
        wr = rec[j]
        W = SetConnectivity(w,w,wr,wr,M,NC,NCon) 
        
        IV = r0[3] - w*r0[2] - wr*r0[3]
        IS = r0[2] - w*r0[3] - wr*r0[2]
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
        
        for i in xrange(len(T_in)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(rec)),2)) + '%'
                
            Ir[sum(NC[0:3]):N,:] = IV + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to VIP neurons
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation
            LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
            
            DI = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            SI = np.mean(R[:,NC[0]:sum(NC[0:2])],1)            
            DSI = SI-DI

            th = np.where(t==T_trans) # cut out transient phase
            idx_max = detect_peaks(DSI, mph=np.mean(DSI[t>T_trans]))
            idx_max = idx_max[(idx_max>th[0])]   
            idx_min = detect_peaks(-DSI, mph=np.mean(-DSI[t>T_trans]))
            idx_min = idx_min[(idx_min>th[0])]
            
            if np.var(DSI[t>T_trans])>1e-3:
                Aout[j,i] = (np.mean(DSI[idx_max])-np.mean(DSI[idx_min]))/2.0
            else:
                Aout[j,i] = np.nan
                
    ### Run reference IN network  
    Aout_ref = np.zeros((len(rec),len(T_in)))
    print 'Run reference IN network'
    Ir[sum(NC[0:3]):N,:] = -20.0 # no VIPs
    
    for j in xrange(len(rec)):
        
        wr = rec[j]
        W = SetConnectivity(0.0,0.0,wr,wr,M,NC,NCon) 
        IS = (1 - wr)*r0[2]
        
        for i in xrange(len(T_in)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(rec)),2)) + '%'
                
            Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to SOM neurons
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation
            LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
            
            DI = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            SI = np.mean(R[:,NC[0]:sum(NC[0:2])],1)            
            DSI = SI-DI

            th = np.where(t==T_trans) # cut out transient phase
            idx_max = detect_peaks(DSI, mph=np.mean(DSI[t>T_trans]))
            idx_max = idx_max[(idx_max>th[0])]   
            idx_min = detect_peaks(-DSI, mph=np.mean(-DSI[t>T_trans]))
            idx_min = idx_min[(idx_min>th[0])]
            
            if np.var(DSI[t>T_trans])>1e-3:
                Aout_ref[j,i] = (np.mean(DSI[idx_max])-np.mean(DSI[idx_min]))/2.0
            else:
                Aout_ref[j,i] = np.nan
     
    os.remove('Data_LinRecNeuMod.dat')           
    return Aout, Aout_ref
    
    
def FreqRespAna_Ada(ada, T_in, w, SimDur):
    
    #  ada: adaptation strengths tested
    #  T_in: oscillation periods tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [100.0] * NC[2] + [100.0] * NC[3]))
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    T_trans = 1000.0  
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]         
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    ### Run full IN network  
    Aout = np.zeros((len(ada),len(T_in)))
    print 'Run full IN network'
    
    for j in xrange(len(ada)):
        
        b = ada[j]
        W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon) 
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
        ModPar = {'T': T,'Ta':Ta,'B':B}
        
        IV = r0[3] - w*r0[2] + b*r0[3]
        IS = r0[2] - w*r0[3] + b*r0[2]
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
                
        for i in xrange(len(T_in)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(ada)),2)) + '%'
                
            Ir[sum(NC[0:3]):N,:] = IV + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to VIP neurons
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation            
            LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
            
            DI = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            SI = np.mean(R[:,NC[0]:sum(NC[0:2])],1)            
            DSI = SI-DI

            th = np.where(t==T_trans) # cut out transient phase
            idx_max = detect_peaks(DSI, mph=np.mean(DSI[t>T_trans]))
            idx_max = idx_max[(idx_max>th[0])]   
            idx_min = detect_peaks(-DSI, mph=np.mean(-DSI[t>T_trans]))
            idx_min = idx_min[(idx_min>th[0])]
            
            if np.var(DSI[t>T_trans])>1e-3:
                Aout[j,i] = (np.mean(DSI[idx_max])-np.mean(DSI[idx_min]))/2.0
            else:
                Aout[j,i] = np.nan
                
    ### Run reference IN network  
    Aout_ref = np.zeros((len(ada),len(T_in)))
    print 'Run reference IN network'
    Ir[sum(NC[0:3]):N,:] = -20.0 # no VIPs
    
    for j in xrange(len(ada)):
        
        b = ada[j]
        W = SetConnectivity(0.0,0.0,0.0,0.0,M,NC,NCon) 
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
        ModPar = {'T': T,'Ta':Ta,'B':B}
        IS = (1 + b)*r0[2]

        for i in xrange(len(T_in)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(ada)),2)) + '%'
                
            Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to SOM neurons
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation            
            LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
            
            DI = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            SI = np.mean(R[:,NC[0]:sum(NC[0:2])],1)            
            DSI = SI-DI

            th = np.where(t==T_trans) # cut out transient phase
            idx_max = detect_peaks(DSI, mph=np.mean(DSI[t>T_trans]))
            idx_max = idx_max[(idx_max>th[0])]   
            idx_min = detect_peaks(-DSI, mph=np.mean(-DSI[t>T_trans]))
            idx_min = idx_min[(idx_min>th[0])]
            
            if np.var(DSI[t>T_trans])>1e-3:
                Aout_ref[j,i] = (np.mean(DSI[idx_max])-np.mean(DSI[idx_min]))/2.0
            else:
                Aout_ref[j,i] = np.nan
    
    os.remove('Data_LinRecAdaNeuMod.dat')              
    return Aout, Aout_ref


def CompCorr_Rec(rec, w, SimDur, ix):
    
    #  rec: recurrences tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    #  ix: index for which the full correlation structure is shown
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 50, 50, 50 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(1,NC)    
    ModPar = {'T': T}
    l = 0.6
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],5000)
    Ir = np.zeros((N,len(It)))
    tini = 1000.0  
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]         
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    ### Run network  
    CorrPop = np.zeros(len(rec))
    SEM = np.zeros(len(rec))
    print 'Run network'    
    
    for j in xrange(len(rec)):
        
        print 'Processing: ' + str(np.round(100.0*j/len(rec),1)) + '%'
        
        wr = rec[j]
        W = SetConnectivity(w,w,wr,wr,M,NC,NCon) 
        
        np.random.seed(132) 
        IV = r0[3] - w*r0[2] - wr*r0[3]
        IS = r0[2] - w*r0[3] - wr*r0[3]    
        I_V = IV + l*np.random.normal(0,IV,size=(NC[3],len(It)))
        I_S = IS + l*np.random.normal(0,IS,size=(NC[2],len(It)))
        Ir[sum(NC[0:3]):N,:] = I_V + (1-l) * np.random.normal(0,IV,size=len(It)) 
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = I_S + (1-l) * np.random.normal(0,IS,size=len(It)) 
        InpPar = {'It':It, 'Ir':Ir}
        
        # run simulation
        LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
        t, R = arr[:,0], arr[:,1:N+1]
        
        RS = R[t>tini,sum(NC[0:2]):sum(NC[0:3])]
        Mtx = R[t>tini, sum(NC[0:2]):N]
        
        if j==ix:
            data = pd.DataFrame(Mtx)
            C = data.corr()
        dataS = pd.DataFrame(RS)
        CS = dataS.corr()
        CMtx = CS.as_matrix()
        
        id = np.triu_indices(len(CMtx),k=1)
        CorrPop[j] = np.mean(CMtx[id])
        SEM[j] = np.std(CMtx[id])/np.sqrt(len(CMtx[id]))
       
    
    os.remove('Data_LinRecNeuMod.dat')                
    return CorrPop, SEM, C
    

def CompCorr_Ada(ada, w, SimDur, ix):
    
    #  ada: adaptation strengths tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    #  ix: index for which the full correlation structure is shown
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 50, 50, 50 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [100.0] * NC[2] + [100.0] * NC[3]))    
    l = 0.6
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],5000)
    Ir = np.zeros((N,len(It)))
    tini = 1000.0  
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]         
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    ### Run network  
    CorrPop = np.zeros(len(ada))
    SEM = np.zeros(len(ada))
    print 'Run network'    
    
    for j in xrange(len(ada)):
        
        print 'Processing: ' + str(np.round(100.0*j/len(ada),1)) + '%'
        
        b = ada[j]
        W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon) 
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
        ModPar = {'T': T,'Ta':Ta,'B':B} 
        
        np.random.seed(132)         
        IV = r0[3] - w*r0[2] + b*r0[3]
        IS = r0[2] - w*r0[3] + b*r0[3]    
        I_V = IV + l*np.random.normal(0,IV,size=(NC[3],len(It)))
        I_S = IS + l*np.random.normal(0,IS,size=(NC[2],len(It)))
        Ir[sum(NC[0:3]):N,:] = I_V + (1-l) * np.random.normal(0,IV,size=len(It)) 
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = I_S + (1-l) * np.random.normal(0,IS,size=len(It)) 
        InpPar = {'It':It, 'Ir':Ir}
        
        # run simulation
        LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
        t, R = arr[:,0], arr[:,1:N+1]
        
        RS = R[t>tini,sum(NC[0:2]):sum(NC[0:3])]
        Mtx = R[t>tini, sum(NC[0:2]):N]
        
        if j==ix:
            data = pd.DataFrame(Mtx)
            C = data.corr()
        dataS = pd.DataFrame(RS)
        CS = dataS.corr()
        CMtx = CS.as_matrix()
        
        id = np.triu_indices(len(CMtx),k=1)
        CorrPop[j] = np.mean(CMtx[id])
        SEM[j] = np.std(CMtx[id])/np.sqrt(len(CMtx[id]))
       
     
    os.remove('Data_LinRecAdaNeuMod.dat')            
    return CorrPop, SEM, C
    
    
def FreqRespAna_MutInhAda(w_all, T_in, b, SimDur):
    
    #  w_all: mutual inhibition tested
    #  T_in: oscillation period tested
    #  b: fixed adaptation strength
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [100.0] * NC[2] + [100.0] * NC[3]))
    B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
    ModPar = {'T': T,'Ta':Ta,'B':B}
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    T_trans = 1000.0  
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]         
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[0:NC[0]] = -20.0 # to ensure that PC's are silent (knocked-out)
    
    ### Run full IN network  
    Aout = np.zeros((len(w_all),len(T_in)))
    print 'Run full IN network'
    
    for j in xrange(len(w_all)):
        
        w = w_all[j]
        W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon) 
        
        IV = r0[3] - w*r0[2] + b*r0[3]
        IS = r0[2] - w*r0[3] + b*r0[2]
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
                
        for i in xrange(len(T_in)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(w_all)),2)) + '%'
                
            Ir[sum(NC[0:3]):N,:] = IV + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to VIP neurons
            # Ir[np.where(Ir<0)] = 0.0
            InpPar = {'It':It, 'Ir':Ir}
            
            # run simulation            
            LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
            
            DI = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            SI = np.mean(R[:,NC[0]:sum(NC[0:2])],1)            
            DSI = SI-DI

            th = np.where(t==T_trans) # cut out transient phase
            idx_max = detect_peaks(DSI, mph=np.mean(DSI[t>T_trans]))
            idx_max = idx_max[(idx_max>th[0])]   
            idx_min = detect_peaks(-DSI, mph=np.mean(-DSI[t>T_trans]))
            idx_min = idx_min[(idx_min>th[0])]
            
            if np.var(DSI[t>T_trans])>1e-3:
                Aout[j,i] = (np.mean(DSI[idx_max])-np.mean(DSI[idx_min]))/2.0
            else:
                Aout[j,i] = np.nan
                
    ### Run reference IN network  
    Aout_ref = np.zeros((len(w_all),len(T_in)))
    print 'Run reference IN network'

    W = SetConnectivity(0.0,0.0,0.0,0.0,M,NC,NCon) 
    Ir[sum(NC[0:3]):N,:] = -20.0
    IS = (1 + b)*r0[2]
            
    for i in xrange(len(T_in)):
        
        print 'Processing: ' + str(np.round(100.0*(len(T_in)+i)/(len(T_in)*len(w_all)),2)) + '%'
            
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to SOM neurons
        # Ir[np.where(Ir<0)] = 0.0
        InpPar = {'It':It, 'Ir':Ir}
        
        # run simulation            
        LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
        t, R = arr[:,0], arr[:,1:N+1]
        
        DI = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
        SI = np.mean(R[:,NC[0]:sum(NC[0:2])],1)            
        DSI = SI-DI

        th = np.where(t==T_trans) # cut out transient phase
        idx_max = detect_peaks(DSI, mph=np.mean(DSI[t>T_trans]))
        idx_max = idx_max[(idx_max>th[0])]   
        idx_min = detect_peaks(-DSI, mph=np.mean(-DSI[t>T_trans]))
        idx_min = idx_min[(idx_min>th[0])]
        
        if np.var(DSI[t>T_trans])>1e-3:
            Aout_ref[:,i] = (np.mean(DSI[idx_max])-np.mean(DSI[idx_min]))/2.0
        else:
            Aout_ref[:,i] = np.nan
         
    os.remove('Data_LinRecAdaNeuMod.dat') 
    return Aout, Aout_ref
    

def BifAna_Ada(ada,w,SimDur):
    
    #  ada: adaptation strength tested
    #  w: mutual inhibition tested
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 1, 1, 10, 10
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [50.0] * NC[2] + [50.0] * NC[3]))
    
    ### Adjust connection weights and probabilities
    M = np.array([[0.0,0.0,0,0],[0.0,0.0,0.0,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    P = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]])
    NCon = np.round(P*NC).astype(np.int64)
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    InMean, InSTD = 25.0, 5.0
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Tini = 3000.0  
           
    Ir[0:NC[0]] = -20.0                         # to ensure that PC's are silent (knocked-out)
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = -20.0     # to ensure that PV's are silent (knocked-out)
    Ir[sum(NC[0:2]):sum(NC[0:4]),:] = np.random.normal(InMean,InSTD,size=(NC[2]+NC[3],len(It)))
    InpPar = {'It':It, 'Ir':Ir}
    
    ### Run network  
    Osc, WTA = np.zeros((len(ada),len(w))), np.zeros((len(ada),len(w)))

    print 'Run network'
    
    for j in xrange(len(ada)):
        
        b = ada[j]
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
        ModPar = {'T': T,'Ta':Ta,'B':B}
        
        for i in xrange(len(w)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(w)+i)/(len(w)*len(ada)),2)) + '%'
            
            w0 = w[i]
            W = SetConnectivity(w0,w0,0.0,0.0,M,NC,NCon) 
        
            # run simulation            
            LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
      
            rS = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            rV = np.mean(R[:,sum(NC[0:3]):sum(NC[0:4])],1)
            
            rS_std, rS_mean = np.std(rS[t>Tini]), np.mean(rS[t>Tini])
            rV_std, rV_mean = np.std(rV[t>Tini]), np.mean(rV[t>Tini])
            
            # criteria to check if WTA or osc WTA - chosen such that it agrees with with visual inspection of example response curves              
            if ((rS_std>0.5*InSTD) or (rV_std>0.5*InSTD)):
                Osc[j,i] = 1
            else:
                dr = np.abs(rS[-1] - rV[-1])
                if (dr > 0.2*np.max([rS_mean,rV_mean]) and (rS_mean<=2.5*rS_std or rV_mean<=2.5*rV_std)):
                    WTA[j,i] = 1
    
    os.remove('Data_LinRecAdaNeuMod.dat') 
    return Osc, WTA
    

def BifAna_Rec(rec,w,SimDur):
    
    #  rec: recurrence tested
    #  w: mutual inhibition tested
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 1, 1, 5, 5 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(1,NC)    
    ModPar = {'T': T}
    
    ### Adjust connection weights and probabilities
    M = np.array([[0.0,0.0,0,0],[0.0,0.0,0.0,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    P = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,1.0,1.0],[0.0,0.0,1.0,1.0]])
    NCon = np.round(P*NC).astype(np.int64)
    NCon[2,2]=NCon[2,2]-1 # no autapses
    NCon[3,3]=NCon[3,3]-1 # no autapses
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    InMean, InSTD = 25.0, 1.0
    It = np.linspace(SimPar[0],SimPar[1],10000)
    Ir = np.zeros((N,len(It)))
    Tini = 4000.0  
    
    Ir[0:NC[0]] = -20.0                         # to ensure that PC's are silent (knocked-out)
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = -20.0     # to ensure that PV's are silent (knocked-out)
    Ir[sum(NC[0:2]):sum(NC[0:4]),:] = np.random.normal(InMean,InSTD,size=(NC[2]+NC[3],len(It)))
    InpPar = {'It':It, 'Ir':Ir}
       
    ### Run network  
    NoAC_SOM, NoAC_VIP = np.zeros((len(rec),len(w))), np.zeros((len(rec),len(w)))

    print 'Run network'
    
    for j in xrange(len(rec)):
        
        for i in xrange(len(w)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(w)+i)/(len(w)*len(rec)),2)) + '%'
            
            w0 = w[i]
            W = SetConnectivity(w0,w0,rec[j],rec[j],M,NC,NCon) 
        
            # run simulation 
            LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
            
            RS = np.mean(R[t>Tini,sum(NC[0:2]):sum(NC[0:3])],0)
            RV = np.mean(R[t>Tini,sum(NC[0:3]):sum(NC[0:4])],0)
            
            NoAC_SOM[j,i] = sum(1*(RS>InSTD))
            NoAC_VIP[j,i] = sum(1*(RV>InSTD))

    
    os.remove('Data_LinRecNeuMod.dat') 
    return NoAC_SOM, NoAC_VIP
      
    
def BifAna_AdaSTF(ada,w,u,tf,SimDur):
    
    #  ada: adaptation strength tested
    #  w: mutual inhibition tested
    #  u: initial release probability (STF parameter)
    #  tf: facilitation time constant (STF parameter)
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 1, 1, 10, 10
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [50.0] * NC[2] + [50.0] * NC[3]))    
    
    ### Define STP parameters
    Us, Tf = SetPlasticityParameters(u,tf,NC)
    STPar = {'Us':Us, 'Tf':Tf}
    
    ### Adjust connection weights and probabilities
    M = np.array([[0.0,0.0,0,0],[0.0,0.0,0.0,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    P = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]])
    NCon = np.round(P*NC).astype(np.int64)
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    InMean, InSTD = 25.0, 5.0
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Tini = 3000.0  
    
    Ir[0:NC[0]] = -20.0                         # to ensure that PC's are silent (knocked-out)
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = -20.0     # to ensure that PV's are silent (knocked-out)
    Ir[sum(NC[0:2]):sum(NC[0:4]),:] = np.random.normal(InMean,InSTD,size=(NC[2]+NC[3],len(It)))
    InpPar = {'It':It, 'Ir':Ir}
    
    ### Run network  
    Osc, WTA = np.zeros((len(ada),len(w))), np.zeros((len(ada),len(w)))
    RS, RV = np.zeros((len(ada),len(w))), np.zeros((len(ada),len(w)))

    print 'Run network'
    
    for j in xrange(len(ada)):
        
        b = ada[j]
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
        ModPar = {'T': T,'Ta':Ta,'B':B}
        
        for i in xrange(len(w)):
            
            print 'Processing: ' + str(np.round(100.0*(j*len(w)+i)/(len(w)*len(ada)),2)) + '%'
            
            w0 = w[i]
            W = SetConnectivity(w0,w0,0.0,0.0,M,NC,NCon) 
            W = W/Us
        
            # run simulation  
            LinNetMod_AdaSTF.run(W, ModPar, STPar, InpPar, SimPar, np.zeros(N))
            arr=np.loadtxt('Data_LinRecAdaNeuModSTP.dat',delimiter=' ')
            t, R = arr[:,0], arr[:,1:N+1]
      
            rS = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
            rV = np.mean(R[:,sum(NC[0:3]):sum(NC[0:4])],1)
            
            rS_std, rS_mean = np.std(rS[t>Tini]), np.mean(rS[t>Tini])
            rV_std, rV_mean = np.std(rV[t>Tini]), np.mean(rV[t>Tini])
            
            # criteria to check if WTA or osc WTA - chosen such that it agrees with with visual inspection of example response curves              
            if ((rS_std>0.5*InSTD) or (rV_std>0.5*InSTD)):
                Osc[j,i] = 1
            else:
                dr = np.abs(rS[-1] - rV[-1])
                if (dr > 0.2*np.max([rS_mean,rV_mean]) and (rS_mean<=2.5*rS_std or rV_mean<=2.5*rV_std)):
                    WTA[j,i] = 1
                    
            RS[j,i], RV[j,i] = rS_mean, rV_mean
    
    os.remove('Data_LinRecAdaNeuModSTP.dat')     
    return Osc, WTA, RS, RV
   
   
def RateTrace(ada,tcw,w,SimDur,Fix=None): 
    
    #  ada: fixed adaptation strength
    #  tcw: fixed adaptation time constant
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    #  Fix: if 'None' parameters are symmetrical, 
    #       otherwise an array with 3 entries (
    #       First: 1/2 = adaptation strength/time constant asymmetric
    #       Second: 1/2 = SOM or VIP fixed
    #       Third: value to be taken)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 1, 1, 10, 10 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
       
    if Fix==None:
        Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [tcw] * NC[2] + [tcw] * NC[3]))
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [ada] * NC[2] + [ada] * NC[3]))
    else:
        if Fix[0]==1: # adaptation strength
            Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [50.0] * NC[2] + [50.0] * NC[3]))
            if Fix[1]==1:
                B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [Fix[2]] * NC[2] + [ada] * NC[3]))
            elif Fix[1]==2:
                B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [ada] * NC[2] + [Fix[2]] * NC[3]))
        elif Fix[0]==2: # adaptation time constant
            B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [ada] * NC[2] + [ada] * NC[3]))
            if Fix[1]==1:
                Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [Fix[2]] * NC[2] + [tcw] * NC[3]))
            elif Fix[1]==2:
                Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [tcw] * NC[2] + [Fix[2]] * NC[3]))             
    ModPar = {'T': T,'Ta':Ta,'B':B}
    
    ### Adjust connection weights and probabilities
    M = np.array([[0.0,0.0,0,0],[0.0,0.0,0.0,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    P = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]])
    NCon = np.round(P*NC).astype(np.int64)
    W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon) 
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    InMean, InSTD = 25.0, 5.0
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    
    Ir[0:NC[0]] = -20.0                         # to ensure that PC's are silent (knocked-out)
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = -20.0     # to ensure that PV's are silent (knocked-out)
    Ir[sum(NC[0:2]):sum(NC[0:4]),:] = np.random.normal(InMean,InSTD,size=(NC[2]+NC[3],len(It)))
    InpPar = {'It':It, 'Ir':Ir}
    
    # run simulation 
    print 'Run simulation'
           
    LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
    arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    rS = R[:,sum(NC[0:2]):sum(NC[0:3])]
    rV = R[:,sum(NC[0:3]):sum(NC[0:4])]
 
    os.remove('Data_LinRecAdaNeuMod.dat')  
    return t, rS, rV
    
    
def RateTrace_Rec(rec,w,SimDur):  
    
    #  rec: fixed recurrence
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 1, 1, 5, 5 # N_E, N_P, N_S, N_V
    N, T, r0, P, M, NCon = SetDefaultPara(1,NC)    
    ModPar = {'T': T}
    
    ### Adjust connection weights and probabilities
    M = np.array([[0.0,0.0,0,0],[0.0,0.0,0.0,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    P = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,1.0,1.0],[0.0,0.0,1.0,1.0]])
    NCon = np.round(P*NC).astype(np.int64)
    NCon[2,2]=NCon[2,2]-1 # no autapses
    NCon[3,3]=NCon[3,3]-1 # no autapses
    W = SetConnectivity(w,w,rec,rec,M,NC,NCon) 
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    InMean, InSTD = 25.0, 1.0
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    
    Ir[0:NC[0]] = -20.0                         # to ensure that PC's are silent (knocked-out)
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = -20.0     # to ensure that PV's are silent (knocked-out)
    Ir[sum(NC[0:2]):sum(NC[0:4]),:] = np.random.normal(InMean,InSTD,size=(NC[2]+NC[3],len(It)))
    InpPar = {'It':It, 'Ir':Ir}
    
    # run simulation 
    print 'Run simulation'
           
    LinNetMod.run(W, ModPar, InpPar, SimPar, np.zeros(N))
    arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    rS = R[:,sum(NC[0:2]):sum(NC[0:3])]
    rV = R[:,sum(NC[0:3]):sum(NC[0:4])]
 
    os.remove('Data_LinRecNeuMod.dat')  
    return t, rS, rV
    

def OscFreq(flg,w,p,SimDur):
    
    #  flg: flag indicating if adaptation strength (0) or time constant (1) is considered
    #  w: fixed mutual inhibition
    #  p: adaptation strengths or time constants tested
    #  SimDur: simulation time (ms)
    
    print 'Set parameters'
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 1, 1, 10, 10
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    
    ### Adjust connection weights and probabilities
    M = np.array([[0.0,0.0,0,0],[0.0,0.0,0.0,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    P = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]])
    NCon = np.round(P*NC).astype(np.int64)
    W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon) 
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    InMean, InSTD = 25.0, 5.0
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    Tini = 1000.0
    
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1]         
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[0:NC[0]] = -20.0                         # to ensure that PC's are silent (knocked-out)
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = -20.0     # to ensure that PV's are silent (knocked-out)
    Ir[sum(NC[0:2]):sum(NC[0:4]),:] = np.random.normal(InMean,InSTD,size=(NC[2]+NC[3],len(It)))
    InpPar = {'It':It, 'Ir':Ir} 
       
    ### Run network  
    fout = np.zeros(len(p))
    print 'Run network'
    
    for k in xrange(len(p)):
        
        print 'Processing: ' + str(np.round(100.0*k/len(p),1)) + '%'    
        
        if flg==0:
            B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [p[k]] * NC[2] + [p[k]] * NC[3]))
            Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [50.0] * NC[2] + [50.0] * NC[3]))
        else:
            B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [1.0] * NC[2] + [1.0] * NC[3]))
            Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [p[k]] * NC[2] + [p[k]] * NC[3]))
        ModPar = {'T': T,'Ta':Ta,'B':B}
           
        LinNetMod_Ada.run(W, ModPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecAdaNeuMod.dat',delimiter=' ')
        t, R = arr[:,0], arr[:,1:N+1]      
        
        rS = np.mean(R[:,sum(NC[0:2]):sum(NC[0:3])],1)
        #rV = np.mean(R[:,sum(NC[0:3]):sum(NC[0:4])],1)
             
        th = np.where(t==Tini) # cut out transient phase
        idx_max = detect_peaks(rS, mph=0.9*np.max(rS[t>Tini]))
        idx_max = idx_max[(idx_max>th[0])] 
        idx_min = detect_peaks(-rS, mph=0.9*np.max(-rS[t>Tini]))
        idx_min = idx_min[(idx_min>th[0])] 
        
        if np.var(rS[t>Tini])>1e-2:
            Amp = (np.mean(rS[idx_max])-np.mean(rS[idx_min]))/2.0
            tc = t[(np.roll(rS,-1)>Amp) & (rS<Amp)]   # t[(rS[1:]>Amp) & (rS[:-1]<Amp)]
            fout[k] = 1000.0/np.mean(np.diff(tc[tc>Tini]))
        else:
            fout[k] = np.nan   
    
    
    os.remove('Data_LinRecAdaNeuMod.dat')         
    return fout


def Hysteresis(x_mod,w,SimDur):
    
    #  x_mod: modulatory input tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters'
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 20, 10, 10, 10 
    N, T, r0, P, M, NCon = SetDefaultPara(0,NC)    
    ModPar = {'T': T}    
    
    ### Define connection weights and probabilities
    P = np.array([[0.1,0.6,0.55,0.0],[0.45,0.5,0.6,0.0],[0.35,0.0,0.0,0.5],[0.1,0.1,0.45,0.0]])
    M = np.array([[0.0,-1.5,0,0],[0.0,-1.5,-1.3,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    NCon = np.round(P*NC).astype(np.int64)
    W = SetConnectivity(w,w,0.0,0.0,M,NC,NCon)

    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    
    IV = r0[3] - w*r0[2]
    IS = r0[2] - w*r0[3]
    IP = r0[1] - M[1,2]*r0[2] - M[1,1]*r0[1] 
    Ir[sum(NC[0:1]):sum(NC[0:2]),:] = IP
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
    Ir[0:NC[0]] = -20.0

    InpPar = {'It':It, 'Ir':Ir}

    ### Run network  
    DI, SI = np.zeros((2,len(x_mod))), np.zeros((2,len(x_mod)))
    rini = np.zeros(N)
    
    print 'Run network - mod. input ascending'   
    for ii in range(len(x_mod)):
        
        print 'Processing: ' + str(np.round(100.0*ii/len(x_mod),1)) + '%'
        Ir[sum(NC[0:3]):N,:] = IV + x_mod[ii]

        LinNetMod.run(W, ModPar, InpPar, SimPar, rini)
        arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
        R = arr[:,1:N+1]
        rini = R[-1,:]
        
        DI[0,ii] = np.mean(R[-1:,sum(NC[0:2]):sum(NC[0:3])],1)
        SI[0,ii] = np.mean(R[-1:,NC[0]:sum(NC[0:2])],1)
    
    print 'Run network - mod. input descending'
    for ii in reversed(range(len(x_mod))):
        
        print 'Processing: ' + str(np.round(100.0*(len(x_mod)-ii)/len(x_mod),1)) + '%'
        Ir[sum(NC[0:3]):N,:] = IV + x_mod[ii]
        
        LinNetMod.run(W, ModPar, InpPar, SimPar, rini)
        arr=np.loadtxt('Data_LinRecNeuMod.dat',delimiter=' ')
        R = arr[:,1:N+1]
        rini = R[-1,:]
        
        DI[1,ii] = np.mean(R[-1:,sum(NC[0:2]):sum(NC[0:3])],1)
        SI[1,ii] = np.mean(R[-1:,NC[0]:sum(NC[0:2])],1)
    
    os.remove('Data_LinRecNeuMod.dat')    
    return SI, DI