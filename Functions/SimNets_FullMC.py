# -*- coding: utf-8 -*-
"""
@author: L. Hertaeg
"""

# %% import packages and functions

import os
import numpy as np
import pandas as pd
from detect_peaks import detect_peaks
from scipy import interpolate

from Functions_CompRateModel import  SetConnectivity, SetPlasticityParameters, SetDefaultPara
import NumInt_LinRecNeuMod_2CompPC_Ada_STP_N as LinNetModAdaSTP
import NumInt_LinRecNeuMod_2CompPC_Ada_N as LinNetModAda
import NumInt_LinRecNeuMod_2CompPC_N as LinNetMod

# %% 

def RunSig(flg,NeuPar,SimDur):
    
    #  flg: flag to indicate if reference (0) or full (1) network is simulated
    #  NeuPar: array of neuron parameters (mutual inhibition, recurrence, adaptation, initial synaptic efficacy) 
    #  SimDur: simulation time (ms)

    print 'Set parameters'  
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10 # N_E, N_P, N_S, N_V
    w0, rec, ada, U = NeuPar[0], NeuPar[1], NeuPar[2], NeuPar[3]
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(1,NC,w0,rec)   
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)

    ### Define modulatory input onto VIP neurons
    inp0 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(6),num=40),2))
    inp1 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(6),num=40),2))
    I_V = np.concatenate((-inp0,np.array([0.0]),inp1))
    I_V = np.sort(I_V)
    
    ### Define STP
    iF = np.mat([[2,3],[3,2]])
    iD = np.mat([])
    
    UF, UD = np.ones((4,4)), np.zeros((4,4)) # arrays
    TF, TD = np.inf*np.ones((4,4)), np.inf*np.ones((4,4))
    
    if np.size(iF,1)>0:
        for ia in xrange(np.size(iF,0)):
            UF[iF[ia,0],iF[ia,1]] = U
            TF[iF[ia,0],iF[ia,1]] = 200.0
    
    if np.size(iD,1)>0:
        for ib in xrange(np.size(iD,0)):
            UD[iD[ib,0],iD[ib,1]] = U
            TD[iD[ib,0],iD[ib,1]] = 200.0
            
    Us, Ux, Tu, Tx = SetPlasticityParameters(UF,UD,TF,TD,NC)
    STPar = {'Us':Us, 'Ux':Ux, 'Tu':Tu, 'Tx':Tx}
    
    W = W/Us 
    Weights = {'W':W, 'WD': WD, 'WEI': WEI} 
    
    ### Define adaptation parameters
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [100.0] * NC[2] + [100.0] * NC[3]))
    B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [ada] * NC[2] + [ada] * NC[3]))
    ModPar = {'G':G,'H':H,'A':A,'T': T,'Ta':Ta, 'B':B}
    
    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
    
    IE, IB, IP, IS, IV = 250.0, 300.0, 3, 3, 3 # in pA, pA, Hz, Hz, Hz
    Ir[0:NC[0],:] = IE
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
    Ir[sum(NC[0:3]):N,:] = IV 
    ID[0:NC[0],:] = IB
      
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
            Ir[sum(NC[0:3]):N,:] = -20.0 # to knock-out VIPs
        InpPar = {'It':It, 'Ir':Ir, 'ID':ID}
        
        # run simulation
        LinNetModAdaSTP.run(Weights, ModPar, STPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecNeuMod_2CompPC_Ada_STP.dat',delimiter=' ')
        R = arr[:,1:N+1]
        
        rates[j,0] = np.mean(R[-1,0:NC[0]]) # rE
        rates[j,1] = np.mean(R[-1,NC[0]:sum(NC[0:2])]) # rP
        rates[j,2] = np.mean(R[-1,sum(NC[0:2]):sum(NC[0:3])]) # rS
        rates[j,3] = np.mean(R[-1,sum(NC[0:3]):sum(NC[0:4])]) # rV
        
    os.remove('Data_LinRecNeuMod_2CompPC_Ada_STP.dat')           
    return I_V, rates
              
              
def FreqRespAna_Ada(ada, T_in, w0, SimDur):
    
    #  ada: adaptation strength tested
    #  T_in: oscillation period tested
    #  w0: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters'
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(0,NC,w0)  
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [100.0] * NC[2] + [100.0] * NC[3]))
    ModPar = {'G':G,'H':H,'A':A,'T': T,'Ta':Ta}    
    
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
    Weights = {'W':W, 'WD': WD, 'WEI': WEI}

    ### Define modulatory input onto VIP neurons
    inp0 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(6),num=40),2))
    inp1 = np.unique(np.round(np.logspace(np.log10(0.01),np.log10(6),num=40),2))
    I_V = np.concatenate((-inp0,np.array([0.0]),inp1))
    I_V = np.sort(I_V)
    
    ### define external input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
    T_trans = 1000.0
    
    IE, IB, IP, IS, IV = 250.0, 300.0, 3, 3, 3 # in pA, pA, Hz, Hz, Hz
    Ir[0:NC[0],:] = IE
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
    ID[0:NC[0],:] = IB
    
    InpPar = {'It':It, 'Ir':Ir, 'ID':ID}  

    ### Run network 
    Aout = np.zeros((len(ada),len(T_in)))
    print 'Run full network'
    
    for j in xrange(len(ada)):
       
       b = ada[j]
       B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
       ModPar['B'] = B
                    
       for i in xrange(len(T_in)):
            
           print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(ada)),2)) + '%' 
            
           Ir[sum(NC[0:3]):N,:] = IV + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to VIP neurons
           
           # run network 
           LinNetModAda.run(Weights, ModPar, InpPar, SimPar, np.zeros(N))
           arr = np.loadtxt('Data_LinRecNeuMod_2CompPC_Ada.dat',delimiter=' ')
           t, R = arr[:,0], arr[:,1:N+1]
           
           RE = np.mean(R[:,:NC[0]],1)
           
           th = np.where(t==T_trans) # cut out transient phase
           idx_max = detect_peaks(RE, mph=np.mean(RE[t>T_trans]), mpd = 500)
           idx_max = idx_max[(idx_max>th[0])]   
           idx_min = detect_peaks(-RE, mph=np.mean(-RE[t>T_trans]), mpd = 500)
           idx_min = idx_min[(idx_min>th[0])]
           
           if np.var(RE[t>T_trans])>1e-3:
               Aout[j,i] = (np.mean(RE[idx_max])-np.mean(RE[idx_min]))/2.0
           else:
               Aout[j,i] = np.nan
               
    ### Run reference network 
    Aout_ref = np.zeros((len(ada),len(T_in)))
    print 'Run reference network'
    Ir[sum(NC[0:3]):N,:] = -20.0
    
    for j in xrange(len(ada)):
       
       b = ada[j]
       B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [b] * NC[2] + [b] * NC[3]))
       ModPar['B'] = B
                    
       for i in xrange(len(T_in)):
            
           print 'Processing: ' + str(np.round(100.0*(j*len(T_in)+i)/(len(T_in)*len(ada)),2)) + '%' 
            
           Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS + 1.0 * np.sin(2*np.pi*It/T_in[i])  # input to SOM neurons
           
           # run network 
           LinNetModAda.run(Weights, ModPar, InpPar, SimPar, np.zeros(N))
           arr = np.loadtxt('Data_LinRecNeuMod_2CompPC_Ada.dat',delimiter=' ')
           t, R = arr[:,0], arr[:,1:N+1]
           
           RE = np.mean(R[:,:NC[0]],1)
           
           th = np.where(t==T_trans) # cut out transient phase
           idx_max = detect_peaks(RE, mph=np.mean(RE[t>T_trans]), mpd = 500)
           idx_max = idx_max[(idx_max>th[0])]   
           idx_min = detect_peaks(-RE, mph=np.mean(-RE[t>T_trans]), mpd = 500)
           idx_min = idx_min[(idx_min>th[0])]
           
           if np.var(RE[t>T_trans])>1e-3:
               Aout_ref[j,i] = (np.mean(RE[idx_max])-np.mean(RE[idx_min]))/2.0
           else:
               Aout_ref[j,i] = np.nan
    
    os.remove('Data_LinRecNeuMod_2CompPC_Ada.dat') 
    return Aout, Aout_ref         


def CompCorr(flg, FB, w0, SimDur):
    
    #  flg: flag to indicate if adaptation (0) or recurrence (1) is considered
    #  FB: feedback strength (adaptation or recurrence)
    #  w0: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 350, 50, 50, 50 # N_E, N_P, N_S, N_V
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(1,NC,w0)  
    Ta = np.diag(np.array([5.0] * NC[0] + [5.0] * NC[1] + [100.0] * NC[2] + [100.0] * NC[3]))
    ModPar = {'G':G,'H':H,'A':A,'T': T,'Ta':Ta}
    
    if flg==0:
        W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
        Weights = {'W':W, 'WD': WD, 'WEI': WEI} 
    elif flg==1:
        B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [0.0] * NC[2] + [0.0] * NC[3]))
        ModPar['B'] = B

    ### define external input
    SimPar = np.array([0.0, SimDur, 0.05]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
    tini = 1000.0
    
    IE, IB, IP, IS, IV = 250.0, 300.0, 3, 3, 3 # in pA, pA, Hz, Hz, Hz
    Ir[0:NC[0],:] = IE
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    ID[0:NC[0],:] = IB
    
    InpPar = {'It':It, 'Ir':Ir, 'ID':ID}

    ### Run network 
    CorrPop = np.zeros((len(FB),4))
    SEM = np.zeros((len(FB),4))
    print 'Run network'
    
    for j in xrange(len(FB)):
        
        print 'Processing: ' + str(np.round(100.0*j/len(FB),1)) + '%'
        
        np.random.seed(186)
        
        if flg==0:
            ada = FB[j]
            B = np.diag(np.array([0.0] * NC[0] + [0.0] * NC[1] + [ada] * NC[2] + [ada] * NC[3]))
            ModPar['B'] = B
        elif flg==1:
            rec = -FB[j]
            M[2,2], M[3,3] = rec, rec
            W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon) 
            Weights = {'W':W, 'WD':WD, 'WEI': WEI}
                   
        # heterog input to VIP and SOM
        np.random.seed(132)
        I_V = IV + 0.6*np.random.normal(0,IV,size=(NC[3],len(It)))
        I_S = IS + 0.6*np.random.normal(0,IS,size=(NC[2],len(It)))
        Ir[sum(NC[0:3]):N,:] = I_V + 0.4 * np.random.normal(0,IV,size=len(It))
        Ir[sum(NC[0:2]):sum(NC[0:3]),:] = I_S + 0.4 * np.random.normal(0,IS,size=len(It)) 
        
        LinNetModAda.run(Weights, ModPar, InpPar, SimPar, np.zeros(N))
        arr=np.loadtxt('Data_LinRecNeuMod_2CompPC_Ada.dat',delimiter=' ')
        t, R = arr[:,0], arr[:,1:N+1]

        for k in range(4):
            Rj = R[t>tini,sum(NC[0:k]):sum(NC[0:k+1])]
            dat = pd.DataFrame(Rj)
            Cj = dat.corr()
            CMtx = Cj.as_matrix()
            
            id = np.triu_indices(len(CMtx),k=1)
            CorrPop[j,k] = np.mean(CMtx[id])
            SEM[j,k] = np.std(CMtx[id])/np.sqrt(len(CMtx[id]))   
     
    os.remove('Data_LinRecNeuMod_2CompPC_Ada.dat')        
    return CorrPop, SEM
    

def RelTransInpStr_Perm(x_mod,w,SimDur):
    
    #  x_mod: range of modulatory inputs tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    np.random.seed(186)
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(0,NC,w)  
    ModPar = {'G':G,'H':H,'A':A,'T': T}
    
    MD[2] = -40.0
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
    Weights = {'W':W, 'WD': WD, 'WEI': WEI} 
    
    ### define external input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
    tini = 2000.0
    
    IE, IB, IP, IS, IV = (25.0/0.07), 100.0, 12, 3.5, 3.5 # in pA, pA, Hz, Hz, Hz 250.0, 300.0, 3, 3, 3
    fx, fy = 5.0, 30.0
    x = IB + (0.5/0.07)*np.sin(fx*It/1000.0) 
    y = IE + (0.1/0.07)*np.sin(fy*It/1000.0)
    
    Ir[0:NC[0],:] = y
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
    ID[0:NC[0],:] = x

    ### Run network       
    A, B = np.zeros((2,len(x_mod))), np.zeros((2,len(x_mod)))
    print 'Run network'
    
    for j in range(2):
        
        # back and forth
        if j==0:
            inds = np.arange(0,len(x_mod))
            Inis = np.zeros(N)
        else:
            inds = np.arange(0,len(x_mod))[::-1]
            Inis = R0[-1,:]
        
        # run over all x_mod
        for i in inds:
            
            print 'Processing: ' + str(np.round(100.0*(j*len(inds)+i)/(2*len(inds)),2)) + '%'
            
            Ir[sum(NC[0:3]):N,:] = IV + x_mod[i] 
            InpPar = {'It':It, 'Ir':Ir, 'ID':ID} 
            
            LinNetMod.run(Weights, ModPar, InpPar, SimPar, Inis)
            arr = np.loadtxt('Data_LinRecNeuMod_2CompPC.dat',delimiter=' ')
            t, R0 = arr[:,0], arr[:,1:N+1]
        
            RE = np.mean(R0[:,0:NC[0]],1)
            f = interpolate.interp1d(t,RE,'cubic')
            R = f(It)        
                        
            R_new = R[It>tini] - np.mean(R[It>tini])
            x_new = x[It>tini] - np.mean(x[It>tini])
            y_new = y[It>tini] - np.mean(y[It>tini])
            M = np.array([x_new,y_new]).transpose()
            beta = np.dot(np.dot(np.linalg.inv(np.dot(M.transpose(),M)),M.transpose()),R_new)
            A[j,i], B[j,i] = beta[0], beta[1]
    
    os.remove('Data_LinRecNeuMod_2CompPC.dat')     
    return A, B


def RelTransInpStr_Imp(w,SimDur):
    
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(0,NC,w)  
    ModPar = {'G':G,'H':H,'A':A,'T': T}
    
    MD[2] = -40.0
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
    Weights = {'W':W, 'WD': WD, 'WEI': WEI}
    
    ### define external input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
    
    IE, IB, IP, IS, IV = (25.0/0.07), 100.0, 12, 3.5, 3.5 # in pA, pA, Hz, Hz, Hz 250.0, 300.0, 3, 3, 3
    fx, fy = 5.0, 30.0
    x = IB + (0.5/0.07)*np.sin(fx*It/1000.0) 
    y = IE + (0.1/0.07)*np.sin(fy*It/1000.0)
    Ts = 2*np.array([0.0,1000.0,2000.0,3000.0,4000.0,5000.0])
    dy = np.array([-120.0,120.0,-120.0,120.0,-120.0,120.0])
    
    Ir[0:NC[0],:] = y
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
    Ir[sum(NC[0:3]):N,:] = IV
    ID[0:NC[0],:] = x
    
    for i in range(len(Ts)):
        Ir[sum(NC[0:3]):N,(It>=Ts[i]) & (It<Ts[i]+10.0)] = IV + G[0]*dy[i]
        
    InpPar = {'It':It, 'Ir':Ir, 'ID':ID} 
    
    print 'Run network'
    LinNetMod.run(Weights, ModPar, InpPar, SimPar, np.zeros(N))
    arr = np.loadtxt('Data_LinRecNeuMod_2CompPC.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    Ts = np.sort(np.append(Ts,[12000.0]))
    A, B = np.zeros(len(It)), np.zeros(len(It))
    f = interpolate.interp1d(t,np.mean(R[:,0:NC[0]],1))
    RE = f(It)
    DTs = 300.0
    for i in range(len(Ts)-1):
        R_new = RE[(It>=Ts[i]+DTs) & (It<Ts[i+1])] - np.mean(RE[(It>=Ts[i]+DTs) & (It<Ts[i+1])])
        x_new = x[(It>=Ts[i]+DTs) & (It<Ts[i+1])] - np.mean(x[(It>=Ts[i]+DTs) & (It<Ts[i+1])])
        y_new = y[(It>=Ts[i]+DTs) & (It<Ts[i+1])] - np.mean(y[(It>=Ts[i]+DTs) & (It<Ts[i+1])])
        M = np.array([x_new,y_new]).transpose()
        beta = np.dot(np.dot(np.linalg.inv(np.dot(M.transpose(),M)),M.transpose()),R_new)
        A[(It>=Ts[i]) & (It<Ts[i+1])] = beta[0]
        B[(It>=Ts[i]) & (It<Ts[i+1])] = beta[1]  
    
    os.remove('Data_LinRecNeuMod_2CompPC.dat')     
    return t, R, x, y, A, B


def Hysteresis(x_mod,w,SimDur):
    
    #  x_mod: range of modulatory inputs tested
    #  w: fixed mutual inhibition
    #  SimDur: simulation time (ms)

    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(0,NC,w)  
    ModPar = {'G':G,'H':H,'A':A,'T': T}
    
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
    Weights = {'W':W, 'WD': WD, 'WEI': WEI}    

    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
     
    IE, IB, IP, IS, IV = 250.0, 300.0, 3, 3, 3 # in pA, pA, Hz, Hz, Hz
    Ir[0:NC[0],:] = IE
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS
    Ir[sum(NC[0:3]):N,:] = IV 
    ID[0:NC[0],:] = IB
    
    InpPar = {'It':It, 'Ir':Ir, 'ID':ID}
    
    ### Run network  
    rini = np.zeros(N)
    rates = np.zeros((2,len(x_mod)))
    
    print 'Run network - mod. input ascending'   
    for ii in range(len(x_mod)):
        
        print 'Processing: ' + str(np.round(100.0*ii/len(x_mod),1)) + '%'
        Ir[sum(NC[0:3]):N,:] = IV + x_mod[ii]

        LinNetMod.run(Weights, ModPar, InpPar, SimPar, rini)
        arr = np.loadtxt('Data_LinRecNeuMod_2CompPC.dat',delimiter=' ')
        R = arr[:,1:N+1]
        rini = R[-1,:]
        
        rates[0,ii] = np.mean(R[-1,0:NC[0]])
    
    print 'Run network - mod. input descending'
    for ii in reversed(range(len(x_mod))):
        
        print 'Processing: ' + str(np.round(100.0*(len(x_mod)-ii)/len(x_mod),1)) + '%'
        Ir[sum(NC[0:3]):N,:] = IV + x_mod[ii]
        
        LinNetMod.run(Weights, ModPar, InpPar, SimPar, rini)
        arr = np.loadtxt('Data_LinRecNeuMod_2CompPC.dat',delimiter=' ')
        R = arr[:,1:N+1]
        rini = R[-1,:]
        
        rates[1,ii] = np.mean(R[-1,0:NC[0]])
     
    os.remove('Data_LinRecNeuMod_2CompPC.dat') 
    return rates  
    
    
def RunSig_2Inp(x_SOM,x_VIP,w0,SimDur):
    
    #  x_SOM: range of modulatory inputs to SOMs
    #  x_VIP: range of modulatory inputs to VIPs
    #  w0: fixed mutual inhibition
    #  SimDur: simulation time (ms)
    
    print 'Set parameters'
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(0,NC,w0)  
    ModPar = {'G':G,'H':H,'A':A,'T': T}
    
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
    Weights = {'W':W, 'WD': WD, 'WEI': WEI}    

    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
     
    IE, IB, IP, IS, IV = 250.0, 300.0, 3, 3, 3 # in pA, pA, Hz, Hz, Hz
    Ir[0:NC[0],:] = IE
    Ir[NC[0]:sum(NC[0:2]),:] = IP
    ID[0:NC[0],:] = IB
    
    print 'Run network'
    rates = np.zeros((len(x_VIP),len(x_SOM)))
        
    for j in xrange(len(x_VIP)):
        for i in xrange(len(x_SOM)):
        
            print 'Processing: ' + str(np.round(100.0*(j*len(x_SOM)+i)/(len(x_SOM)*len(x_VIP)),2)) + '%'       
           
            # run network
            Ir[sum(NC[0:3]):N,:] = IV + x_VIP[j]
            Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS + x_SOM[i]
            InpPar = {'It':It, 'Ir':Ir, 'ID':ID}
                   
            LinNetMod.run(Weights, ModPar, InpPar, SimPar, np.zeros(N))
            arr = np.loadtxt('Data_LinRecNeuMod_2CompPC.dat',delimiter=' ')
            R = arr[:,1:N+1]
            
            rates[j,i] = np.mean(R[-1,0:NC[0]]) 
    
    os.remove('Data_LinRecNeuMod_2CompPC.dat')         
    return rates
    

def Run_MismatchDetection(flg,w,SimDur): 
    
    #  flg: flag to indicate the stimulation protocol
    #  w: fixed mutual inhibition, w = wSV, wVS = w - 0.2
    #  SimDur: simulation time (ms)
    
    print 'Set parameters' 
    
    ### Define neuron parameters & set connection weights and probabilities
    NC = 70, 10, 10, 10 # N_E, N_P, N_S, N_V
    N, G, H, A, T, P, M, MD, wEP, NCon = SetDefaultPara(0,NC,w) 
    T[:NC[0],:] = 6*T[:NC[0],:]
    ModPar = {'G':G,'H':H,'A':A,'T': T}
    
    ### Adjust connection weights and probabilities
    wPP = -1.0
    wPS = -0.2            
    wEP = -(1+abs(wPP))/(1-abs(wPS))/0.07

    M = np.array([[0.0,0.0,0,0],[1.0,wPP,wPS,0],[1.0,0.0,0.0,w],[1.0,0.0,w-0.2,0.0]]) 
    W, WD, WEI = SetConnectivity(M,MD,wEP,NC,NCon)
    Weights = {'W':W, 'WD': WD, 'WEI': WEI}    

    ### Define background (external) input
    SimPar = np.array([0.0, SimDur, 0.2]) 
    It = np.linspace(SimPar[0],SimPar[1],int(2*SimDur))
    Ir = np.zeros((N,len(It)))
    ID = np.zeros((NC[0],len(It)))
    
    xm = 150.0 # additional motot-related input (in pA)
    xv = 150.0 # additional visual input (in pA)   
    SD = 50.0
    
    IE, IB, IP, IS, IV = 400.0, 0.0, 3, 3, 3 # in pA, pA, Hz, Hz, Hz
    
    Inp_v = np.zeros(len(It))
    Inp_m = np.zeros(len(It))
    
    if flg<2:
        Ti = np.array([500.0,2500.0,7500.0])
        Tj = np.array([1500.0,3500.0,9500.0])
        for ii in range(len(Ti)):
            Inp_v[(It>=Ti[ii]) & (It<Tj[ii])] = xv
            Inp_m[(It>=Ti[ii]) & (It<Tj[ii])] = xm
            X = np.random.uniform(low=200.0,high=800.0)
            Inp_v[(It>=X+Ti[ii]) & (It<X+Ti[ii]+100.0)] = 0.0            
        if flg==1: # playback session
            Inp_m = np.zeros(len(It))
    elif flg==2:
        Inp_v[(It>=1000.0) & (It<1450.0)] = xv
        Inp_v[(It>=1550.0) & (It<2000.0)] = xv
        Inp_m[(It>=1000.0) & (It<2000.0)] = xm    
       
    Ir[0:NC[0],:] = IE + Inp_v + np.random.normal(0.0,SD,size=(NC[0],len(It)))
    Ir[NC[0]:sum(NC[0:2]),:] = IP + Inp_v*0.07 + np.random.normal(0.0,SD*0.07,size=(NC[1],len(It)))
    Ir[sum(NC[0:2]):sum(NC[0:3]),:] = IS + Inp_v*0.07 + np.random.normal(0.0,SD*0.07,size=(NC[2],len(It)))
    Ir[sum(NC[0:3]):N,:] = IV + Inp_m*0.07  + np.random.normal(0.0,SD*0.07,size=(NC[3],len(It)))
    ID[0:NC[0],:] = IB + Inp_m + np.random.normal(0.0,SD,size=(NC[0],len(It)))
    
    InpPar = {'It':It, 'Ir':Ir, 'ID':ID}
    
    print 'Run network'
    LinNetMod.run(Weights, ModPar, InpPar, SimPar, np.zeros(N))
    arr = np.loadtxt('Data_LinRecNeuMod_2CompPC.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    os.remove('Data_LinRecNeuMod_2CompPC.dat')         
    return t, R, Inp_v, Inp_m
