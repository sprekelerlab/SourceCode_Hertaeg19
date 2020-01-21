# -*- coding: utf-8 -*-
"""
@author: L. Hertäg
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def SetDefaultPara(flg,NC):
    
    #  flg: flag to indicate if network is simulated with (1) or without (0) recurrence
    #  NC: number of cells per cell type
    
    N = sum(NC)
    T = 10.0*np.eye(N) 
    r0 = [0.0,3.0,3.0,3.0]
    
    if flg==0:
        P = np.array([[0.1,0.6,0.55,0.0],[0.45,0.5,0.6,0.0],[0.35,0.0,0.0,0.5],[0.1,0.1,0.45,0.0]])
    else:
        P = np.array([[0.1,0.6,0.55,0.0],[0.45,0.5,0.6,0.0],[0.35,0.0,0.5,0.5],[0.1,0.1,0.45,0.5]])
    M = np.array([[0.0,-1.5,0,0],[0.0,-1.5,-1.3,0],[0.0,0,0,-1.0],[0.0,0,-1.0,0]])
    NCon = np.round(P*NC).astype(np.int64)
    
    return N, T, r0, P, M, NCon
    

# define function to set connectivity
def SetConnectivity(w_SV,w_VS,w_SS,w_VV,M,NC,NCon):
    
    #  w_XY: total weight from neuron type Y onto neuron type X
    #  M: array of all total weights
    #  NC: number of cells per cell type
    #  NCon: number of connections between cell types
    
    np.random.seed(186)

    M[2,3], M[3,2] = w_SV, w_VS
    M[2,2], M[3,3] = w_SS, w_VV
    
    K=[]
    for i in range(16):
        m,n = np.unravel_index(i,(4,4))
        Mtx=np.zeros((NC[m],NC[n]))
        if M[m,n]!=0.0:
            if m==n:
                for l in xrange(NC[m]):
                    r = (M[m,n]/NCon[m,n])*np.array([0] * (NC[n]-1-NCon[m,n]) + [1] * NCon[m,n])
                    np.random.shuffle(r)
                    r = np.insert(r,l,0)         
                    Mtx[l,:] = r           
            else:
                for l in xrange(NC[m]):
                    r = (M[m,n]/NCon[m,n])*np.array([0] * (NC[n]-NCon[m,n]) + [1] * NCon[m,n])
                    np.random.shuffle(r)      
                    Mtx[l,:] = r           
        K.append(Mtx)
    
    W = np.bmat([[K[0],K[1],K[2],K[3]],[K[4],K[5],K[6],K[7]],[K[8],K[9],K[10],K[11]],[K[12],K[13],K[14],K[15]]])
    W = np.asarray(W)
    
    return W
    
    
def SetPlasticityParameters(u,tf,NC):
    
    #  u: initial release probability
    #  tf: facilitation time constant
    #  NC: number of cells per cell type
       
    UEE, UEP, UES, UEV = np.ones((NC[0],NC[0])), np.ones((NC[0],NC[1])), np.ones((NC[0],NC[2])), np.ones((NC[0],NC[3])) 
    UPE, UPP, UPS, UPV = np.ones((NC[1],NC[0])), np.ones((NC[1],NC[1])), np.ones((NC[1],NC[2])), np.ones((NC[1],NC[3]))
    USE, USP, USS, USV = np.ones((NC[2],NC[0])), np.ones((NC[2],NC[1])), np.ones((NC[2],NC[2])), u*np.ones((NC[2],NC[3]))
    UVE, UVP, UVS, UVV = np.ones((NC[3],NC[0])), np.ones((NC[3],NC[1])), u*np.ones((NC[3],NC[2])), np.ones((NC[3],NC[3]))
    
    TEE, TEP, TES, TEV = np.ones((NC[0],NC[0])), np.ones((NC[0],NC[1])), np.ones((NC[0],NC[2])), np.ones((NC[0],NC[3]))
    TPE, TPP, TPS, TPV = np.ones((NC[1],NC[0])), np.ones((NC[1],NC[1])), np.ones((NC[1],NC[2])), np.ones((NC[1],NC[3]))
    TSE, TSP, TSS, TSV = np.ones((NC[2],NC[0])), np.ones((NC[2],NC[1])), np.ones((NC[2],NC[2])), tf*np.ones((NC[2],NC[3]))
    TVE, TVP, TVS, TVV = np.ones((NC[3],NC[0])), np.ones((NC[3],NC[1])), tf*np.ones((NC[3],NC[2])), np.ones((NC[3],NC[3]))
    
    Us = np.bmat([[UEE,UEP,UES,UEV],[UPE,UPP,UPS,UPV],[USE,USP,USS,USV],[UVE,UVP,UVS,UVV]])
    Tf = np.bmat([[TEE,TEP,TES,TEV],[TPE,TPP,TPS,TPV],[TSE,TSP,TSS,TSV],[TVE,TVP,TVS,TVV]])
    Us, Tf = np.asarray(Us), np.asarray(Tf)
    
    return Us, Tf


def PhasePlane_SOMVIP(w,xt,minFS):
    
    #  w: mutual inhibition w = [wsv,wvs]
    #  xt: external stimulation xt = [xs,xv]
    #  minFS: minimum fontsize 
    
    tau = 0.01 # sec
    
    # compute FPs:
    FP = np.zeros((3,2)) 
    
    FP[0,0] = xt[0]
    FP[2,1] = xt[1]
    FP[1,0] = (xt[0]-w[0]*xt[1])/(1-w[0]*w[1])
    FP[1,1] = xt[1] - w[1]*FP[1,0]
    
    # compute nullclines
    rs = np.linspace(0.0,1.1*xt[0],100)
    null_v = xt[1] - w[1]*rs # rv-nullcline
    null_v[null_v<0] = 0.0
    rv = np.linspace(0.0,1.1*xt[1],100)
    null_s = xt[0] - w[0]*rv # rs-nullcline
    null_s[null_s<0] = 0.0
    
    # compute vector field
    x = np.linspace(0, max(rs), 10)
    y = np.linspace(0, 1.1*xt[1], 10)
    xv, yv = np.meshgrid(x, y)
    u = (-xv - w[0]*yv + xt[0])/tau
    v = (-yv - w[1]*xv + xt[1])/tau  
    speed = np.sqrt(u*u + v*v)
    
    plt.plot(rs,null_v,lw=2,color='#51a76d', clip_on=False, zorder=3)
    plt.plot(null_s,rv,lw=2,color='#91b9d2', clip_on=False, zorder=3)
    if FP[0,0] >= xt[1]/w[1]:
        plt.plot(FP[0,0],FP[0,1],'o', mfc='k', mec='k', Markersize=6, clip_on=False, zorder=5)   
    if FP[2,1] >= xt[0]/w[0]:
        plt.plot(FP[2,0],FP[2,1], 'o', mfc='k', mec='k', Markersize=6, clip_on=False, zorder=5)
    if all(FP[1,:]>0):
        if np.sqrt(w[0]*w[1])>=1:
            plt.plot(FP[1,0],FP[1,1], 'o', mfc='none', mec='k', Markersize=6, linewidth=1, mew=1, zorder=5)
        else:
            plt.plot(FP[1,0],FP[1,1],'ko',Markersize=6, zorder=5)
    plt.streamplot(xv, yv, u, v, linewidth=2.0*speed/speed.max(),color=[0.7,0.7,0.7,1.0],density=0.5,arrowsize=5.0)   
    
    plt.xlim([0,1.1*xt[0]])
    plt.ylim([0,1.1*xt[1]])    
    
    ax = plt.gca()
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_xticks([0,5])
    ax.set_yticks([0,2,4,6])
    ax.set_xlabel(r'SOM rate (s$^{-1}$)',fontsize=minFS)
    ax.set_ylabel(r'VIP rate (s$^{-1}$)',fontsize=minFS)
    
    sns.despine()
    
    return 