# -*- coding: utf-8 -*-
"""
@author: L. Hertäg
"""

import numpy as np
dtype = np.float32


def SetDefaultPara(flg,NC,w0,rec=0.0):
    
    #  flg: flag to indicate if network is simulated with (1) or without (0) recurrence
    #  NC: number of cells per cell type
    #  w0: fixed mutual inhibition
    #  rec: recurrence
    
    N = sum(NC)
    G = np.ones(N, dtype=dtype)
    G[:NC[0]] = dtype(0.07) # in 1/(pA*s)    
    H = np.zeros(N, dtype=dtype)
    H[:NC[0]] = dtype(200.0) # in pA
    A = np.array([0.31,0.27,100.0,400.0], dtype=dtype) # [lE,lD,c,TC] 
    T = dtype(10.0)*np.eye(N,dtype=dtype) 
    
    if flg==0:
        P = np.array([[0.1,0.6,0.55,0.0],[0.45,0.5,0.6,0.0],[0.35,0.0,0.0,0.5],[0.1,0.0,0.45,0.0]], dtype=dtype)
        M = np.array([[0.0,0.0,0,0],[1.0,-1.5,-1.3,0],[1.0,0.0,0.0,w0],[1.0,0.0,w0,0.0]], dtype=dtype) 
    else:
        P = np.array([[0.1,0.6,0.55,0.0],[0.45,0.5,0.6,0.0],[0.35,0.0,0.5,0.5],[0.1,0.0,0.45,0.5]], dtype=dtype)
        M = np.array([[0.0,0.0,0,0],[1.0,-1.5,-1.3,0],[1.0,0.0,rec,w0],[1.0,0.0,w0,rec]], dtype=dtype) 
    NCon = np.round(P*NC).astype(np.int32)    
    MD = np.array([6.0,0.0,-28.0,0.0], dtype=dtype) # wDE, wDP, WDS, wDV (in pA*s)
    wEP = dtype(-10.0) # (in pA*s)

    return N, G, H, A, T, P, M, MD, wEP, NCon
    

def SetPlasticityParameters(UF,UD,TF,TD,NC):
    
    #  UF: initial release probability for all cell type combinations (facilitation)
    #  UD: initial values for all cell type combinations (depression)
    #  TF: facilitation time constant for all cell type combinations
    #  TD: depression time constant for all cell type combinations
    #  NC: number of cells per cell type   

    K, L, M, N = [], [], [], []
    
    for i in range(16):
        m,n = np.unravel_index(i,(4,4))
        Mtx = UF[m,n]*np.ones((NC[m],NC[n]), dtype=dtype)
        K.append(Mtx)
        Mtx = TF[m,n]*np.ones((NC[m],NC[n]), dtype=dtype)
        L.append(Mtx)
        Mtx = UD[m,n]*np.ones((NC[m],NC[n]), dtype=dtype)
        M.append(Mtx)
        Mtx = TD[m,n]*np.ones((NC[m],NC[n]), dtype=dtype)
        N.append(Mtx)
            
    Us = np.bmat([[K[0],K[1],K[2],K[3]],[K[4],K[5],K[6],K[7]],[K[8],K[9],K[10],K[11]],[K[12],K[13],K[14],K[15]]])
    Ux = np.bmat([[M[0],M[1],M[2],M[3]],[M[4],M[5],M[6],M[7]],[M[8],M[9],M[10],M[11]],[M[12],M[13],M[14],M[15]]])
    Tu = np.bmat([[L[0],L[1],L[2],L[3]],[L[4],L[5],L[6],L[7]],[L[8],L[9],L[10],L[11]],[L[12],L[13],L[14],L[15]]])
    Tx = np.bmat([[N[0],N[1],N[2],N[3]],[N[4],N[5],N[6],N[7]],[N[8],N[9],N[10],N[11]],[N[12],N[13],N[14],N[15]]])
    Us, Ux, Tu, Tx = np.asarray(Us, dtype=dtype), np.asarray(Ux, dtype=dtype), np.asarray(Tu, dtype=dtype), np.asarray(Tx, dtype=dtype)
    
    return Us, Ux, Tu, Tx 


# define function to set connectivity
def SetConnectivity(M,MD,wEP,NC,NCon):
    
    #  M: array of all total weights, excluding E/I --> E (PC soma)
    #  MD: array of all total weights from E/I --> D (PC dendrites)
    #  wEP: total weight from PC neurons to soma of PCs
    #  NC: number of cells per cell type 
    #  NCon: number of connections between cell types

    # create weight matrix (connections onto INs)
    K=[]
    for i in xrange(16):
        m,n = np.unravel_index(i,(4,4))
        Mtx = np.zeros((NC[m],NC[n]), dtype=dtype)
        if NCon[m,n]>0:
            if m==n:
                for l in xrange(NC[m]):
                    r = M[m,n]*np.array([0] * (NC[n]-1-NCon[m,n]) + [1] * NCon[m,n], dtype=dtype)/NCon[m,n]
                    np.random.shuffle(r)
                    r = np.insert(r,l,0)         
                    Mtx[l,:] = r            
            else:
                for l in xrange(NC[m]):
                    r = M[m,n]*np.array([0] * (NC[n]-NCon[m,n]) + [1] * NCon[m,n], dtype=dtype)/NCon[m,n]
                    np.random.shuffle(r)      
                    Mtx[l,:] = r           
        K.append(Mtx)
    
    W = np.bmat([[K[0],K[1],K[2],K[3]],[K[4],K[5],K[6],K[7]],[K[8],K[9],K[10],K[11]],[K[12],K[13],K[14],K[15]]])
    W = np.asarray(W, dtype=dtype)
    
    # create weights onto pyramidal cells (soma & dendrites)
    K = []
    for i in xrange(4):
        Mtx=np.zeros((NC[0],NC[i]), dtype=dtype)
        if i in [0,2]:
            for l in xrange(NC[0]):
                r = MD[i]*np.array([0] * (NC[i]-NCon[0,i]) + [1] * NCon[0,i], dtype=dtype)/NCon[0,i]  
                np.random.shuffle(r)
                Mtx[l,:] = r
        K.append(Mtx)
    WD = np.bmat([K[0],K[1],K[2],K[3]])
    #WD = np.asarray(WD, dytpe=dtype)
    
    K=[]
    for i in xrange(4):
        Mtx=np.zeros((NC[0],NC[i]))
        if i==1:
            for l in xrange(NC[0]):
                r = wEP*np.array([0] * (NC[1]-NCon[0,1]) + [1] * NCon[0,1])/NCon[0,1]  
                np.random.shuffle(r)
                Mtx[l,:] = r
        K.append(Mtx)
    WEI = np.bmat([K[0],K[1],K[2],K[3]])
    #WEI = np.asarray(WEI, dytpe=dtype)
    
    return W, WD, WEI
       
    