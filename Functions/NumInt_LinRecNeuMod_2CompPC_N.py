# -*- coding: utf-8 -*-
"""
@author: L. Hertäg
"""

"""
/********************/
/* Include libaries */
/********************/ """

import numpy as np
from scipy import interpolate

"""
/***************************/
/* computational functions */
/***************************/ """

# diff equations defining the linear rate-based model
def DiffEq (r, W, G, H, Tin, I):
     
    dr = np.zeros(len(I))
    dr = np.dot(Tin,-r + G*(np.dot(W,r) + I - H))  
    
    return dr
    

# numerical solution of the differential equation defined in DiffEq
def update (r, W, G, H, Tin, I, dt):

    dr1, dr2 = np.zeros(len(I)), np.zeros(len(I))
    r0 = np.copy(r)    
    
    dr1 = DiffEq(r0,W,G,H,Tin,I)
    r0 += dt * dr1
    
    dr2 = DiffEq(r0,W,G,H,Tin,I)
    r += dt/2.0 * (dr1 + dr2)
    
    idx=np.where( r < 0 )
    r[idx]=0
    
    return r


def CompInp2SomaPyr(r,X,XD,WD,WEI,A):
    
    ND = len(XD)
    
    IS = X[0:ND] + np.dot(WEI,r)
    ID = XD + np.dot(WD,r)
    I0 = A[0]*IS + (1-A[1])*ID
    IC = A[2] * 0.5*(np.sign(I0 - A[3]) + 1)
    
    ID = ID + IC
    ID[ID<0.0] = 0.0 # rectify    
    
    X[0:ND] = (1-A[0])*IS + A[1]*ID
    # A = [lE,lD,c,TC]
    
    return X

"""
/*****************/
/* Main function */
/*****************/ """

def run(Weights, ModPar, InpPar, SimPar, IPar):
    
    # unfold dictionaries
    W = Weights['W']
    WD = Weights['WD']
    WEI = Weights['WEI']
    G = ModPar['G']
    H = ModPar['H']
    A = ModPar['A']
    T = ModPar['T']
    It = InpPar['It']
    Ir = InpPar['Ir']
    ID = InpPar['ID']
    
    # rename & define parameters
    T0, Tstop, dt0 = SimPar[0], SimPar[1], SimPar[2]
    ti, N, r = T0, len(IPar), IPar
    Tin = np.linalg.inv(T)

    # definition: external input
    f = interpolate.interp1d(It,Ir)
    g = interpolate.interp1d(It,ID)
    
    # open file & write initial conditions
    fp = open('Data_LinRecNeuMod_2CompPC.dat','w')
    fp.write("%f" % ti)
    for i in xrange(N):
        fp.write(" %f" % r[i])
    fp.write("\n")   
    
    # main loop
    while ti<=Tstop:
        
        Iall, Idend = f(ti), g(ti) 
        I = CompInp2SomaPyr(r,Iall,Idend,WD,WEI,A)
        r = update(r,W,G,H,Tin,I,dt0)
        
        fp.write("%f" % (ti+dt0))
        for i in xrange(N):
            fp.write(" %f" % r[i])
        fp.write("\n") 
        
        ti+=dt0
        
    fp.closed
    return
    