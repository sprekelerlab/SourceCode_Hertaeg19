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
def DiffEq (r, a, W, B, G, H, Tin, Tain, I):
     
    dr, da = np.zeros(len(I)), np.zeros(len(I))
    dr = np.dot(Tin,-r + G*(np.dot(W,r) + I - H - a))  
    da = np.dot(Tain, -a + np.dot(B,r))
    
    return dr, da
    

# numerical solution of the differential equation defined in DiffEq
def update (r, a, W, B, G, H, Tin, Tain, I, dt):

    dr1, dr2 = np.zeros(len(I)), np.zeros(len(I))
    da1, da2 = np.zeros(len(I)), np.zeros(len(I))
    r0, a0 = np.copy(r), np.copy(a)    
    
    dr1, da1 = DiffEq(r0,a0,W,B,G,H,Tin,Tain,I)
    r0 += dt * dr1
    a0 += dt * da1
    
    dr2, da2 = DiffEq(r0,a0,W,B,G,H,Tin,Tain,I)
    r += dt/2.0 * (dr1 + dr2)
    a += dt/2.0 * (da1 + da2)
    
    idx=np.where( r < 0 )
    r[idx]=0

    r = np.round(r,8) # for numerical stability ...
    
    return r, a


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
    Ta = ModPar['Ta']
    B = ModPar['B']
    It = InpPar['It']
    Ir = InpPar['Ir']
    ID = InpPar['ID']
    
    # rename & define parameters
    T0, Tstop, dt0 = SimPar[0], SimPar[1], SimPar[2]
    ti, N = T0, len(IPar)
    Tin, Tain = np.linalg.inv(T), np.linalg.inv(Ta)
    r, a = IPar, np.zeros(N)

    # definition: external input
    f = interpolate.interp1d(It,Ir)
    g = interpolate.interp1d(It,ID)
    
    # open file & write initial conditions
    fp = open('Data_LinRecNeuMod_2CompPC_Ada.dat','w')
    fp.write("%f" % ti)
    for i in xrange(N):
        fp.write(" %f" % r[i])
    fp.write("\n")   
    
    # main loop
    while ti<=Tstop:
        
        Iall, Idend = f(ti), g(ti) 
        I = CompInp2SomaPyr(r,Iall,Idend,WD,WEI,A)
        r, a = update(r,a,W,B,G,H,Tin,Tain,I,dt0)
        
        fp.write("%f" % (ti+dt0))
        for i in xrange(N):
            fp.write(" %f" % r[i])
        fp.write("\n") 
        
        ti+=dt0
        
    fp.closed
    return
    