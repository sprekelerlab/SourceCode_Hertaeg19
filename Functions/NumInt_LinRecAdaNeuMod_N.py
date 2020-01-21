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

# diff equations defining the linear rate based model
def DiffEq (r, a, W, B, Tin, Tain, I, N):
    
    dr, da = np.zeros(N), np.zeros(N)
    dr = np.dot(Tin,-r + np.dot(W,r) + I - a) 
    da = np.dot(Tain, -a + np.dot(B,r))
        
    return dr, da
    

# numerical solution of the differential equation defined in DiffEq
def update (r, a, W, B, Tin, Tain, I, dt, N):

    dr1, da1 = np.zeros(N), np.zeros(N)
    dr2, da2 = np.zeros(N), np.zeros(N)
    r0, a0 = np.copy(r), np.copy(a)    
    
    dr1, da1 = DiffEq(r0,a0,W,B,Tin,Tain,I,N)
    r0 += dt * dr1
    a0 += dt * da1
    
    dr2, da2 = DiffEq(r0,a0,W,B,Tin,Tain,I,N)
    r += dt/2.0 * (dr1 + dr2)
    a += dt/2.0 * (da1 + da2)
    
    idx=np.where( r < 0 )
    r[idx]=0
    
    r = np.round(r,8) # for numerical stability ...
    
    return r, a


"""
/*****************/
/* Main function */
/*****************/ """

def run(W, ModPar, InpPar, SimPar, IPar):
    
    # unfold dictionaries
    T = ModPar['T']
    Ta = ModPar['Ta']
    B = ModPar['B']
    It = InpPar['It']
    Ir = InpPar['Ir']
    
    # initialization & declaration
    T0, Tstop, dt0 = SimPar[0], SimPar[1], SimPar[2]
    ti, N, r = T0, np.size(W,0), IPar
    Tin, Tain = np.linalg.inv(T), np.linalg.inv(Ta)
    a = np.zeros(N)

    # definition: external input
    f = interpolate.interp1d(It,Ir)
    
    # open file & write initial conditions
    fp = open('Data_LinRecAdaNeuMod.dat','w')
    fp.write("%f" % ti)
    for i in xrange(N):
        fp.write(" %f" % r[i])
    fp.write("\n")   
    
    # main loop
    while ti<=Tstop:
        
        I = f(ti) 
        r, a = update(r,a,W,B,Tin,Tain,I,dt0,N)
        
        fp.write("%f" % (ti+dt0))
        for i in xrange(N):
            fp.write(" %f" % r[i])
        fp.write("\n") 
        
        ti += dt0
        
    fp.closed
    return
    