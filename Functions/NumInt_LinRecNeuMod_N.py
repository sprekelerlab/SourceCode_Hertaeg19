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
def DiffEq(r, W, Tin, I, N):
    
    dr = np.zeros(N)
    dr = np.dot(Tin,-r + np.dot(W,r) + I) 
        
    return dr
    

# numerical solution of the differential equation defined in DiffEq
def update(r, W, Tin, I, dt, N):

    dr1 = np.zeros(N)
    dr2 = np.zeros(N)
    r0 = np.copy(r)    
    
    dr1 = DiffEq(r0,W,Tin,I,N)
    r0 += dt * dr1
    
    dr2 = DiffEq(r0,W,Tin,I,N)
    r += dt/2.0 * (dr1 + dr2)
    
    idx=np.where( r < 0 )
    r[idx]=0

    r = np.round(r,8) # for numerical stability ...
    
    return r


"""
/*****************/
/* Main function */
/*****************/ """

def run(W, ModPar, InpPar, SimPar, IPar):

    # unfold dictionaries
    T = ModPar['T']
    It = InpPar['It']
    Ir = InpPar['Ir']
    
    # initialization & declaration
    T0, Tstop, dt0 = SimPar[0], SimPar[1], SimPar[2]
    ti, N, r = T0, np.size(W,0), IPar
    Tin = np.linalg.inv(T)

    # definition: external input
    f = interpolate.interp1d(It,Ir)
    
    # open file & write initial conditions
    fp = open('Data_LinRecNeuMod.dat','w')
    fp.write("%f" % ti)
    for i in xrange(N):
        fp.write(" %f" % r[i])
    fp.write("\n")   
    
    # main loop
    while ti<=Tstop:
        
        I = f(ti) 
        r = update(r,W,Tin,I,dt0,N)
        
        fp.write("%f" % (ti+dt0))
        for i in xrange(N):
            fp.write(" %f" % r[i])
        fp.write("\n") 
        
        ti+=dt0
        
    fp.closed
    return
    