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
def DiffEq(r, u, W, US, Tin, Tf, I, N):
    
    dr, du = np.zeros(N), np.zeros((N,N))
    dr = np.dot(Tin,-r + np.dot(u*W,r) + I) 
    du = (US-u)/Tf + np.dot(US*(1-u),np.diag(r/1000.0))
        
    return dr, du
    

# numerical solution of the differential equation defined in DiffEq
def update(r, u, W, US, Tin, Tf, I, dt, N):

    dr1, du1 = np.zeros(N), np.zeros((N,N))
    dr2, du2 = np.zeros(N), np.zeros((N,N))
    r0, u0 = np.copy(r), np.copy(u)    
    
    dr1, du1 = DiffEq(r0,u0,W,US,Tin,Tf,I,N)
    r0 += dt * dr1
    u0 += dt * du1
    
    dr2, du2 = DiffEq(r0,u0,W,US,Tin,Tf,I,N)
    r += dt/2.0 * (dr1 + dr2)
    u += dt/2.0 * (du1 + du2)
    
    idx = np.where( r < 0 )
    r[idx] = 0
     
    r = np.round(r,8) # for numerical stability ...
    
    return r


"""
/*****************/
/* Main function */
/*****************/ """

def run(W, ModPar, STPar, InpPar, SimPar, IPar):
    
    # unfold dictionaries
    T = ModPar['T']
    It = InpPar['It']
    Ir = InpPar['Ir']
    Us = STPar['Us']
    Tf = STPar['Tf']
    
    # initialization & declaration
    T0, Tstop, dt0 = SimPar[0], SimPar[1], SimPar[2]
    ti, N, r = T0, np.size(W,0), IPar
    Tin = np.linalg.inv(T)
    u = np.copy(Us)

    # definition: external input
    f = interpolate.interp1d(It,Ir)
    
    # open file & write initial conditions
    fp = open('Data_LinRecNeuModSTP.dat','w')
    fp.write("%f" % ti)
    for i in xrange(N):
        fp.write(" %f" % r[i])
    fp.write("\n")   
    
    # main loop
    while ti<=Tstop:
        
        I = f(ti) 
        r = update(r,u,W,Us,Tin,Tf,I,dt0,N)
        
        fp.write("%f" % (ti+dt0))
        for i in xrange(N):
            fp.write(" %f" % r[i])
        fp.write("\n") 
        
        ti += dt0
        
    fp.closed
    return
    