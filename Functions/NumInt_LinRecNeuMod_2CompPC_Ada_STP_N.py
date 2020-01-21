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
def DiffEq (r, a, u, x, W, B, G, H, Tin, Tain, Us, Ux, Tu, Tx, I):
    
    dr = np.dot(Tin,-r + G*(np.dot(u*x*W,r) + I - H - a)) 
    da = np.dot(Tain, -a + np.dot(B,r))
    du = (Us-u)/Tu + np.dot(Us*(1.0-u),np.diag(r/1000.0))
    dx = (1-x)/Tx - np.dot(Ux*x,np.diag(r/1000.0)) 
    
    return dr, da, du, dx
    

# numerical solution of the differential equation defined in DiffEq
def update (r, a, u, x, W, B, G, H, Tin, Tain, Us, Ux, Tu, Tx, I, dt):

    r0, a0, u0, x0 = np.copy(r), np.copy(a), np.copy(u), np.copy(x)    
    
    dr1, da1, du1, dx1 = DiffEq(r0,a0,u0,x0,W,B,G,H,Tin,Tain,Us,Ux,Tu,Tx,I)
    r0 += dt * dr1
    a0 += dt * da1
    u0 += dt * du1
    x0 += dt * dx1
    
    dr2, da2, du2, dx2 = DiffEq(r0,a0,u0,x0,W,B,G,H,Tin,Tain,Us,Ux,Tu,Tx,I)
    r += dt/2.0 * (dr1 + dr2)
    a += dt/2.0 * (da1 + da2)
    u += dt/2.0 * (du1 + du2)
    x += dt/2.0 * (dx1 + dx2)
    
    r[r<0]=0
    
    r = np.round(r,8) # for numerical stability ...
    u = np.round(u,8)
    x = np.round(x,8)
    
    return r, a, u, x
    

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

def run(Weights, ModPar, STPar, InpPar, SimPar, IPar):
    
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
    Us = STPar['Us']
    Ux = STPar['Ux']
    Tu = STPar['Tu']
    Tx = STPar['Tx']
    
    # rename & define parameters
    T0, Tstop, dt0 = SimPar[0], SimPar[1], SimPar[2]
    ti, N, r = T0, len(IPar), IPar
    u, x, a = np.copy(Us), np.ones((N,N)), np.zeros(N)
    Tin, Tain = np.linalg.inv(T), np.linalg.inv(Ta)

    # definition: external input
    f = interpolate.interp1d(It,Ir)
    g = interpolate.interp1d(It,ID)
    
    # open file & write initial conditions
    fp = open('Data_LinRecNeuMod_2CompPC_Ada_STP.dat','w')
    fp.write("%f" % ti)
    for i in xrange(N):
        fp.write(" %f" % r[i])
    fp.write("\n")   
    
    # main loop
    while ti<=Tstop:
        
        Iall, Idend = f(ti), g(ti) 
        I = CompInp2SomaPyr(r,Iall,Idend,WD,WEI,A)
        r, a, u, x = update(r,a,u,x,W,B,G,H,Tin,Tain,Us,Ux,Tu,Tx,I,dt0)
        
        fp.write("%f" % (ti+dt0))
        for i in xrange(N):
            fp.write(" %f" % r[i])
        fp.write("\n") 
        
        ti+=dt0
        
    fp.closed
    return
    