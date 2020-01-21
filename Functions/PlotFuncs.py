# -*- coding: utf-8 -*-
"""
@author: L. Hertaeg
"""

# %% import packages and functions

import numpy as np
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap

from Functions_INRateModel import  PhasePlane_SOMVIP

# %% Universal plotting settings

sns.set_style('ticks')
sns.set_context('paper')
my_dpi = 500
minFS = 8
inch = 2.54

PathFig = './Plots/'
    
# %% Plotting functions
    
##### Figure 1 B #####
def Plot_1B(x,rates,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
    
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
       
    plt.plot(x,rates[:,0],color='#9e3039')
    plt.plot(x,rates[:,1],color='#053c5e')
    plt.plot(x,rates[:,2],color='#87b2ce')
    plt.plot(x,rates[:,3],color='#51a76d')
    
    ax = plt.gca()
    ax.tick_params(size=2.0,pad=2.0)
    plt.xlim([-4.0,4.0])
    plt.ylim([0.0,7])
    plt.yticks([0,2,4,6])
    plt.xticks([-4,-2,0,2,4])
    plt.xlabel(r'modulatory input (s$^{-1}$)',fontsize=minFS)
    plt.ylabel(r'rate (s$^{-1}$)',fontsize=minFS)  
    
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_1B.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
    
    
##### Figure 1 D #####
def Plot_1D(x0,r0,x1,r1,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
    
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)  
    
    # Note: for sake of comparability, PC rate in reference network is shifted such that both rates align at x_mod=0      
    y = r1[x1==0.0,0]
    f = interpolate.interp1d(r0[:,0],x0)
    I1 = f(y)  
    
    minR = min(r1[:,0])
    maxR = max(r1[:,0])
    
    g = interpolate.interp1d(x1,r1[:,0],kind='linear',bounds_error=False,fill_value=(minR,maxR))
    plt.plot(x0,g(x0),'-',color='#9e3039')
    plt.plot(x0-I1,r0[:,0],'--',color='#9e3039')
       
    ax = plt.gca()
    ax.tick_params(size=2.0,pad=2.0)
    plt.ylim([minR,maxR+0.1]) # Note: PC in reference net goes down to zero (lower bound) for the parameter configuration chosen in the paper
    plt.xlim([-4.0,4.0])
    plt.yticks([2,2.5,3])
    plt.xticks([-4,-2,0,2,4])
    plt.xlabel(r'modulatory input (s$^{-1}$)',fontsize=minFS)
    plt.ylabel(r'PC rate (s$^{-1}$)',fontsize=minFS)
    plt.legend(['full','ref'],loc=0,fontsize=minFS,handlelength=1.5,borderpad=0.1)
    ax.arrow(0.9, 2.7, 0.8, 0.0, head_width=0.03, head_length=0.1, fc='k', ec='k',lw=2.0)
    
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_1D.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)


##### Figure 1 F #####
def Plot_1F(x_ref,r_ref,x_full,r_full,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
    
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    SI0, SI = r_ref[:,1], r_full[:,1]   # PV rates
    DI0, DI = r_ref[:,2], r_full[:,2]   # SOM rates
         
    id0 = np.max(np.where(SI0==0.0))
    DI0[:id0+1] = DI0[id0] # introduce artificially maximal SOM rate 

    plt.plot(x_full,SI-DI,color='k')
    plt.plot(x_ref,SI0-DI0,'--', color = 'k')  
  
    plt.xlim([-5.0,5.0])
    plt.xticks([-5,0,5])
    plt.yticks([-4,0,4])

    plt.ylim([(SI-DI)[0]+0.15,(SI-DI)[-1]+1.0])
    plt.legend(['full','ref'],loc=0,fontsize=minFS)
    
    ax2 = plt.gca()
    ax2.arrow(0.7, 3.0, 0.7, 0.0, head_width=0.1, head_length=0.1, fc='k', ec='k',lw=2.0)
    ax2.tick_params(size=2.0,pad=2.0)
    ax2.set_xlabel(r'modulatory input (s$^{-1}$)',fontsize=minFS)
    ax2.set_ylabel(r'diff. PV & SOM rate (s$^{-1}$)',fontsize=minFS) 
    
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_1F.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        

##### Figure 1 G #####
def Plot_1G(w,x_ref,r_ref,x_full,r_full,x_KO,r_KO,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)  
    
    SI1, SI0, SI = r_KO[:,:,1], r_ref[:,1], r_full[:,:,1]   # PV rates
    DI1, DI0, DI = r_KO[:,:,2], r_ref[:,2], r_full[:,:,2]   # SOM rates
    
    id0 = np.max(np.where(SI0==0.0))
    DI0[:id0+1] = DI0[id0] # introduce artificially maximal SOM rate 
        
    # compute slope in reference network
    y0, y1 = (SI0-DI0)[0], (SI0-DI0)[-1]
    Func = ((SI0-DI0)-(y0+y1)/2.0)**2    
    id0 = np.argmin(Func)  
    g0, s = np.polyfit(x_ref[id0-1:id0+2], (SI0-DI0)[id0-1:id0+2],1) 
       
    # amplification factor as a function of mutual inhibition strength
    gain_full, gain_KO = np.zeros(len(w)), np.zeros(len(w))
    
    for i in xrange(len(w)):
        
        # control
        Z = SI[i,:]-DI[i,:]
            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_full[i], s = np.polyfit(x_full[id0-1:id0+2], Z[id0-1:id0+2],1)

        # avoid discretization effects 
        x0, x1 = (y0-s)/gain_full[i], (y1-s)/gain_full[i]        
        dx = x1-x0
        if dx<0.05:
            gain_full[i]=np.inf
          
        # SOM2VIP KO
        DInh = SI1[i,:]-DI1[i,:]
            
        y0, y1 = DInh[0], DInh[-1]
        Func = (DInh-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_KO[i], err = np.polyfit(x_KO[id0-1:id0+2], DInh[id0-1:id0+2],1)
        
    
    plt.plot(np.abs(w),np.log2(gain_full/g0),'k.-')
    plt.plot(np.abs(w),np.log2(gain_KO/g0),'.-',color='#a30015')
    plt.annotate(r'VIP $\leftrightarrow $ SOM' + '\n' + '(bidirectional)',xy=(0.4,4.9),fontsize=minFS,color='black')
    plt.annotate(r'VIP $\rightarrow $ SOM' + '\n' + '(feed-forward)',xy=(0.6,-2.8),fontsize=minFS,color='#a30015')
    plt.ylim([min(np.log2(gain_KO/g0))-0.2,max(np.log2(gain_full[~np.isinf(gain_full)]/g0))+0.5])
    plt.xlabel('mutual inhibition strength',fontsize=minFS) # $\mathrm{\hat{w}}$
    plt.ylabel('amplification index',fontsize=minFS) 
    
    ax = plt.gca()
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_yticks([-3,0,3,6])
    ax.set_xticks([0,0.4,0.8,1.2])
    ax.axhspan(0.0, max(np.log2(gain_full[~np.isinf(gain_full)]/g0))+0.5, color='#fcefec')
    ax.axhspan(min(np.log2(gain_KO/g0))-0.2,0.0, color='#f2f3f8')
 
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_1G.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        
        
##### Figure 2 A #####
def Plot_2A(w,x_ref,r_ref,x_no,r_no,x_weak,r_weak,x_strong,r_strong,fig_x=5.0,fig_y=5.0,Do_save=1):        
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    ax = plt.gca()
    ax_in = inset_axes(ax,width="22%",height="30%",loc=4, bbox_to_anchor=(0.1,0.08,1-0.1,0.8), bbox_transform=ax.transAxes)
    
#    bbox_to_anchor=(bbox_ll_x,bbox_ll_y,bbox_w-bbox_ll_x,bbox_h), 
#               loc='upper left',
#               bbox_transform=parent_axes.transAxes
    
    SI0, SI, SI1, SI2 = r_ref[:,1], r_no[:,:,1], r_weak[:,:,1], r_strong[:,:,1]       # PV rates
    DI0, DI, DI1, DI2 = r_ref[:,2], r_no[:,:,2], r_weak[:,:,2], r_strong[:,:,2]       # SOM rates
    
    # reference network
    id0 = np.min(np.where(SI0==0.0))
    DI0[:id0+1] = DI0[id0] # introduce artificially maximal SOM rate
        
    Z = SI0-DI0
    y0, y1 = Z[0], Z[-1]
    Func = (Z-(y0+y1)/2.0)**2  
    id0 = np.argmin(Func)        
    g0, s = np.polyfit(x_ref[id0-1:id0+2], Z[id0-1:id0+2],1)
    
    gain_no, gain_weak, gain_strong = np.zeros(len(w)), np.zeros(len(w)), np.zeros(len(w))
    
    for i in xrange(len(w)):
            
        # no STF
        Z = SI[i,:]-DI[i,:]            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_no[i], err = np.polyfit(x_no[id0-1:id0+2], Z[id0-1:id0+2],1)
    
        # weak STF    
        Z = SI1[i,:]-DI1[i,:]            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_weak[i], err = np.polyfit(x_weak[id0-1:id0+2], Z[id0-1:id0+2],1)
        
        # strong STF    
        Z = SI2[i,:]-DI2[i,:]            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_strong[i], err = np.polyfit(x_strong[id0-1:id0+2], Z[id0-1:id0+2],1)
        
    C = sns.dark_palette('#a30015', n_colors=3,reverse=True)
    ax.plot(np.abs(w),np.log2(gain_strong/g0),'.-',color=C[0])
    ax.plot(np.abs(w),np.log2(gain_weak/g0),'.-',color=C[1])
    ax.plot(np.abs(w),np.log2(gain_no/g0),'k.-')
    ax.legend(['strong STF','weak STF','no STF'],loc=0,fontsize=minFS-2,borderpad=0.01,handlelength=1.0)
    ax.set_xlim([0.0,1.2])
    ax.set_ylim([-4.0,6.7])
    ax.set_xlabel(r'mutual inhibition strength')
    ax.set_ylabel('amplification index')
        
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_yticks([-3,0,3,6])
    ax.set_xticks([0,0.5,1.0])
    ax.axhspan(0.0, 9.0, color='#fcefec')
    ax.axhspan(-4.0,0.0, color='#f2f3f8')
    
    # inset
    x = np.linspace(0,0.6,100) # time in s
    b, c, w = 0.1, 3.0, 0.8 # tau_f in s, rate in 1/s, initial weight 
    u_test = [0.1,0.5]
    
    for j in range(2):
        a = u_test[j]
        k = b*c*(a-1)/(a*b*c+1)
        wss = (b*c+1)/(a*b*c+1)
        y = w*k*np.exp(-(a*b*c+1)*x/b) + w*wss
        ax_in.plot(x,y,color=C[j],lw=1.0)
    ax_in.axhline(0.8,color='k',lw=1.0)
    
    ax_in.set_ylim([0.75,1.05])
    ax_in.tick_params(size=2.0,pad=-0.5,labelsize=7)
    ax_in.set_yticks([0.8,1.0])
    ax_in.set_xticks([])
    ax_in.set_ylabel('weight',fontsize=minFS-2)
    ax_in.set_xlabel('time',fontsize=minFS-2)

    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_2A.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        

##### Figure 2 B #####
def Plot_2B(wr,x_ref,r_ref,x_full,r_full,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)  
    
    SI0, SI = r_ref[:,:,1], r_full[:,:,1]   # PV rates
    DI0, DI = r_ref[:,:,2], r_full[:,:,2]   # SOM rates
    
    # amplification factor as a function of recurrence     
    g0, gain = np.zeros(len(wr)), np.zeros(len(wr))
    
    for i in xrange(len(wr)):
        
        # reference newtork
        id0 = np.min(np.where(SI0[i,:]==0.0))
        DI0[i,:id0+1] = DI0[i,id0] # introduce artificially maximal SOM rate
            
        Z = SI0[i,:]-DI0[i,:]
                
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        g0[i], s = np.polyfit(x_ref[id0-1:id0+2], Z[id0-1:id0+2],1)
    
        # full network
        Z = SI[i,:]-DI[i,:]
            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain[i], err = np.polyfit(x_full[id0-1:id0+2], Z[id0-1:id0+2],1)
     
    plt.plot(np.abs(wr),np.log2(gain/g0),'k.-')
    plt.ylim([min(np.log2(gain/g0))-0.2,max(np.log2(gain/g0))+0.5])
    plt.xlabel('recurrence strength')
    plt.ylabel('amplification index')
    
    ax = plt.gca()
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_yticks([-1,0,1])
    ax.set_xticks([0.2,0.4,0.6,0.8])
    ax.set_xlim(left=0.05)
    ax.set_ylim(top=1.1)

    ax.axhspan(0.0, max(np.log2(gain/g0))+0.5, color='#fcefec')
    ax.axhspan(min(np.log2(gain/g0))-0.2,0.0, color='#f2f3f8') # powderblue')

    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_2B.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        

##### Figure 3 A #####
def Plot_3A(T_in,Aout,Aout_ref,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)

    N = np.size(Aout,0)
    ColAll = sns.light_palette('#633c5d',n_colors=N+1)
    C = ColAll[1:]    

    for j in xrange(N):
        plt.semilogx(1000.0/T_in, np.log2(Aout[j,:]/Aout_ref[j,:]),'.-',color=C[j])
    plt.ylabel('frequency-resolved\namplification index')
    plt.xlabel('frequency (Hz)')
        
    ax = plt.gca()
    ax.set_xlim([1.5,23])
    #ax.set_ylim([0.1,2.1])
    ax.tick_params(axis='x', which='both', size=2.0,pad=2.0)
    ax.tick_params(axis='y', which='both', size=2.0,pad=2.0)
    ax.set_xticks([2, 5, 10])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    axins1 = inset_axes(ax,width="50%", height="5%", loc=1)
    
    cmap = ListedColormap(C)
    cb = mpl.colorbar.ColorbarBase(axins1, cmap=cmap,orientation='horizontal', ticks=[0.1,0.5,0.9])
    cb.outline.set_visible(False)
    cb.ax.set_title('recurrence')
    cb.ax.set_xticklabels(['-0.1', '-0.5', '-0.9'])
    axins1.xaxis.set_ticks_position("bottom")
    axins1.tick_params(size=2.0,pad=2.0)
   
    sns.despine()    
 
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_3A.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)


##### Figure 3 B #####
def Plot_3B(T_in,Aout,Aout_ref,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)

    N = np.size(Aout,0)
    ColAll = sns.light_palette('#042c45',n_colors=N+1)
    C = ColAll[1:] 
    
    for j in xrange(N):
        plt.semilogx(1000.0/T_in, np.log2(Aout[j,:]/Aout_ref[j,:]),'.-',color=C[j])
    plt.ylabel('frequency-resolved\namplification index')
    plt.xlabel('frequency (Hz)')
    
    ax = plt.gca()
    ax.set_xlim([1.5,23])
    ax.set_ylim(bottom=-1.2)
    ax.tick_params(axis='x', which='both', size=2.0,pad=2.0)
    ax.tick_params(axis='y', which='both', size=2.0,pad=2.0)
    ax.set_xticks([2, 5, 10])
    #ax.set_yticks([0.5,1.5,2.5])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    
    ax.arrow(2, -1.0, 0.0, -0.1, lw=1.0, head_width=0.1, head_length=0.05, fc='k', ec='k')
    ax.plot(2,-0.9,'p',color='k')
    ax.arrow(6.7, -1.0, 0.0, -0.1, lw=1.0, head_width=0.35, head_length=0.05, fc='k', ec='k')
    ax.plot(6.666,-0.9,'s',color='k')   
    
    axins1 = inset_axes(ax, width="50%", height="5%", loc=1)
    
    cmap = ListedColormap(C)
    cb = mpl.colorbar.ColorbarBase(axins1, cmap=cmap,orientation='horizontal', ticks=[0.3,0.7])
    cb.outline.set_visible(False)
    cb.ax.set_title('adaptation')
    cb.ax.set_xticklabels(['0.3', '0.7'])
    axins1.xaxis.set_ticks_position("bottom")
    axins1.tick_params(size=2.0,pad=2.0)
    
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_3B.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        
        
##### Figure 3 C #####
def Plot_3C(ada,T_in,Aout,Aout_ref,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    plt.plot(ada,np.log2(Aout[:,T_in==150.0]/Aout_ref[:,T_in==150.0]),'s-',color='k')
    plt.plot(ada,np.log2(Aout[:,T_in==600.0]/Aout_ref[:,T_in==600.0]),'p-',color='k')
    plt.ylabel('frequency-resolved\namplification index')
    plt.xlabel('adaptation strength')
    
    ax = plt.gca()
    #ax.set_ylim([0.6,2.7])
    ax.set_xlim([0.05,1.0])
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_xticks([0.3,0.6,0.9])
    ax.text(0.44,np.log2(1.53),'high-frequency',fontsize=minFS,rotation=12)
    ax.text(0.55,np.log2(1.03),'low-frequency',fontsize=minFS,rotation=-33)

    sns.despine() 

    if Do_save==1:
        fig.savefig(PathFig + 'Fig_3C.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)     
        
        
##### Figure 3 D #####
def Plot_3D(ada,rec,Corr_ada,SEM_ada,Corr_rec,SEM_rec,CMtx_ada,CMtx_rec,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    G01 = gridspec.GridSpec(8,36,hspace=0.1,wspace=0.2)
    ColAda = np.array([0.1029209 , 0.24977696, 0.34156199, 1.0])
    ColRec = np.array([0.44451263, 0.30416095, 0.42292007, 1.0])
    
    ### SOM-SOM correlation
    plt.subplot(G01[1:7,18:]) # SOM-SOM-corr
    ax = plt.gca() 
    
    ada= np.round(ada,2)
    plt.plot(ada,Corr_ada,'.-',color=ColAda,label='ada') # ,markevery=3
    ax.add_patch(patches.Rectangle((0.95, 0.38),0.12,0.03,fill=False,edgecolor=ColAda,lw=1.5))
    ax.arrow(1.0, 0.42, 0.0, 0.015, lw=1.0, head_width=0.03, head_length=0.01, fc=ColAda, ec=ColAda)  
    plt.fill_between(ada, Corr_ada-SEM_ada, Corr_ada+SEM_ada, facecolor=ColAda, alpha=0.3)
    
    rec = np.round(rec,2)
    plt.plot(abs(rec),Corr_rec,'.-',color=ColRec,label='rec') #,markevery=3
    ax.add_patch(patches.Rectangle((0.95, 0.26),0.12,0.03,fill=False,edgecolor=ColRec,lw=1.5))
    ax.arrow(1.0, 0.25, 0.0, -0.015, lw=1.0, head_width=0.03, head_length=0.01, fc=ColRec, ec=ColRec)   
    plt.fill_between(abs(rec), Corr_rec-SEM_rec, Corr_rec+SEM_rec, facecolor=ColRec, alpha=0.3)    
    
    plt.xlim([-0.1,2.0])
    plt.legend(loc=3,handlelength = 1,borderpad=0.1)    
    plt.xlabel('adaptation/recurrence \nstrength')
    plt.ylabel('SOM-SOM corr')
   
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_yticks([0.2,0.4,0.6])
    ax.set_xticks([0,1,2])
    sns.despine()
    
    ### example correlation matrices
    ax2 = plt.subplot(G01[0:3,2:12]) # corr mtx 1
    ax3 = plt.subplot(G01[5:,2:12]) # corr mtx 2
    ax4 = plt.subplot(G01[2:5,0:1]) # colorbar
    
    cmap = 'summer'
    sns.heatmap(CMtx_ada,ax=ax2,square=True,cmap=cmap,vmax=1.0,vmin=-0.5,xticklabels=False,yticklabels=False,cbar=False)
    ax2.add_patch(patches.Rectangle((1, 1),49,49,fill=False,edgecolor=ColAda,lw=1.5))
    
    plt.text(-0.79, 0.42, 'SOM', transform=ax.transAxes, va='top',fontsize=minFS)
    plt.text(-0.55, 0.42, 'VIP', transform=ax.transAxes, va='top',fontsize=minFS)    
    plt.text(-1.1,0.7,'corr', transform=ax.transAxes, va='top',fontsize=minFS,rotation=90)
    plt.text(-1.04,0.92,'1.0', transform=ax.transAxes, va='top',fontsize=minFS)
    plt.text(-1.06,0.32,'-0.5', transform=ax.transAxes, va='top',fontsize=minFS)
        
    sns.heatmap(CMtx_rec,ax=ax3,square=True,cmap=cmap,vmax=1.0,vmin=-0.5,
                xticklabels=False,yticklabels=False,cbar=True,cbar_ax=ax4,
                cbar_kws={"ticks":[]}) # "ticks":[-0.5,1]
    ax3.add_patch(patches.Rectangle((1, 1),49,49,fill=False,edgecolor=ColRec,lw=1.5))
    ax4.tick_params(size=1.0,labelsize=minFS) 
    ax4.yaxis.set_ticks_position('right')    
 
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_3D.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 


##### Figure 3 E and F #####
def Plot_3EF(ada,w,Aout_a,Aout_a_ref,Aout,Aout_ref,T_in,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    G01 = gridspec.GridSpec(8,36,hspace=0.1,wspace=0.2)
    
    ### adaptation strength
    ax1 = plt.subplot(G01[5:,0:12])
    
    N = np.size(Aout_a,0)
    MaxAmp, ResFreq = np.zeros(N), np.zeros(N)
    for j in xrange(N):
        g = interpolate.interp1d(1000.0/T_in, Aout_a[j,:]/Aout_a_ref[j,:],kind='cubic')
        f = 1000.0/np.linspace(T_in[0],T_in[-1],100)
        FreqResp = g(f)
        
        MaxAmp[j] = np.log2(np.max(FreqResp))
        ResFreq[j] = f[np.argmax(FreqResp)]
    
    ax1.plot(ada,MaxAmp,'.-',color='k')
    ax1.tick_params(axis='both', size=2.0,pad=2.0)
    ax1.set_xlabel('adaptation strength')
    ax1.set_ylabel('max. amplification')
    ax1.set_ylim([-1.3,1.7])
    ax1.set_xticks([0.5,0.75])
    #ax1.set_yticks([2,4])
    sns.despine(ax=ax1)
    
    ax11 = ax1.twinx()
    ax11.plot(ada,ResFreq,'.-',color='#9e3039')
    ax11.tick_params(axis='both', size=2.0,pad=2.0)
    ax11.tick_params('y', colors='#9e3039')
    ax11.spines['right'].set_color('#9e3039')
    ax11.set_ylabel('resonance freq.', color='#9e3039')
    ax11.set_ylim([3.5,5.4])
    ax11.set_yticks([4,5])
    sns.despine(ax=ax11,bottom=False,left=False,right=False,top=True)

    ### mutual inhibition
    ax2 = plt.subplot(G01[0:3,0:12])

    N = np.size(Aout,0)    
    MaxAmp, ResFreq = np.zeros(N), np.zeros(N)
    for j in xrange(N):
        
        g = interpolate.interp1d(1000.0/T_in, Aout[j,:]/Aout_ref[j,:])
        f = 1000.0/np.linspace(T_in[0],T_in[-1],100)
        FreqResp = g(f)
        
        MaxAmp[j] = np.log2(np.max(FreqResp))
        ResFreq[j] = f[np.argmax(FreqResp)]
    
    ax2.plot(abs(w),MaxAmp,'.-',color='k')
    ax2.tick_params(axis='both', size=2.0,pad=2.0)
    ax2.set_xlabel('mut. inh. strength')
    ax2.set_ylabel('max. amplification')
    ax2.set_ylim([-1.3,1.7])
    ax2.set_xticks([0.5,0.75])
    #ax2.set_yticks([2,4])
    sns.despine(ax=ax2)
    
    ax21 = ax2.twinx()
    ax21.plot(abs(w),ResFreq,'.-',color='#9e3039')
    ax21.tick_params(axis='both', size=2.0,pad=2.0)
    ax21.tick_params('y', colors='#9e3039')
    ax21.spines['right'].set_color('#9e3039')
    ax21.set_ylabel('resonance freq.', color='#9e3039')
    ax21.set_ylim([3.5,5.4])
    ax21.set_yticks([4,5])
    sns.despine(ax=ax21,bottom=False,left=False,right=False,top=True)
    
    # Schema ...
    ax3 = plt.subplot(G01[1:7,18:])
    
    mu, sigma = 10.0, 1.0
    x = np.linspace(mu - 3*sigma, mu + 3.5*sigma, 100)
    ax3.plot(x,mpl.mlab.normpdf(x, mu, sigma),'--',color=[0.4,0.4,0.4])
    ax3.plot(x,1.3*mpl.mlab.normpdf(x, mu, sigma),color=[0.4,0.4,0.4])    
    
    mu, sigma = 12.0, 1.0
    ax3.plot(x,mpl.mlab.normpdf(x, mu, sigma),color=[0.4,0.4,0.4])
    ax3.text(10.5,0.52,'mutual inhibition',fontsize=minFS,color='k')
    ax3.arrow(10,0.41,0,0.08, lw=1.0, head_width=0.3, head_length=0.02, fc='k', ec='k')  
    ax3.arrow(10,0.40,1.7,0, lw=1.0, head_width=0.02, head_length=0.3, fc='#9e3039', ec='#9e3039')
    ax3.text(12,0.42,'adaptation\nstrength',fontsize=minFS,color='#9e3039')
    
    ax3.tick_params(axis='both', size=2.0,pad=2.0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('frequency',fontsize=minFS)
    ax3.set_ylabel('normalized amplitude',fontsize=minFS)
    ax3.yaxis.set_label_coords(0.1,0.6)
    sns.despine(ax=ax3)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_3EF.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
    

##### Figure S1 #####
def Plot_S1(w,x_ref,r_ref,x_sv,r_sv,x_vs,r_vs,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    ax = plt.gca()
    
    SI0, SI_sv, SI_vs = r_ref[:,1], r_sv[:,:,1], r_vs[:,:,1]   # PV rates
    DI0, DI_sv, DI_vs = r_ref[:,2], r_sv[:,:,2], r_vs[:,:,2]   # SOM rates
    
    id0 = np.max(np.where(SI0==0.0))
    DI0[:id0+1] = DI0[id0] # introduce artificially maximal SOM rate 
        
    # compute slope in reference network
    y0, y1 = (SI0-DI0)[0], (SI0-DI0)[-1]
    Func = ((SI0-DI0)-(y0+y1)/2.0)**2    
    id0 = np.argmin(Func)  
    g0, s = np.polyfit(x_ref[id0-1:id0+2], (SI0-DI0)[id0-1:id0+2],1) 
       
    # amplification factor as a function of mutual inhibition strength
    gain_sv, gain_vs = np.zeros(len(w)), np.zeros(len(w))
    
    for i in xrange(len(w)):
        
        # wsv fixed
        Z = SI_sv[i,:]-DI_sv[i,:]
            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_sv[i], s = np.polyfit(x_sv[id0-1:id0+2], Z[id0-1:id0+2],1)
        
        # avoid discretization effects 
        x0, x1 = (y0-s)/gain_sv[i], (y1-s)/gain_sv[i]        
        dx = x1-x0
        if dx<0.05:
            gain_sv[i]=np.inf
          
        # wvs fixed
        DInh = SI_vs[i,:]-DI_vs[i,:]
            
        y0, y1 = DInh[0], DInh[-1]
        Func = (DInh-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_vs[i], err = np.polyfit(x_vs[id0-1:id0+2], DInh[id0-1:id0+2],1)
        
        # avoid discretization effects 
        x0, x1 = (y0-s)/gain_vs[i], (y1-s)/gain_vs[i]        
        dx = x1-x0
        if dx<0.05:
            gain_vs[i]=np.inf
        
    #gain1[gain1>300] = np.inf
    plt.plot(np.abs(w),np.log2(gain_vs/g0),'.-',color='k') # wvs fixed
    plt.plot(np.abs(w),np.log2(gain_sv/g0),'.-',color=[0.5,0.5,0.5]) # wsv fixed 
    plt.text(0.3,0.2,r'SOM$\rightarrow$VIP' + '\nfixed', transform=ax.transAxes, va='top',fontsize=minFS)
    plt.text(0.4,0.95,r'VIP$\rightarrow$SOM' + '\nfixed', transform=ax.transAxes, va='top',color=[0.5,0.5,0.5],fontsize=minFS)
    plt.ylim([min(np.log2(gain_vs/g0))-0.2,max(np.log2(gain_sv[~np.isinf(gain_sv)]/g0))+0.5])
    plt.xlabel('mutual inhibition strength')
    plt.ylabel('amplification index')       
    
    ax.tick_params(size=2.0,pad=2.0)
    ax.set_xticks([0.3,0.6,0.9])
    ax.set_xlim([0.05,1.05])
    ax.axhspan(0.0, max(np.log2(gain_sv[~np.isinf(gain_sv)]/g0))+0.5, color='#fcefec')
    ax.axhspan(min(np.log2(gain_vs/g0))-0.2,0.0, color='#f2f3f8')
    ax.set_yticks([0,5])
    #ax.set_ylim(top=7.0)
    
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S1.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        
        
##### Figure 3S2 #####
def Plot_S2(wr,x0,r0,x_ref,r_ref,x_vv,r_vv,x_ss,r_ss,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    ax = plt.gca()
    
    SI00, SI0, SI_vv, SI_ss = r0[0,:,1], r_ref[:,:,1], r_vv[:,:,1], r_ss[:,:,1]   # PV rates
    DI00, DI0, DI_vv, DI_ss = r0[0,:,2], r_ref[:,:,2], r_vv[:,:,2], r_ss[:,:,2]   # SOM rates
    
    # control condition - VIPs have recurrence that varies
    id0 = np.min(np.where(SI00==0.0))
    DI00[:id0+1] = DI00[id0] # introduce artificially maximal SOM rate 
    Z = SI00-DI00
            
    y0, y1 = Z[0], Z[-1]
    Func = (Z-(y0+y1)/2.0)**2  
    id0 = np.argmin(Func)        
    g01, s = np.polyfit(x0[id0-1:id0+2], Z[id0-1:id0+2],1)
       
    # amplification factor as a function of mutual inhibition strength
    g0, gain_vv, gain_ss = np.zeros(len(wr)), np.zeros(len(wr)), np.zeros(len(wr))
    
    for i in xrange(len(wr)):
        
        # reference network
        id0 = np.max(np.where(SI0[i,:]==0.0))
        DI0[i,:id0+1] = DI0[i,id0] # introduce artificially maximal SOM rate 
            
        Z = SI0[i,:]-DI0[i,:]
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2    
        id0 = np.argmin(Func)  
        g0[i], s = np.polyfit(x_ref[id0-1:id0+2], Z[id0-1:id0+2],1) 
        
        # wvv fixed
        Z = SI_vv[i,:]-DI_vv[i,:]
            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_vv[i], s = np.polyfit(x_vv[id0-1:id0+2], Z[id0-1:id0+2],1)
          
        # wss fixed
        Z = SI_ss[i,:]-DI_ss[i,:]
            
        y0, y1 = Z[0], Z[-1]
        Func = (Z-(y0+y1)/2.0)**2  
        id0 = np.argmin(Func)        
        gain_ss[i], err = np.polyfit(x_ss[id0-1:id0+2], Z[id0-1:id0+2],1)

        
    #gain1[gain1>300] = np.inf
    plt.plot(np.abs(wr),np.log2(gain_vv/g0),'.-',color='k') # wvv fixed
    plt.plot(np.abs(wr),np.log2(gain_ss/g01),'.-',color=[0.5,0.5,0.5]) # wss fixed 
    plt.text(0.05,0.35,r'VIP$\rightarrow$VIP' + '\nfixed', transform=ax.transAxes, va='top',fontsize=minFS)
    plt.text(0.3,0.85,r'SOM$\rightarrow$SOM' + '\nfixed', transform=ax.transAxes, va='top',color=[0.5,0.5,0.5],fontsize=minFS)
    plt.xlim([0.05,0.95])
    plt.ylim([min(np.log2(gain_ss/g01))-0.2, max(np.log2(gain_ss/g01))+0.25])
    plt.xlabel('recurrence strength')
    plt.ylabel('amplification index')       
    
    ax.tick_params(size=2.0,pad=2.0)
    ax.axhspan(0.0, max(np.log2(gain_ss/g01))+0.25, color='#fcefec')
    ax.axhspan(min(np.log2(gain_ss/g01))-0.2,0.0, color='#f2f3f8')
    ax.set_xticks([0.3,0.6,0.9])
    ax.set_yticks([-1.0,-0.5,0.0])
    
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S2.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        
        
##### Figure 4A - left #####
def Plot_4A_L(w,ada,Osc,WTA,Para,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    ax = plt.gca()
      
    myColors = ['#f2f3f8','#fcefec','#fedac1','#f2c396']
    # attenuation, amplification, WTA, osc WTA
    cmap1= LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    
    PP = np.zeros((len(ada),len(w)))
    for j in xrange(len(ada)):
        
        for i in xrange(len(w)):
            
            w0 = w[i]
            b = abs(w0)*(1+np.sqrt(5))/2 - 1
            
            if ((WTA[j,i]==0.0) & (Osc[j,i]==0.0)): # all IN active
               if (ada[j]<=b): # amplification
                   PP[j,i] = 1
               else: # attenuation
                   PP[j,i] = 0
            elif ((WTA[j,i]>0.0) & (Osc[j,i]==0.0)): # WTA
                PP[j,i] = 2
            elif ((WTA[j,i]==0.0) & (Osc[j,i]>0.0)): # osc WTA
                PP[j,i] = 3
            else:
                PP[j,i] = 4
    
    Text = np.chararray((len(ada),len(w)),itemsize=15)
    Text[:] = ' '
    Text[np.round(ada,2)==0.52,np.round(w,2)==-0.7]=r'$\diamond$' 
    Text[np.round(ada,2)==0.2,np.round(w,2)==-1.3]=r'$\star$' 
    Text[np.round(ada,2)==0.36,np.round(w,2)==-1.3]=r'$\circ$' 
    Text[np.round(ada,2)==0.72,np.round(w,2)==-1.3]=r'$\bullet$' 
    Anno = pd.DataFrame(Text, columns=w, index=ada)
    
    data = pd.DataFrame(PP, columns=np.round(abs(w),2), index=np.round(ada,2)) 
    ax = sns.heatmap(data,cmap=cmap1,square=True,
                     annot=Anno, fmt = '',annot_kws={"fontsize": 12.0},
                     xticklabels=20,yticklabels=20,vmin=0.0,vmax=3.0,cbar=False)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    
    # add first theoretical line - boundary to osc WTA
    tau, tcw = Para[0], Para[1]
    m0 = -1 - tau/tcw
    mb,ab = ax.get_xbound(),ax.get_ybound()
    p = np.polyfit((w[0],w[-1]),mb,1)
    ax.vlines(p[0]*m0+p[1], 0.14*mb[1]/ada[-1], ab[1], color='k', linestyle='-')
    
    # add second theoretical line - transition boundary WTA
    q = np.polyfit((ada[0],ada[-1]),ab,1)
    x0 = np.linspace(-0.1,-1.5,100)
    y0 = abs(x0) - 1
    x1 = p[0]*x0 + p[1]
    y1 = q[0]*y0 + q[1]
    ax.plot(x1,y1,color='k',linestyle='-')
    
    # add line to distinguish between attenuation and amplification
    wi = np.linspace(-(np.sqrt(5)-1)/2,-1.2,100)
    bi = abs(wi)*(1+np.sqrt(5))/2 - 1
    ax.plot(p[0]*wi+p[1],q[0]*bi+q[1],color='k',linestyle='-') #'--')
    
    ax.text(0.05,0.6,'attenuation (a)',transform=ax.transAxes, fontsize=minFS) 
    ax.text(0.45,0.55,'amplification (b)',transform=ax.transAxes, fontsize=minFS, rotation=59)        
    ax.text(0.72,0.02,'switch (c)',transform=ax.transAxes,fontsize=minFS)
    ax.text(0.93,0.92,'oscillation',transform=ax.transAxes,fontsize=minFS,rotation=270)
    ax.text(0.81,0.92,'(d)',transform=ax.transAxes,fontsize=minFS)
    
    ax.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    
    plt.xlabel('mutual inhibition strength',fontsize=minFS)
    plt.ylabel('adaptation strength',fontsize=minFS)
    ax.invert_yaxis()
    
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_4A_L.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 


##### Figure 4A - right #####
def Plot_4A_R(t,rS1,rV1,rS2,rV2,rS3,rV3,rS4,rV4,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(2,2)   
 
    SubT = [r'$\diamond$',r'$\star$',r'$\circ$',r'$\bullet$']
                 
    ax1 = plt.Subplot(fig, G[0,0]) 
    fig.add_subplot(ax1)
    ax1.plot(t,rS1,'-',color='#91b9d2',alpha=0.3)
    ax1.plot(t,rV1,'-',color='#51a76d',alpha=0.3)
    ax1.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax1.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax1.set_ylabel(r'rate (s$^{-1}$)',fontsize=minFS)
    ax1.set_title(SubT[0],fontsize=12)
    ax1.set_xlim([4000,5000])
    ax1.set_ylim([0,25])
    ax1.set_yticks([0,20])
    ax1.set_xticks([])
    sns.despine(ax=ax1)
    
    ax2 = plt.Subplot(fig, G[0,1])  
    fig.add_subplot(ax2)   
    ax2.plot(t,rS2,'-',color='#91b9d2',alpha=0.3)
    ax2.plot(t,rV2,'-',color='#51a76d',alpha=0.3)
    ax2.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax2.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax2.set_title(SubT[1],fontsize=12)
    ax2.text(4600,12.0,'SOM',fontsize=minFS,color='#91b9d2')
    ax2.text(4100,3.1,'VIP',fontsize=minFS,color='#51a76d')
    ax2.set_xlim([4000,5000])
    ax2.set_ylim([0,25])
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2)
    
    ax3 = plt.Subplot(fig, G[1,0]) 
    fig.add_subplot(ax3)      
    ax3.plot(t,rS3,'-',color='#91b9d2',alpha=0.3)
    ax3.plot(t,rV3,'-',color='#51a76d',alpha=0.3)
    ax3.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax3.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax3.set_ylabel(r'rate (s$^{-1}$)',fontsize=minFS)
    ax3.set_title(SubT[2],fontsize=12)
    ax3.set_xlim([4000,5000])
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel('time (s)',fontsize=minFS)
    ax3.set_xticks([4000,4500,5000])
    ax3.set_xticklabels([4,4.5,5])
    ax3.set_ylim([0,25])
    ax3.set_yticks([0,20])
    sns.despine(ax=ax3)
    
    ax4 = plt.Subplot(fig, G[1,1])   
    fig.add_subplot(ax4)      
    ax4.plot(t,rS4,'-',color='#91b9d2',alpha=0.3)
    ax4.plot(t,rV4,'-',color='#51a76d',alpha=0.3)
    ax4.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax4.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax4.set_title(SubT[3],fontsize=12)
    ax4.set_xlim([4000,5000])
    ax4.set_ylim(bottom=0)
    ax4.set_xlabel('time (s)',fontsize=minFS)
    ax4.set_xticks([4000,4500,5000])
    ax4.set_xticklabels([4,4.5,5])
    ax4.set_ylim([0,25])
    ax4.set_yticks([])
    sns.despine(ax=ax4)

    if Do_save==1:
        fig.savefig(PathFig + 'Fig_4A_R.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        
        
##### Figure 4B #####     
def Plot_4B(w,b,tcw,fout_b_1,fout_b_2,fout_tcw_1,fout_tcw_2,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(1,2)  
    
    # adaptation strength
    ax2 = plt.Subplot(fig, G[0,0])
    fig.add_subplot(ax2)
    
    ax2.plot(b,fout_b_2,':k',alpha=1.0)
    x = np.linspace(b[0],b[-1],50)
    y = np.sqrt(4*x/(0.05*0.01) - (1/0.01 - 1/0.05 - abs(w[1])/0.01)**2.0)/(4*np.pi)
    ax2.plot(x,y,'-k',alpha=1.0)    
    
    ax2.plot(b,fout_b_1,':k',alpha=0.5)
    x = np.linspace(b[0],b[-1],50)
    y = np.sqrt(4*x/(0.05*0.01) - (1/0.01 - 1/0.05 - abs(w[0])/0.01)**2.0)/(4*np.pi)
    ax2.plot(x,y,'-k',alpha=0.5)
        
    ax2.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax2.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax2.legend(['sim','theo'],handlelength=1,fontsize=minFS,loc=4)
    
    ax2.set_ylim([0.1,10.0])
    ax2.set_xlim([0.2,2.1])
    ax2.set_xticks([1,2])
    ax2.set_ylabel('osc. freq. (Hz)',fontsize=minFS)
    ax2.set_xlabel('adaptation \nstrength',fontsize=minFS)
    sns.despine(ax=ax2)
    
    # adaptation time constant
    ax3 = plt.Subplot(fig, G[0,1])
    fig.add_subplot(ax3)
    
    ax3.plot(tcw,fout_tcw_1,':k',alpha=0.5) 
    x = np.linspace(tcw[0],tcw[-1],50)
    y = np.sqrt(4*1.0*1000.0/(x*0.01) - (1/0.01 - 1000.0/x - abs(w[0])/0.01)**2.0)/(4*np.pi)
    ax3.plot(x,y,'-k',alpha=0.5)
    
    ax3.plot(tcw,fout_tcw_2,':k',alpha=1.0) 
    x = np.linspace(tcw[0],tcw[-1],50)
    y = np.sqrt(4*1.0*1000.0/(x*0.01) - (1/0.01 - 1000.0/x - abs(w[1])/0.01)**2.0)/(4*np.pi)
    ax3.plot(x,y,'-k',alpha=1.0)
        
    ax3.set_ylim([0.1,10.0])
    ax3.set_yticks([])
    ax3.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax3.tick_params(axis='y', which='both', size=2.0, pad=2.0)  
    ax3.set_xticks([50,100,150])
    ax3.set_xlabel('adaptation time\nconstant (ms)',fontsize=minFS)
    sns.despine(ax=ax3)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_4B.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        

##### Figure 4C #####
def Plot_4C(w,ada,u,tf,Para,Osc,WTA,RS,RV,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    ax = plt.gca()
      
    myColors = ['#f2f3f8','#fcefec','#fedac1','#f2c396']
    # attenuation, amplification, WTA, osc WTA
    cmap1= LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    
    PP = np.zeros((len(ada),len(w)))
    A = 100*np.ones((len(ada),len(w)))
    t = tf/1000.0
    
    for j in xrange(len(ada)):
        for i in xrange(len(w)):
            if ((WTA[j,i]==0.0) & (Osc[j,i]==0.0)):
               # compute approximated amplificatoin index
               q, v = abs(w[i]), ada[j]
               gv = ((1+2*t*RV[j,i])*(1+u*t*RV[j,i])-(1+t*RV[j,i])*u*t*RV[j,i])/(1+u*t*RV[j,i])**2 
               gs = ((1+2*t*RS[j,i])*(1+u*t*RS[j,i])-(1+t*RS[j,i])*u*t*RS[j,i])/(1+u*t*RS[j,i])**2
               A[j,i] = ((1+v)**2 - q**2*gs*gv - q*gv*(1+v))**2     
    idx = np.argmin(A,1)
    
    for j in xrange(len(ada)):
        for i in xrange(len(w)):
            if ((WTA[j,i]==0.0) & (Osc[j,i]==0.0)): # all IN active
               if (i>=idx[j]): # amplification
                   PP[j,i] = 1
               else: # attenuation
                   PP[j,i] = 0
            elif ((WTA[j,i]>0.0) & (Osc[j,i]==0.0)): # WTA
                PP[j,i] = 2
            elif ((WTA[j,i]==0.0) & (Osc[j,i]>0.0)): # osc WTA
                PP[j,i] = 3
            else:
                PP[j,i] = 4 
                
    data = pd.DataFrame(PP, columns=np.round(abs(w),2), index=np.round(ada,2)) 
    ax = sns.heatmap(data,cmap=cmap1,square=True,xticklabels=20,yticklabels=20,vmin=0.0,vmax=3.0,cbar=False) # every third tick is written
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)  

    ax.text(0.03,0.75,'attenuation',transform=ax.transAxes,fontsize=minFS,rotation=70) 
    ax.text(0.12,0.5,'amplification',transform=ax.transAxes,fontsize=minFS,rotation=59)        
    ax.text(0.7,0.3,'switch',transform=ax.transAxes,fontsize=minFS)
    ax.text(0.471,0.92,'osc.',transform=ax.transAxes,fontsize=minFS)         
    
    ax.tick_params(axis='x', which='both', size=2.0)
    ax.tick_params(axis='y', which='both', size=2.0)
    
    plt.xlabel('mutual inhibition strength',fontsize=minFS)
    plt.ylabel('adaptation strength',fontsize=minFS)
    ax.invert_yaxis()              
        
    # add first theoretical line
    mb,ab = ax.get_xbound(),ax.get_ybound()
    q = np.polyfit((ada[0],ada[-1]),ab,1)
    p = np.polyfit((w[0],w[-1]),mb,1)
    X, a  = Para[0], 1.0 + Para[1]/Para[2] # X: mean input, a = 1 + tau/tcw
    
    def Func1(x):
        # x(=b) is optimized
        r = (t*u*X - 1 - x - w + np.sqrt((t*u*X + 1 + x + w)**2 + 4*t*w*X*(1-u)))/(2*t*((1+x)*u + w))
        u0 = (1+t*r)/(1+u*t*r)
        return((a-u0*w)**2)
        
    x0 = np.linspace(-0.685,-1.19,100)                 
    y0 = np.zeros(len(x0))
    for i in range(len(x0)):
        w = abs(x0[i])
        b0 = t*X*(a*u-w)/(w-a) - (1+w/u)
        y0[i] = fmin(Func1,b0)
    x1 = p[0]*x0 + p[1]
    y1 = q[0]*y0 + q[1]
    
    ax.plot(x1,y1,color='k',linestyle='-')
    
    # add second theoretical line    
    x0 = np.linspace(-0.1,-1.19,100) 
    rv = (t*u*X - abs(x0) + np.sqrt((abs(x0)-t*u*X)**2 + 4*X*abs(x0)*t))/(2*abs(x0)*t)
    y0 = X/rv - 1
    x1 = p[0]*x0 + p[1]
    y1 = q[0]*y0 + q[1]
    
    ax.plot(x1,y1,color='k',linestyle='-')
        
    # add line to distinguish attenuation and amplification
    def Func2(x):
        # x(=b) is optimied
        r = (t*u*X - 1 - x - w + np.sqrt((t*u*X + 1 + x + w)**2 + 4*t*w*X*(1-u)))/(2*t*((1+x)*u + w))
        g = ((1+2*t*r)*(1+u*t*r)-(1+t*r)*u*t*r)/(1+u*t*r)**2
        z = (1+x)**2-(g*w)**2 - w*g*(1+x)
        return(z**2)
    
    x0 = np.linspace(-0.1,-0.7,100)                 
    y0 = np.zeros(len(x0))
    for i in range(len(x0)):
        w = abs(x0[i])
        y0[i] = fmin(Func2,0.5)
    x1 = p[0]*x0 + p[1]
    y1 = q[0]*y0 + q[1]
  
    ax.plot(x1,y1,color='k',linestyle='-')
    
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_4C.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        

##### Figure S3 #####      
def Plot_S3(aS,aV,t,RS_all,RV_all,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    NS, NV = len(aS), len(aV)
    gs = gridspec.GridSpec(NS,NV)
    
    for i in range(NS):
        for j in range(NV):
        
            rS = RS_all[str(NV*i+j)]
            rV = RV_all[str(NV*i+j)]
            
            plt.subplot(gs[i, j])
            plt.plot(t,rS,'-',color='#91b9d2',alpha=0.3)
            plt.plot(t,rV,'-',color='#51a76d',alpha=0.3)
            plt.xlim([4000,5000])
            plt.ylim([0,25])
            
            ax = plt.gca()
            ax.tick_params(size=2.0,pad=2.0)
            
            if j>0:
                plt.yticks([])
            else:
                plt.yticks([0,10,20])
                
            if i<NS-1:
                plt.xticks([])
            else:
                ax.set_xticks([4000,4500,5000])
                ax.set_xticklabels([4,4.5,5])

    fig.text(0.5, -0.02, 'time (s)', ha='center')
    fig.text(-0.02, 0.5, r'rate (s$^{-1}$)', va='center', rotation='vertical')
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S3.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 

        
##### Figure S4 #####      
def Plot_S4(TS,TV,t,RS_all,RV_all,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    NS, NV = len(TS), len(TV)
    gs = gridspec.GridSpec(NS,NV)
    
    for i in range(NS):
        for j in range(NV):
        
            rS = RS_all[str(NV*i+j)]
            rV = RV_all[str(NV*i+j)]
            
            plt.subplot(gs[i, j])
            plt.plot(t,rS,'-',color='#91b9d2',alpha=0.3)
            plt.plot(t,rV,'-',color='#51a76d',alpha=0.3)
            plt.xlim([4000,5000])
            plt.ylim([0,25])
            
            ax = plt.gca()
            ax.tick_params(size=2.0,pad=2.0)
            
            if j>0:
                plt.yticks([])
            else:
                plt.yticks([0,10,20])
                
            if i<NS-1:
                plt.xticks([])
            else:
                ax.set_xticks([4000,4500,5000])
                ax.set_xticklabels([4,4.5,5])

    fig.text(0.5, -0.02, 'time (s)', ha='center')
    fig.text(-0.02, 0.5, r'rate (s$^{-1}$)', va='center', rotation='vertical')
    sns.despine()
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S4.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        

##### Figure S5 - left #####        
def Plot_S5_L(rec, w, N, NoAC_SOM, NoAC_VIP, fig_x=5.0, fig_y=5.0, Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    
    myColors = ['#f2f3f8','#fcefec','#fedac1','#dfd6e1','#b6a0ba']
    cmap1 = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
 
    # compute PP
    print 'Interpolation of data'
 
    dm = min(np.abs(np.diff(w)))
    dr = min(np.abs(np.diff(rec)))
    r_new = np.arange(rec[0],rec[-1]-dr,-dr)
    m_new = np.arange(w[0],w[-1]-dm,-dm)
    PP = np.zeros((len(r_new),len(m_new)))
    
    for j in xrange(len(r_new)): #rec_test
        
        for i in xrange(len(m_new)): #m_test
            
            wm = abs(m_new[i])
            r = abs(r_new[j])
            m = (np.abs(w + wm)).argmin()
            n = (np.abs(rec + r)).argmin()
            
            if ((NoAC_SOM[n,m]>1) & (NoAC_VIP[n,m]>1)): # all IN active
                if r <= wm*(1+np.sqrt(5))/2.0 - 1:
                    PP[j,i] = 1 # amplification
                else:
                    PP[j,i] = 0 # attenuation
            elif (((NoAC_SOM[n,m]>1) and (NoAC_VIP[n,m]<=1)) or ((NoAC_SOM[n,m]<=1) and (NoAC_VIP[n,m]>1))): # classical WTA
                PP[j,i] = 2
            elif ((NoAC_SOM[n,m]==1) & (NoAC_VIP[n,m]==1)): # WTA in each pop separately 
                PP[j,i] = 3
            elif (((NoAC_SOM[n,m]==1) and (NoAC_VIP[n,m]==0)) or ((NoAC_SOM[n,m]==0) and (NoAC_VIP[n,m]==1))): # total WTA
                PP[j,i] = 4
            else:
                PP[j,i] = 5
                 
    Text = np.chararray((len(r_new),len(m_new)),itemsize=15)
    Text[:] = ' '
    Text[np.round(r_new,2)==-2.52,np.round(m_new,2)==-1.0]=r'$\diamond$' 
    Text[np.round(r_new,2)==-4.52,np.round(m_new,2)==-2.0]=r'$\circ$' 
    Text[np.round(r_new,2)==-1.52,np.round(m_new,2)==-4.5]=r'$\star$'  
    Text[np.round(r_new,2)==-4.52,np.round(m_new,2)==-5.5]=r'$\bullet$' 
    Anno = pd.DataFrame(Text, columns=m_new, index=r_new)
    
    print 'Create data frame' 
    data = pd.DataFrame(PP, columns=np.round(abs(m_new),1), index=np.round(np.abs(r_new),1)) 
    ax = sns.heatmap(data,cmap=cmap1,annot=Anno,fmt = '',annot_kws={"fontsize": minFS,"color":'k'},xticklabels=140,yticklabels=40,vmin=0.0,vmax=5.0,cbar=False) 
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    
    # transition boundary (transition to pathological states)
    mb,rb = ax.get_xbound(),ax.get_ybound()
    q = np.polyfit((abs(rec[0]),abs(rec[-1])),rb,1)
    p = np.polyfit((w[0],w[-1]),mb,1)
    ax.axhline(q[0]*(N-1)+q[1], color='k', linestyle='-')
    
    # transition boundary (transition to classical WTA)
    x0 = np.linspace(w[0],-5.0,50) 
    y0 = abs(x0) - 1
    x1 = p[0]*x0 + p[1]
    y1 = q[0]*y0 + q[1]  
    ax.plot(x1,y1,color='k',linestyle='-')
    
    # transition between attenuation and amplification
    wi = np.linspace(-(np.sqrt(5)-1)/2,-3.08,100)
    ri = abs(wi)*(1+np.sqrt(5))/2.0 - 1   
    ax.plot(p[0]*wi+p[1],q[0]*ri+q[1],color='k',linestyle='-')
    
    ax.text(0.2,0.94,'pathological states',transform=ax.transAxes, fontsize=minFS) 
    ax.text(0.12,0.6,'attenuation',transform=ax.transAxes, fontsize=minFS,rotation=59)        
    ax.text(0.3,0.6,'amplification',transform=ax.transAxes,fontsize=minFS,rotation=48)
    ax.text(0.7,0.1,'switch',transform=ax.transAxes,fontsize=minFS) 
    
    ax.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    
    plt.xlabel('mutual inhibition strength',fontsize=minFS)
    plt.ylabel('recurrence strength',fontsize=minFS)
    ax.invert_yaxis()
    
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False) 
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S5_L.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        
        
##### Figure S5 - right #####        
def Plot_S5_R(t,rS1,rV1,rS2,rV2,rS3,rV3,rS4,rV4,fig_x=5.0, fig_y=5.0, Do_save=1): 

    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(2,2)   
 
    SubT = [r'$\diamond$',r'$\star$',r'$\circ$',r'$\bullet$']
                 
    ax1 = plt.Subplot(fig, G[0,0]) 
    fig.add_subplot(ax1)
    ax1.plot(t,rS1,'-',color='#91b9d2',alpha=0.8)
    ax1.plot(t,rV1,'-',color='#51a76d',alpha=0.8)
    ax1.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax1.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax1.set_ylabel(r'rate (s$^{-1}$)',fontsize=minFS)
    ax1.set_title(SubT[0],fontsize=12)
    ax1.set_xlim([0,1000])
    ax1.set_ylim([0,12])
    ax1.set_yticks([0,10])
    ax1.set_xticks([])
    sns.despine(ax=ax1)
    
    ax2 = plt.Subplot(fig, G[0,1])  
    fig.add_subplot(ax2)   
    ax2.plot(t,rS2,'-',color='#91b9d2',alpha=0.8)
    ax2.plot(t,rV2,'-',color='#51a76d',alpha=0.8)
    ax2.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax2.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax2.set_title(SubT[1],fontsize=12)
    ax2.text(100,1.0,'SOM',fontsize=minFS,color='#91b9d2')
    ax2.text(600,7.2,'VIP',fontsize=minFS,color='#51a76d')
    ax2.set_xlim([0,1000])
    ax2.set_ylim([0,12])
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2)
    
    ax3 = plt.Subplot(fig, G[1,0]) 
    fig.add_subplot(ax3)      
    ax3.plot(t,rS3,'-',color='#91b9d2',alpha=0.8)
    ax3.plot(t,rV3,'-',color='#51a76d',alpha=0.8)
    ax3.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax3.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax3.set_ylabel(r'rate (s$^{-1}$)',fontsize=minFS)
    ax3.set_title(SubT[2],fontsize=12)
    ax3.set_xlim([0,1000])
    ax3.set_ylim([0,28])
    ax3.set_xlabel('time (s)',fontsize=minFS)
    ax3.set_xticks([0,500,1000])
    ax3.set_xticklabels([0,0.5,1])
    ax3.set_yticks([0,25])
    sns.despine(ax=ax3)
    
    ax4 = plt.Subplot(fig, G[1,1])   
    fig.add_subplot(ax4)      
    ax4.plot(t,rS4,'-',color='#91b9d2',alpha=0.8)
    ax4.plot(t,rV4,'-',color='#51a76d',alpha=0.8)
    ax4.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax4.tick_params(axis='y', which='both', size=2.0, pad=2.0)
    ax4.set_title(SubT[3],fontsize=12)
    ax4.set_xlim([0,1000])
    ax4.set_ylim([0,28])
    ax4.set_xlabel('time (s)',fontsize=minFS)
    ax4.set_xticks([0,500,1000])
    ax4.set_xticklabels([0,0.5,1])
    ax4.set_yticks([])
    sns.despine(ax=ax4)
 
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S5_R.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        

##### Figure S6A #####
def Plot_S6A(x_ctr,r_ctr,x_w,r_w,x_wr,r_wr,x_STF,r_STF,x_ada,r_ada,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(2,2)
    
    # control condition    
    mr_ref = 0.5*(min(r_ctr[:,0])+max(r_ctr[:,0]))
    out_ref = (r_ctr[:,0]-mr_ref)
    
    # mutual inhibition
    ax1 = plt.subplot(G[0,0])
        
    mr = 0.5*(min(r_w[:,0])+max(r_w[:,0]))
    out = (r_w[:,0]-mr)
    ax1.plot(x_w,out/max(out),'-',color='#a30015')
    ax1.plot(x_ctr,out_ref/max(out_ref),'-',color='#d58b94',label='ref')
    ax1.set_title('mutual inh.',fontsize=minFS)
    ax1.set_xlim([-2.0,2.0])
    ax1.set_ylim([-1.05,1.2])
    ax1.set_yticks([-1.0,0,1.0])
    ax1.tick_params(axis='y', size=1.0)
    ax1.set_xticks([])
    sns.despine(ax=ax1)
    ax1.set_ylabel('PC rate',ha='right',fontsize=minFS)
    
    # recurrence
    ax4 = plt.subplot(G[0,1])
    
    mr = 0.5*(min(r_wr[:,0])+max(r_wr[:,0]))
    out = (r_wr[:,0]-mr)
    ax4.plot(x_wr,out/max(out),'-',color='#a30015',label='FB up')
    ax4.plot(x_ctr,out_ref/max(out_ref),'-',color='#d58b94',label='ref')
    ax4.set_title('recurrence',fontsize=minFS)
    ax4.text(-1.5,0.5,'weaker',fontsize=minFS,color='#d58b94')
    ax4.text(-0.1,-0.7,'stronger',fontsize=minFS,color='#a30015')
    ax4.set_xlim([-2.0,2.0])
    ax4.set_ylim([-1.05,1.2])
    ax4.set_yticks([])
    ax4.set_xticks([])
    sns.despine(ax=ax4)
    
    # adaptation
    ax2 = plt.subplot(G[1,1])
    
    mr = 0.5*(min(r_ada[:,0])+max(r_ada[:,0]))
    out = (r_ada[:,0]-mr)
    ax2.plot(x_ada,out/max(out),'-',color='#a30015',label='FB up')
    ax2.plot(x_ctr,out_ref/max(out_ref),'-',color='#d58b94',label='ref')
    ax2.set_title('adaptation',fontsize=minFS)
    ax2.set_xlim([-2.0,2.0])
    ax2.set_xticks([-2,0,2])
    ax2.set_ylim([-1.05,1.2])
    ax2.tick_params(axis='x', size=1.0)
    ax2.set_xlabel(r'mod. input (s$^{-1}$)',fontsize=minFS)
    ax2.set_yticks([])
    sns.despine(ax=ax2)
    
    # STF
    ax3 = plt.subplot(G[1,0])
   
    mr = 0.5*(min(r_STF[:,0])+max(r_STF[:,0]))
    out = (r_STF[:,0]-mr)
    ax3.plot(x_STF,out/max(out),'-',color='#a30015',label='STF up')
    ax3.plot(x_ctr,out_ref/max(out_ref),'-',color='#d58b94',label='ref')
    ax3.set_title('STF',fontsize=minFS)
    ax3.set_xlim([-2.0,2.0])
    ax3.set_xticks([-2,0,2])
    ax3.set_ylim([-1.05,1.2])
    ax3.set_yticks([-1.0,0,1.0])
    ax3.tick_params(axis='both', size=1.0)
    ax3.set_xlabel(r'mod. input (s$^{-1}$)',fontsize=minFS)
    ax3.set_ylabel('normalized',fontsize=minFS,ha='left')
    sns.despine(ax=ax3)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S6A.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig) 
        

##### Figure S6B #####
def Plot_S6B(ada,T_in,Aout,Aout_ref,fig_x=5.0,fig_y=5.0,Do_save=1): 
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi) 
    
    N = np.size(Aout,0)
    ColAll = sns.light_palette('#042c45',n_colors=N+1)
    C = ColAll[1:] 
     
    for j in xrange(N):
        plt.semilogx(1000.0/T_in, np.log2(Aout[j,:]/Aout_ref[j,:]),'.-',color=C[j])
    
    plt.ylabel('frequency-resolved\namplification index',fontsize=minFS)     
    plt.xlabel('frequency (Hz)',fontsize=minFS)
    
    ax2 = plt.gca()
    ax2.tick_params(axis='x', which='both', size=2.0, pad=2.0)
    ax2.tick_params(axis='y', which='both', size=2.0, pad=2.0)
        
    ax2.set_xlim([1.5,23])
    #ax.set_ylim([0.3,2.7])
    ax2.tick_params(axis='x', which='both', size=2.0,pad=2.0)
    ax2.tick_params(axis='y', which='both', size=2.0,pad=2.0)
    ax2.set_xticks([2, 5, 10])
    #ax2.set_yticks([0.1,0.2,0.3,0.4])
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    axins1 = inset_axes(ax2,width="50%",height="5%",loc=3,borderpad=2)
    
    cmap = ListedColormap(C)
    cb = mpl.colorbar.ColorbarBase(axins1, cmap=cmap,orientation='horizontal', ticks=[0.3,0.7])
    cb.outline.set_visible(False)
    cb.ax.set_title('adaptation',fontsize=minFS)
    cb.ax.set_xticklabels(['0.3', '0.7'])
    axins1.xaxis.set_ticks_position("bottom")
    axins1.tick_params(size=2.0,pad=2.0)
    
    sns.despine(ax=ax2)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S6B.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        
        
##### Figure S6C #####
def Plot_S6C(FB, CorrPop_ada, SEM_ada, CorrPop_rec, SEM_rec, fig_x=5.0, fig_y=5.0, Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi) 
    
    ColAda = sns.light_palette('#042c45',n_colors=11)
    ColRec = sns.light_palette('#633c5d',n_colors=11)
    ic = 9
    
    plt.plot(abs(FB),CorrPop_rec[:,2],'.-',color=ColRec[ic],label='rec')
    plt.fill_between(abs(FB), CorrPop_rec[:,2]-SEM_rec[:,2], CorrPop_rec[:,2]+SEM_rec[:,2], facecolor=ColRec[ic], alpha=0.3)  
    plt.plot(FB,CorrPop_ada[:,2],'.-',color=ColAda[ic],label='ada')
    plt.fill_between(FB, CorrPop_ada[:,2]-SEM_ada[:,2], CorrPop_ada[:,2]+SEM_ada[:,2], facecolor=ColAda[ic], alpha=0.3) 
    
    plt.legend(loc=0)
    plt.xlabel('adaptation/recurrence strength',fontsize=minFS)
    plt.ylabel('SOM-SOM corr',fontsize=minFS)
    
    ax = plt.gca()
    ax.tick_params(axis='both', size=2.0)
    ax.set_xlim([-0.1,2.0])
    ax.set_xticks([0,0.5,1,1.5])
    
    sns.despine(ax=ax)  
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S6C.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)


##### Figure 5 B #####
def Plot_5B(x_mod,A,B,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)

    plt.plot(x_mod,A[0,:]/(A[0,:]+B[0,:]),'>-',color='#4c86a8',markevery=3) # som ->
    plt.plot(x_mod,A[1,:]/(A[1,:]+B[1,:]),'<-',color='#4c86a8',markevery=(1,3)) # som <-
    plt.plot(x_mod,B[0,:]/(A[0,:]+B[0,:]),'>-',color='#cb904d',markevery=3) # den ->
    plt.plot(x_mod,B[1,:]/(A[1,:]+B[1,:]),'<-',color='#cb904d',markevery=(1,3)) # den <-
    plt.text(-0.45,0.81,'bottom-up',color='#cb904d',fontsize=minFS)
    plt.text(-0.45,0.11,'top-down',color='#4c86a8',fontsize=minFS)

    ax1 = plt.gca()
    ax1.tick_params(axis='both', size=2.0, pad=2.0)
    ax1.set_yticks([0,0.5,1])
    ax1.set_xticks([-1,0,1])
    ax1.set_ylim([-0.02,1.05])
    ax1.set_xlim([-1.1,1.1])
    ax1.set_ylabel('Rel. transmission\nof input streams',fontsize=minFS)
    ax1.set_xlabel(r'modulatory input (s$^{-1}$)',fontsize=minFS)
    sns.despine()

    if Do_save==1:
        fig.savefig(PathFig + 'Fig_5B.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        
        
##### Figure 5 C #####
def Plot_5C(t,R,x,y,A,B,SimDur,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(4,1,hspace=0.5)
    
    NC = 70, 10, 10, 10
    IE, IB = (25.0/0.07), 100.0
    It = np.linspace(0.0,SimDur,int(2*SimDur))
     
    Re = np.mean(R[:,0:NC[0]],1)
    f = interpolate.interp1d(t,Re)
    RE = f(It)

    ### Input streams
    ax = plt.subplot(G[0,0])
    ax.plot(It,(y-IE)/max(y-IE),color='#cb904d')
    ax.plot(It,(x-IB)/max(x-IB),color='#4c86a8',lw=1.2)
    ax.set_xticks([])
    ax.set_ylabel('normalized\ninputs',fontsize=minFS)
    ax.tick_params(axis='both', size=2.0, pad=2.0)
    ax.set_xlim([0,12000])
    ax.set_ylim([-1.1,1.1])
    ax.set_yticks([-1,0,1])
    sns.despine(ax=ax,bottom=True)
    
    ### PC rate
    ax3 = plt.subplot(G[1:3,0])
    ax3.plot(It,RE,color='#9e3039')
    ax3.arrow(2000,2.76,0,0.15, head_width=120, head_length=0.05, fc='#499f68', ec='#499f68',lw=2.0)
    ax3.arrow(4000,3.0,0,-0.15, head_width=120, head_length=0.05, fc='#499f68', ec='#499f68',lw=2.0)
    ax3.arrow(6000,2.76,0,0.15, head_width=120, head_length=0.05, fc='#499f68', ec='#499f68',lw=2.0)
    ax3.arrow(8000,3,0,-0.15, head_width=120, head_length=0.05, fc='#499f68', ec='#499f68',lw=2.0)
    ax3.arrow(10000,2.76,0,0.15, head_width=120, head_length=0.05, fc='#499f68', ec='#499f68',lw=2.0)
    ax3.text(250,2.76,'pulses\nonto VIP',color='#499f68',fontsize=minFS)
    ax3.text(300,1.5,'SOM on\nVIP off',color='k',fontsize=minFS)
    ax3.text(4300,1.5,'SOM on\nVIP off',color='k',fontsize=minFS)
    ax3.text(8300,1.5,'SOM on\nVIP off',color='k',fontsize=minFS)
    ax3.text(2300,2.1,'SOM off\nVIP on',color='k',fontsize=minFS)
    ax3.text(6300,2.1,'SOM off\nVIP on',color='k',fontsize=minFS)
    ax3.text(10300,2.1,'SOM off\nVIP on',color='k',fontsize=minFS)

    ax3.set_ylabel(r'PC rate (s$^{-1}$)',fontsize=minFS)
    ax3.set_xticks([])
    ax3.tick_params(axis='both', size=2.0, pad=2.0)
    ax3.set_xlim([0,12000])
    ax3.set_ylim([1.5,3.06])
    ax3.set_yticks([1.5,2.0,2.5,3.0])
    sns.despine(ax=ax3,bottom=True)
    
    ax4 = plt.subplot(G[3,0])
    ax4.plot(It,A/(A+B),color='#4c86a8')
    ax4.tick_params(axis='both', size=2.0)
    ax4.set_xlim([0,12000])
    ax4.set_ylim([-0.02,0.4])
    ax4.set_yticks([0.0,0.2,0.4])
    ax4.set_yticklabels([0.0,0.2,0.4])
    ax4.set_xticks([0,2000,4000,6000,8000,10000,12000])
    ax4.set_xticklabels([0,2,4,6,8,10,12])
    ax4.set_ylabel('Rel. contribution\ntop-down',fontsize=minFS)   #  (r'$\frac{\alpha}{\alpha+\beta}$') 
    ax4.set_xlabel('time (s)',fontsize=minFS)
    sns.despine(ax=ax4)

    if Do_save==1:
        fig.savefig(PathFig + 'Fig_5C.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        

##### Figure S7 #####
def Plot_S7(w_IN,x_IN,SI,DI,x_full,rates,fig_x=5.0,fig_y=5.0,Do_save=1):
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(3,6)
    
    # example phase planes
    x = np.zeros(2)
    x[1] = 3 - w_IN*3
    x[0] = 3 - w_IN*3
    
    plt.subplot(G[0,0:2])
    PhasePlane_SOMVIP(np.abs([w_IN,w_IN]),x+[0,-0.4],minFS)
    ax = plt.gca()
    plt.text(-0.4, 1.2, 'A', transform=ax.transAxes, va='top',fontweight='bold')
    
    plt.subplot(G[0,2:4])
    PhasePlane_SOMVIP(np.abs([w_IN,w_IN]),x+[0,0.0],minFS)
    
    plt.subplot(G[0,4:])
    PhasePlane_SOMVIP(np.abs([w_IN,w_IN]),x+[0,0.4],minFS)
    
    # Hysterese in IN-net
    plt.subplot(G[1,1:5])
        
    plt.plot(x_IN, SI[0,:]-DI[0,:],'k>-',Markersize=4,markevery=3,zorder=9)
    plt.plot(x_IN, SI[1,:]-DI[1,:],'k<-',Markersize=4,markevery=(1,3),zorder=9)
    plt.xlabel(r'modulatory input (s$^{-1}$)',fontsize=minFS)
    plt.ylabel('difference\nPV & SOM rate ' + r'(s$^{-1}$)',fontsize=minFS)
    plt.xlim([-0.55,0.55])
    plt.ylim([-5.1,5.1])
    
    ax = plt.gca()
    ax.set_xticks([-0.4,-0.2,0.0,0.2,0.4])
    ax.set_yticks([-5,0,5])
    ax.axvline(x=0.4,color=[0.8,0.8,0.8],lw=2,zorder=0)
    ax.annotate('', xy=(0.397, 4.8), xytext=(0.45, 9.0),arrowprops=dict(arrowstyle="<-", color=[0.8,0.8,0.8],lw=2),zorder=0)
    ax.axvline(x=-0.4,color=[0.8,0.8,0.8],lw=2,zorder=0)
    ax.annotate('', xy=(-0.397, 4.8), xytext=(-0.45, 9.0),arrowprops=dict(arrowstyle="<-", color=[0.8,0.8,0.8],lw=2),zorder=0)
    ax.axvline(x=0.0,color=[0.8,0.8,0.8],lw=2,zorder=0)
    ax.annotate('', xy=(0.0, 4.8), xytext=(0.0, 7.0),arrowprops=dict(arrowstyle="<-", color=[0.8,0.8,0.8],lw=2),zorder=0)
    
    sns.despine()
    
    # Hysterese in full microcircuit
    plt.subplot(G[2,1:5])
    ax = plt.gca()
        
    plt.plot(x_full, rates[0,:],'>-',Markersize=4,color='#9e3039',markevery=3)
    plt.plot(x_full, rates[1,:],'<-',Markersize=4,color='#9e3039',markevery=(1,3))
    plt.xlabel(r'modulatory input (s$^{-1}$)',fontsize=minFS)
    plt.ylabel(r'PC rate (s$^{-1}$)',fontsize=minFS)
    plt.text(-0.25, 1.2, 'B', transform=ax.transAxes, va='top',fontweight='bold')
    plt.ylim([1.4,3.0])
    plt.xlim([-0.55,0.55])
    
    ax.set_xticks([-0.4,-0.2,0.0,0.2,0.4])
    ax.set_yticks([1.5,2.0,2.5,3.0])

    sns.despine()
     
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_S7.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)


##### Figure 6 A #####
def Plot_6A(x_SOM,x_VIP,rates,fig_x=5.0,fig_y=5.0,Do_save=1):    
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(2,1)
    
    plt.subplot(G[0,0])

    C = sns.dark_palette('#a30015', n_colors=len(x_SOM))
    C1 = sns.dark_palette('#7bc950', n_colors=len(x_SOM))
    plt.axhline(y=0.5,color=[0.7,0.7,0.7],ls='--')
                  
    for i in range(0,len(x_SOM),3):
        norm = max(rates[:,i])-min(rates[:,i])
        r = rates[:,i]/norm
        plt.plot(x_VIP,r-min(r),'-',color=C[i])
        f = interpolate.interp1d(r-min(r), x_VIP)
        I = f(0.5)
        plt.plot(I,0.5,'.',color=C1[i], ms=6, clip_on=False, zorder=5)
  
    ax = plt.gca()
    ax.arrow(0,0.42,1.7,0,head_width=0.05, head_length=0.2, fc='k', ec='k',lw=1.0)
    ax.text(0,0.15,'mod. input\nonto SOM',fontsize=minFS)
    ax.tick_params(axis='both', size=2.0)
    ax.set_yticks([0,0.5,1])
    ax.set_xticks([-2,0,2])
    ax.set_ylabel('normalized\nPC rate ' +  r'(s$^{-1}$)',fontsize=minFS)
    ax.set_xlabel('modulatory input onto VIP',fontsize=minFS)
    ax.set_ylim(bottom=0)
    ax.set_xlim([-2.8,2.0])
    sns.despine(ax=ax)
    

    plt.subplot(G[1,0])
    plt.axhline(y=0.5,color=[0.7,0.7,0.7],ls='--')
                    
    for i in range(0,len(x_SOM),3):
        norm = max(rates[:,i])-min(rates[:,i])
        r = rates[:,i]/norm
        plt.plot(x_VIP-x_SOM[i],r-min(r),'-',color=C[i])
        f = interpolate.interp1d(r-min(r), x_VIP-x_SOM[i])
        I = f(0.5)
        plt.plot(I,0.5,'.',color=C1[i], ms=6, clip_on=False, zorder=5)
  
    ax = plt.gca()
    ax.tick_params(axis='both', size=2.0)
    ax.set_yticks([0,0.5,1])
    ax.set_xticks([-2,0,2])
    ax.set_ylabel('normalized\nPC rate ' +  r'(s$^{-1}$)',fontsize=minFS)
    ax.set_xlabel('diff. mod. input VIP & SOM',fontsize=minFS)
    ax.set_ylim(bottom=0)
    ax.set_xlim([-2.8,2.0])
    sns.despine(ax=ax)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_6A.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        

##### Figure 6 C & D #####
def Plot_6CD(t,R_mis,Xv_mis,Xm_mis,R_play,Xv_play,Xm_play,SimDur,fig_x=5.0,fig_y=5.0,Do_save=1):    
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    G = gridspec.GridSpec(1,2)
    
    It = np.linspace(0.0,SimDur,int(2*SimDur))
    NC = 70, 10, 10, 10 
     
    dI = Xm_mis-Xv_mis
    Ti = np.where(np.diff(dI)>0)[0]
    Tj = np.where(np.diff(dI)<0)[0]
    
    ### mismatch
    ax1 = plt.subplot(G[0,0])
    for i in range(len(Ti)):
        ax1.axvspan(It[Ti[i]],It[Tj[i]],ymax=0.75,color=[0.8,0.8,0.8],alpha=0.2)
    
    Z = np.mean(R_mis[:,:NC[0]],1)
    M1 = np.mean(Z[(t>=4000.0) & (t<7000.0)])
    y1 = Z-M1
    Max1 = np.max(y1)
    
    ax1.plot(t,y1/Max1+2.16,'#9e3039')
    ax1.add_patch(patches.Rectangle((5500.0, 2.8),1000,0.03,fill=True,ec='black',fc='black',lw=1))
    ax1.text(5500,2.7,'1 s', va='top', ha='left',fontsize=minFS-2)
    dr = (max(R_mis[:,0])-min(R_mis[:,0]))
    ds = 1.0/dr
    ax1.add_patch(patches.Rectangle((5500.0, 2.8),30,ds,fill=True,ec='black',fc='black',lw=1))
    ax1.text(3850,2.7+ds,r'1 s$^{-1}$', va='top', ha='left',fontsize=minFS-2)
    ax1.text(-1300,2.16,'PC', va='center', ha='left',fontsize=minFS,color='#9e3039') 
    ax1.text(-1300,0.9,'V', va='center', ha='left',fontsize=minFS,color='#cb904d')
    ax1.text(-1300,0.0,'M', va='center', ha='left',fontsize=minFS,color='#4c86a8')

    ax1.plot(It,Xv_mis/max(2*Xv_mis)+0.9,'#cb904d')
    ax1.plot(It,Xm_mis/max(2*Xm_mis),'#4c86a8',clip_on=False)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xlim(left=100.0)
    ax1.set_ylim([0,4.6])
    ax1.text(0.5,0.9,'Mismatch',transform=ax1.transAxes, va='top', ha='center',fontsize=minFS)
    sns.despine(ax=ax1,bottom=True,top=True,left=True,right=True)
    
    ### playback
    ax2 = plt.subplot(G[0,1])
    
    for i in range(len(Ti)):
        ax2.axvspan(It[Ti[i]],It[Tj[i]],ymin=0.0,ymax=0.75,color=[0.8,0.8,0.8],alpha=0.2)
    
    Z = np.mean(R_play[:,:NC[0]],1)  # R[:,0]
    M1 = np.mean(Z[(t>=4000.0) & (t<7000.0)])
    y1 = Z-M1
    ax2.plot(t,y1/Max1+2.16,'#9e3039')
    #ax2.text(-1300,2.16,'PC', va='center', ha='left',fontsize=minFS,color='#9e3039')
    
    ax2.plot(It,Xv_play/max(2*Xv_play)+0.9,'#cb904d')
    ax2.plot(It,Xm_play,'#4c86a8',clip_on=False) # do not divide by max as it is zero 
    #ax2.text(-1300,0.9,'V', va='center', ha='left',fontsize=minFS,color='#cb904d')
    #ax2.text(-1300,0.0,'M', va='center', ha='left',fontsize=minFS,color='#4c86a8')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(left=100.0)
    ax2.set_ylim([0,4.6])
    ax2.text(0.5,0.9,'Playback',transform=ax2.transAxes, va='top', ha='center',fontsize=minFS)
    sns.despine(ax=ax2,bottom=True,top=True,left=True,right=True)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_6CD.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
 

##### Figure 6E #####
def Plot_6E(t,R_weak,R_strong,SimDur,fig_x=5.0,fig_y=5.0,Do_save=1):  
    
    if Do_save==1:
        plt.ioff()
        
    fig = plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(fig_x/inch, fig_y/inch), tight_layout='true',dpi=my_dpi)
    ax1 = plt.gca()
    
    NC = 70, 10, 10, 10
    C = ['#afa39a','#9e3039']
    
    ax1.axvspan(1450.0, 1550.0, color=[0.8,0.8,0.8],alpha=0.1)
    ax1.plot(t,np.mean(R_weak[:,:NC[0]],1),color=C[0])
    ax1.plot(t,np.mean(R_strong[:,:NC[0]],1),color=C[1])
        
    ax1.set_title('mut. inhibition',fontsize=minFS)
    ax1.set_xlim([500,2500])
    ax1.set_ylim([1.0,3.6])
    ax1.tick_params(size=2.0)
    ax1.set_xticks([1000,1500,2000])
    ax1.set_yticks([1,2,3])
    ax1.text(900,1.2,'weaker', va='center', ha='left',fontsize=minFS,color='#afa39a')
    ax1.text(1650,2.2,'stronger', va='center', ha='left',fontsize=minFS,color='#9e3039')
    ax1.set_xlabel('time (ms)',fontsize=minFS)
    ax1.set_ylabel(r'PC rate (s$^{-1}$)',fontsize=minFS)
    sns.despine(ax=ax1)
    
    if Do_save==1:
        fig.savefig(PathFig + 'Fig_6E.png', bbox_inches='tight',transparent=True,dpi=my_dpi)
        plt.close(fig)
        