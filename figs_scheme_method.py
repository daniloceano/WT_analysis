#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:24:57 2021

@author: danilocoutodsouza
"""

import numpy as np
import pylab as pl
import SLP_maps as smaps
import cmocean.cm as cmo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import maps


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from itertools import cycle


sel_wts = [2,22,28,34]
times = [3,3,3,2]

# ---------------    
def plot_SLP(ax,WT,proj,time,source):
    lims = [-70, 0, -62, -25]
    # get data
    if source == 'cfsr':
        data = smaps.get_CFSR_data(WT)
        data = smaps.convert_lon(data)
        data =  data.sel(longitude=slice(lims[0]-10,lims[1]+10), 
                          latitude=slice(lims[3]+30,lims[2]-10))
        slp = data.prmsl/100
        lat = slp.latitude
        lon = slp.longitude
    elif source == 'olam':
        lims_olam = [-54, -44.05, -34, -25.05]
        data = smaps.get_OLAM_data(WT)
        data =  data.sel(lon=slice(lims_olam[0],lims_olam[1]), 
                         lat=slice(lims_olam[2],lims_olam[3]))   
        slp = data.sslp/100
        lat = slp.lat
        lon = slp.lon
    # set limits for plotting
    min_ = round(int(np.amin(slp).values))
    max_ = round(int(np.amax(slp).values))+5
    norm = maps.MidpointNormalize(vmin=min_, vcenter=1014, vmax=max_)
    ax.pcolormesh(lon, lat, slp[time], cmap=cmo.balance,
                norm=norm, shading='nearest', transform=proj)
    
   
def plot_SLP_globe(ax,WT,proj,time):
    lims = [-70, 0, -62, -25]
    # get data
    data = smaps.get_CFSR_data(WT)
    data = smaps.convert_lon(data)
    slp = data.prmsl/100
    lat = slp.latitude
    lon = slp.longitude
    # set limits for plotting
    min_ = round(int(np.amin(slp).values))
    max_ = round(int(np.amax(slp).values))
    norm = maps.MidpointNormalize(vmin=min_, vcenter=1014, vmax=max_)
    # plot data for the entire globe
    ax.pcolormesh(lon, lat, slp[time], cmap=cmo.balance,
                norm=norm, shading='nearest', transform=proj, alpha = 0.25)
    # slice data (area used for the analysis)
    data =  data.sel(longitude=slice(lims[0],lims[1]), 
                          latitude=slice(lims[3],lims[2]))
    slp = data.prmsl/100
    lat = slp.latitude
    lon = slp.longitude
    # set limits for plotting
    min_ = round(int(np.amin(slp).values))
    max_ = round(int(np.amax(slp).values))
    norm = maps.MidpointNormalize(vmin=min_, vcenter=1014, vmax=max_)
    # plot sliced data
    ax.pcolormesh(lon, lat, slp[time], cmap=cmo.balance,
                norm=norm, shading='nearest', transform=proj)

# ---------------     
def globes():
    # Make figure
    proj = ccrs.PlateCarree()
    ortho =ccrs.Orthographic(central_longitude=-40,central_latitude=-20)
    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(2, 2, hspace=0, wspace=0)
    
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    
    for i, wt, t in zip(range(4),sel_wts, times):
        ax = (fig.add_subplot(gs[i], projection=ortho))
        # Plot SLP and wind
        plot_SLP_globe(ax,wt,proj,t)
        # Draw boxes for analysis
        l = 1
        smaps.draw_box(ax,proj,-70, 0, -62, -25, 'k-',3)
        # smaps.draw_box(ax,proj,-69, -34, -43, -26, 'k--',l)
        # smaps.draw_box(ax,proj,-69, -34, -61, -44, 'k--',l)
        # smaps.draw_box(ax,proj,-33, -1, -43, -26, 'k--',l)
        # smaps.draw_box(ax,proj,-33, -1,  -61, -44, 'k--',l)
        # Cosmedics
        ax.coastlines()
        ax.gridlines()
        # if wt == 1:
        #     ax.text(0,.8,'A', transform=ax.transAxes, fontsize=18,bbox=props)
            
    pl.savefig('../Figures/scheme_methodology/globes.png', format='png',transparent=True)     
    
def KKM_demo():
    '''
    
    Demo of K-means method
    
    Adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
    '''
    # Generate sample data
    rng = np.random.default_rng(12345)
    centers = []
    for i in range(36):
        tmp = rng.integers(low=-100, high=100, size=2)
        tmp = [tmp[0],tmp[1]]
        centers.append(tmp)
    X, _ = make_blobs(n_samples=5000, centers=centers, cluster_std=5)
    
    # Compute clustering with MeanShift
    
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
        
    # Plot result
    fig, ax = plt.subplots(figsize=(5,5) , tight_layout = False)
    cmap = plt.cm.get_cmap(cmo.phase)
    colors = []
    for i in range(10):
        colors.append(cmap(i/10))
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1],'.', color=col,alpha=0.6)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    # plot cosmedics             
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PC1',fontsize=18)
    plt.ylabel('PC2',fontsize=18)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    # plt.text(-0.1,.9,'B', transform=ax.transAxes, fontsize=18,bbox=props)
    pl.savefig('../Figures/scheme_methodology/kkm.png', format='png',
               bbox_inches = 'tight', pad_inches = 0.1) 


def make_map_CFSR():
    # Make figure
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(5.75,5))
    gs = gridspec.GridSpec(2, 2, hspace=0, wspace=0)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    
    for i, wt, t in zip(range(4),sel_wts, times):
        ax = (fig.add_subplot(gs[i], projection=proj))
        ax.set_extent([-80, 10, -63, 5]) 
        # Plot SLP and wind
        plot_SLP(ax,wt,proj,t,'cfsr')
        # Draw boxes for analysis
        l = 1
        smaps.draw_box(ax,proj,-70, 0, -62, -25, 'k-',3) 
        smaps.draw_box(ax,proj,-68, -34.5, -43.5, -27, 'k--',l) #top left
        smaps.draw_box(ax,proj,-68, -34.5, -60, -44.5, 'k--',l) # bottom left
        smaps.draw_box(ax,proj,-32.5, -3, -43.5, -27, 'k--',l) # top right
        smaps.draw_box(ax,proj,-32.5, -3,  -60, -44.5, 'k--',l) # bottom right
        # Cosmedics
        ax.coastlines()
        # ax.set_aspect('equal')
        # if wt == 1:
        #     ax.text(-0.2,.8,'C', transform=ax.transAxes, fontsize=18,bbox=props)
            
    pl.savefig('../Figures/scheme_methodology/cfsr.png', format='png',transparent=True) 

      
# ---------------
def make_map_OLAM():
    lims = [-54, -44.05, -34, -25.05]
    # Make figure
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(5.5,5))        
    gs = gridspec.GridSpec(2, 2, hspace=0, wspace=0)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Loop for 00, 06, 12 and 18
    for i, wt, t in zip(range(4),sel_wts, times):
        ax = (fig.add_subplot(gs[i], projection=proj))
        ax.set_extent(lims) 
        # Plot SLP and wind
        plot_SLP(ax,wt,proj,t*2,'olam')
        # Cosmedics
        maps.map_features(ax)
        maps.Brazil_states(ax)
        # if wt == 1:
        #     ax.text(-0.2,.8,'D', transform=ax.transAxes, fontsize=18,bbox=props)
            
    pl.savefig('../Figures/scheme_methodology/olam.png', format='png',transparent=True) 
    
# ---------------  
def main():     
    globes()
    KKM_demo()
    make_map_CFSR()    
    make_map_OLAM()
    
# ---------------
if __name__ == "__main__": 
    main()    