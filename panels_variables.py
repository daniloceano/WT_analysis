#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:13:39 2021

@author: danilocoutodsouza
"""

import SLP_maps as smaps

import numpy as np 
import pylab as pl
import cmocean.cm as cmo

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


import maps

# ---------------    
def plot_var(ax,WT,proj,var):
    # cmap for rain and wind
    col_hcl = [
         [0.9921568627450981, 0.6588235294117647, 0.7058823529411765],       
         [0.9294117647058824, 0.4392156862745098, 0.6627450980392157],       
         [0.8, 0.16470588235294117, 0.6470588235294118], 
         [0.5294117647058824, 0.058823529411764705, 0.5254901960784314],  
         [0.36470588235294116,0.1568627450980392, 0.39215686274509803],  
         [0.3215686274509804, 0.2549019607843137, 0.4549019607843137],      
         [0.1843137254901961, 0.4627450980392157, 0.5725490196078431], 
         [0.0, 0.5843137254901961, 0.6862745098039216],
         [0.09411764705882353, 0.7411764705882353, 0.6901960784313725],
         [0.9450980392156862, 0.9450980392156862, 0.9450980392156862]
         ]   
    col_hcl.reverse()
    cmap = LinearSegmentedColormap.from_list(
            'MyMap', col_hcl, N=20)
    cmap.set_under('white')
    # 
    lims = [-54, -44.05, -34, -25.05]
    data = smaps.get_OLAM_data(WT)
    if var in ['slp','wind']: 
        time = data.time[5]
    else:
        time = data.time[-1]
    data =  data.sel(lon=slice(lims[0],lims[1]), 
                     lat=slice(lims[2],lims[3]),
                     time=time) 
    slp = data.sslp/100
    prec = data.pt
    u = data.uwnd
    v = data.vwnd
    lat = slp.lat
    lon = slp.lon
    ws = np.sqrt(u**2 + v**2)
    if var == 'slp':
        # set limits for plotting
        cmap = cmo.balance
        levels = np.arange(1000,1036,2)
        norm = colors.DivergingNorm(vmin=1000, vcenter=1014, vmax=1035)
        cf1 = ax.contourf(lon, lat, slp, levels=levels, cmap=cmap,
                          norm=norm) 
        ax.contour(lon, lat, slp, cf1.levels,colors='grey', linewidths=1) 
        qv = ax.quiver(lon[::20],lat[::20],u[::20,::20],
                       v[::20,::20],color= 'k')
        ax.quiverkey(qv,-0.3, 1.07, 10, r'$10 \frac{m}{s}$', labelpos = 'E',
                           coordinates='axes', labelsep = 0.05,
                           fontproperties={'size': 14, 'weight': 'bold'})
    if var == 'prec':
        # set limits for plotting
        levels = np.arange(0,301,25)
        norm = plt.Normalize(0, 300)
        cf1 = ax.contourf(lon, lat, prec, levels=levels, cmap=cmap, 
                          norm=norm) 
        
    if var == 'wind':
        # set limits for plotting
        levels = np.arange(0,23,2)
        norm = plt.Normalize(0, 22)
        cf1 = ax.contourf(lon, lat, ws, levels=levels, cmap=cmap, norm=norm) 
        ax.contour(lon, lat, ws, cf1.levels,colors='grey', linewidths=1) 
        qv = ax.quiver(lon[::20],lat[::20],u[::20,::20],
                       v[::20,::20],color= 'k')
        ax.quiverkey(qv,-0.3, 1.07, 10, r'$10 \frac{m}{s}$', labelpos = 'E',
                           coordinates='axes', labelsep = 0.05,
                           fontproperties={'size': 14, 'weight': 'bold'})
    return cf1    
        
# ------------------   
def grid_labels_params(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator(range(-35,-25,2))
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray', 'rotation' : None}
    ax.outline_patch.set_edgecolor('gray')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax

# ---------------    
def main():
    for var in ['slp','prec','wind']:
        proj = ccrs.PlateCarree() 
        lims = [-54, -44.05, -34, -25.05]
        fig = pl.figure(constrained_layout=False,figsize=(15,12))
        gs = gridspec.GridSpec(ncols=6, nrows=6, figure=fig,
                             left= 0.05,right= 0.9, top = 0.95, bottom = 0.05,
                             wspace=0.2, hspace=0.2) 
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        wt = 1
        for col in range(6):
            for row in range(6):
                ax = (fig.add_subplot(gs[row, col], projection=proj))
                ax.set_extent(lims) 
                # Plot variable
                cf1 = plot_var(ax,wt,proj,var)
                # Cosmedics
                grid_labels_params(ax)
                maps.map_features(ax)
                maps.Brazil_states(ax)
                ax.text(0.05,0.8,str(wt), transform=ax.transAxes, fontsize=16,bbox=props)
                wt += 1
        # colorbar
        
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1+0.008, pos.y0*4.4, 0.02, pos.height*4])
        if var == 'slp':
            norm = maps.MidpointNormalize(vmin=1000, vcenter=1014, vmax=1035)
            cbar = plt.colorbar(cf1, cax=cbar_ax, orientation='vertical',
                                norm=norm)
            cbar.ax.tick_params(labelsize=12) 
            cbar.mappable.set_clim(1000,1035)
            cbar.ax.set_title('(hPa)', rotation=0, fontsize= 14)
        elif var == 'prec':
            norm = plt.Normalize(0, 300)
            cbar = plt.colorbar(cf1, cax=cbar_ax,
                                orientation='vertical', extend='max',
                                norm=norm)
            cbar.ax.tick_params(labelsize=12) 
            cbar.ax.set_title('(mm)', rotation=0, fontsize= 14)
        elif var == 'wind':
            norm = plt.Normalize(0, 22)
            cbar = plt.colorbar(cf1, cax=cbar_ax,
                                orientation='vertical', norm=norm,
                                extend='max')
            cbar.ax.tick_params(labelsize=12) 
            cbar.ax.set_title('(m/s)', rotation=0, fontsize= 14)
            
        pl.savefig('../Figures/panels/'+var+'.png', format='png')
        pl.savefig('../Figures/panels/'+var+'.tiff', format='tiff', dpi=600)
        
# ------------------
if __name__ == "__main__": 
    main()
        