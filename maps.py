#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:39:08 2021

Create South America map and Southern Brazil mini map with surface ellvation

@author: danilocoutodsouza
"""

import csv

import numpy as np
import pylab as pl
import xarray as xr
import cmocean.cm as cmo

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, COASTLINE
from cartopy.feature import BORDERS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ------------------
def map_features(ax):
    ax.add_feature(COASTLINE)
    ax.add_feature(BORDERS, edgecolor='gray')
    return ax
 
# ------------------   
def grid_labels_params(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 14, 'color': 'gray'}
    gl.ylabel_style = {'size': 14, 'color': 'gray'}
    ax.outline_patch.set_edgecolor('gray')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


# ------------------
def Brazil_states(ax):    
    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                                  name='admin_1_states_provinces_lines')
    _ = ax.add_feature(states, edgecolor='gray')
    
    cities = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none',
                                  name='populated_places')
    _ = ax.add_feature(cities)
    
# ------------------    
def highlight_state(ax,state_acronym):
    lon = []
    lat = []
    # get lon from csv
    with open('../'+str(state_acronym)+'.ll') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            lon.append(float(f'{row[0]}'))
            lat.append(float(f'{row[1]}'))
            line_count += 1
    ax.fill(lon,lat, color='coral', alpha=0.4, transform=ccrs.PlateCarree())  
    return ax

# ------------------
def draw_box(ax,proj,box_west,box_east,box_top,box_bot):
    # make lines
    xtop,ytop = np.linspace(box_east,box_west),np.linspace(box_top,box_top)
    xbot,ybot = np.linspace(box_east,box_west),np.linspace(box_bot,box_bot)
    xleft,yleft = np.linspace(box_west,box_west),np.linspace(box_bot,box_top)
    xright,yright = np.linspace(box_east,box_east),np.linspace(box_bot,box_top)        
    # plot lines
    ax.plot(xtop,ytop , 'r-', transform=proj)
    ax.plot(xbot,ybot , 'r-', transform=proj) 
    ax.plot(xright,yright , 'r-', transform=proj) 
    ax.plot(xleft,yleft , 'r-', transform=proj)
    return ax

def cmap_topo():
    topo_cm = cm.get_cmap(cmo.topo, 512)
    newcmp = ListedColormap(topo_cm(np.linspace(0.2, 1, 256)))
    return newcmp

# ------------------
def topograpgy(fig,ax,lims):
    # open topo file
    topo = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/Mestrado/ROAD/Weather_types/Data/ETOPO1_Bed_c_gmt4.grd')
    # get map limits
    lon0, lon1 = lims[0],lims[1]
    lat0, lat1 = lims[2],lims[3]
    # slice data for saving time
    topo = topo.sel(y=slice(lat0,lat1), x=slice(lon0, lon1))
    min_ = round(int(np.amin(topo.z).values),-2)
    max_ = round(int(np.amax(topo.z).values),-2)
    norm = MidpointNormalize(vmin=min_, vcenter=0, vmax=max_)
    cf = ax.pcolormesh(topo.x, topo.y, topo.z, cmap=cmo.topo,
                norm=norm, shading='nearest')
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1-0.08, pos.y0+0.08, 0.02, pos.height/3])
    cb = plt.colorbar(cf,cax=cbar_ax, orientation='vertical',
                            ticks = [min_, 0, max_])
    if lat1 < 0:
        c = 'wheat'
    else:
        c = 'k'
    cb.outline.set_edgecolor(c)
    cb.outline.set_linewidth(1)
    cbar_ax.tick_params(axis='both', colors=c)
    return ax

# ------------------
def surf():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    topo = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/Mestrado/ROAD/Weather_types/Data/ETOPO1_Bed_c_gmt4.grd')
    topo = topo.sel(y=slice(-34,-26), x=slice(-54, -45))
    X, Y = np.meshgrid(topo.x, topo.y)
    ax.plot_surface(X, Y, topo.z, cmap=cmo.topo, linewidth=0.8, rstride=1)
    ax.view_init(70, 270)

# from Matplolib documentation:
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
# ------------------
def main():
    '''
    Make the final figure 
    '''
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9.5,5) , constrained_layout=False)
    gs = gridspec.GridSpec(1, 2, hspace=0, wspace=0,
                           left=0.05, right=0.99,
                           width_ratios=[.7, 1])
    # South America map
    ax1 = (fig.add_subplot(gs[0], projection=proj))
    lims = [-82, -25, -60, 15]
    ax1.set_extent([-82, -25, -57, 13]) 
    map_features(ax1)
    topograpgy(fig,ax1,lims)
    for state in ['PR','SC','RS']:
        ax1 = highlight_state(ax1,state)
    draw_box(ax1,proj,-54, -45, -34, -26)
    # Southern Brazil map
    ax2 = (fig.add_subplot(gs[1], projection=proj))
    lims = [-54, -45, -34, -26]
    ax2.set_extent(lims)
    topograpgy(fig,ax2,lims)
    # cosmedics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs = [ax1,ax2]
    for ax, label in zip(axs,['A','B']):
        grid_labels_params(ax)
        Brazil_states(ax)
        ax.text(0.85,0.85, label, fontsize = 18, transform=ax.transAxes, bbox=props)
        
    pl.savefig('../Figures/map.png', format='png')
    pl.savefig('../Figures/map.tiff', format='tiff', dpi=600)

# ------------------
if __name__ == "__main__": 
    main()
    
