#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:18:11 2021

Make:
    1) Maps of CFSR SLP/wind fields for WT analysis
    2) Maps of OLAM SLP/wind fields for high resolution WT analysis
    3) Gifs of both CSFR and OLAM data suitable for supplementary material

@author: danilocoutodsouza
"""
import cfgrib
import csv

import numpy as np 
import pylab as pl
import cmocean.cm as cmo
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from celluloid import Camera

import maps

def convert_lon(df):
    
    """
    Convert longitudes from 0:360 range to -180:180
    """
    
    df.coords['longitude'] = (df.coords['longitude'] + 180) % 360 - 180
    df = df.sortby(df.longitude)
    
    return df

# ---------------
def get_WT_dates():  
    '''
    Creates an array containing dates (stirngs) from all Weather Types
    '''      
    file = '/Users/danilocoutodsouza/Documents/UFSC/Mestrado/ROAD/Weather_types/Data/Datas_selecionadas_36WT.csv'    
    years = []
    months = []
    days = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            years.append(f'{row[0]}') 
            months.append(f'{row[1]}')
            days.append(f'{row[2]}')
    dates = []
    for year,month,day in zip(years,months,days):
        dates.append(year+'-'+month+'-'+day)
    return dates

# ---------------
def get_CFSR_data(WT):
    # get day from WT
    date = dates[WT-1]
    year = date[:4]
    month = date[5:7]
    # open files
    slp_data = cfgrib.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/CFSR/'+
                                   'prmsl.l.gdas.'+
                                   '197901-201012.grb2/prmsl.l.gdas.'+
                                   year+month+'.grb2',
                                   engine='cfgrib')
    wnd_data = cfgrib.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/CFSR/'+
                                   '/wnd1000.l.gdas.'+
                                   '197901-201012.grb2//wnd1000.l.gdas.'+
                                   year+month+'.grb2',
                                   engine='cfgrib')
    # slice to get only the choosen day
    slp_data = slp_data.sel(time=date)
    wnd_data = wnd_data.sel(time=date)
    # merge data
    data = slp_data.assign(wnd_data)
    return data

# ---------------
def get_OLAM_data(WT):
    # get day from WT
    date = dates[WT-1]
    # open files
    if WT < 10:
        file = '0'+str(WT)
    else:
        file = str(WT)
    slp_data = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                               'Mestrado/ROAD/Weather_types/Data/'+
                               'OLAM_netcdf_36WT/alltimes/'+
                               'OLAM_WT'+file+'_full_slp.nc')
    uwnd_data = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                               'Mestrado/ROAD/Weather_types/Data/'+
                               'OLAM_netcdf_36WT/alltimes/'+
                               'OLAM_WT'+file+'_full_uwnd.nc')
    vwnd_data = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                               'Mestrado/ROAD/Weather_types/Data/'+
                               'OLAM_netcdf_36WT/alltimes/'+
                               'OLAM_WT'+file+'_full_vwnd.nc')
    prec_data = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                               'Mestrado/ROAD/Weather_types/Data/'+
                               'OLAM_netcdf_36WT/snapshot/'+
                               'OLAM_WT'+file+'_accprecip.nc')
    # merge data
    data = slp_data.assign(uwnd_data).assign(vwnd_data).assign(prec_data)
    # slice to get only the choosen day
    data = data.sel(time=date)
    return data
 
# ---------------    
def plot_SLP(ax,WT,proj,time,source):
    # get data
    if source == 'cfsr':
        data = get_CFSR_data(WT)
        data = convert_lon(data)
        data =  data.sel(longitude=slice(lims_cfsr[0]-2,lims_cfsr[1]), 
                         latitude=slice(lims_cfsr[3],lims_cfsr[2]-15))
        slp = data.prmsl/100
        u = data.u
        v = data.v
        lat = slp.latitude
        lon = slp.longitude
        skip = 2
    elif source == 'olam':
        data = get_OLAM_data(WT)
        data =  data.sel(lon=slice(lims_olam[0],lims_olam[1]), 
                         lat=slice(lims_olam[2],lims_olam[3]))   
        slp = data.sslp/100
        u = data.uwnd
        v = data.vwnd
        lat = slp.lat
        lon = slp.lon
        skip = 20
    
    # set limits for plotting
    min_ = round(int(np.amin(slp).values))
    max_ = round(int(np.amax(slp).values))
    norm = maps.MidpointNormalize(vmin=min_, vcenter=1014, vmax=max_)
    ax.pcolormesh(lon, lat, slp[time], cmap=cmo.balance,
                norm=norm, shading='nearest', transform=proj)
    ax.quiver(lon[::skip], lat[::skip],
              u[time][::skip,::skip], v[time][::skip,::skip],
              transform=proj)    
    
 
# ------------------
def draw_box(ax,proj,box_west,box_east,box_top,box_bot,line, lw):
    # make lines
    xtop,ytop = np.linspace(box_east,box_west),np.linspace(box_top,box_top)
    xbot,ybot = np.linspace(box_east,box_west),np.linspace(box_bot,box_bot)
    xleft,yleft = np.linspace(box_west,box_west),np.linspace(box_bot,box_top)
    xright,yright = np.linspace(box_east,box_east),np.linspace(box_bot,box_top)  
    # plot lines
    ax.plot(xtop,ytop , line, linewidth=lw, transform=proj)
    ax.plot(xbot,ybot , line, linewidth=lw, transform=proj) 
    ax.plot(xright,yright , line, linewidth=lw, transform=proj) 
    ax.plot(xleft,yleft , line, linewidth=lw, transform=proj)
    return ax


# ------------------   
def grid_labels_params(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5,linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 14, 'color': 'gray'}
    gl.ylabel_style = {'size': 14, 'color': 'gray'}
    ax.outline_patch.set_edgecolor('gray')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax
    
# ---------------
def make_map_CFSR(WT,lims):
    # Make figure
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8,6.6) , constrained_layout=False)      
    gs = gridspec.GridSpec(2, 2, hspace=0.1, wspace=0,
                           left=0, right=0.95)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Loop for 00, 06, 12 and 18
    for i in range(4):
        ax = (fig.add_subplot(gs[i], projection=proj))
        ax.set_extent(lims) 
        # Plot SLP and wind
        plot_SLP(ax,WT,proj,i,'cfsr')
        # Draw boxes for analysis
        l = 1
        draw_box(ax,proj,-70, 0, -62, -25, 'k-',3)
        draw_box(ax,proj,-69, -34, -43, -26, 'k--',l)
        draw_box(ax,proj,-69, -34, -61, -44, 'k--',l)
        draw_box(ax,proj,-33, -1, -43, -26, 'k--',l)
        draw_box(ax,proj,-33, -1,  -61, -44, 'k--',l)
        # Cosmedics
        grid_labels_params(ax)
        maps.map_features(ax)
        maps.Brazil_states(ax)
        ax.text(0.05,0.8,str(i), transform=ax.transAxes, fontsize=16,bbox=props)
        
    pl.savefig('../Figures/cfsr_maps/'+str(WT)+'.png', format='png')
    
# ---------------
def make_map_OLAM(WT,lims):
    # Make figure
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8,14) , constrained_layout=True)        
    gs = gridspec.GridSpec(4, 2, hspace=0.1, wspace=0,
                           left=0, right=0.9)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Loop for 00, 06, 12 and 18
    for i in range(8):
        ax = (fig.add_subplot(gs[i], projection=proj))
        ax.set_extent(lims) 
        # Plot SLP and wind
        plot_SLP(ax,WT,proj,i,'olam')
        # Cosmedics
        grid_labels_params(ax)
        maps.map_features(ax)
        maps.Brazil_states(ax)
        ax.text(0.05,0.8,str(i), transform=ax.transAxes, fontsize=16,bbox=props)
        
    pl.savefig('../Figures/olam_maps/'+str(WT)+'.png', format='png')    
       
    
def make_gif(source):
    # Make figure
    proj = ccrs.PlateCarree()
    if source == 'cfsr':
        fig = plt.figure(figsize=(20,15) , constrained_layout=True)
    elif source == 'olam':
        fig = plt.figure(figsize=(18,15) , constrained_layout=True)
    gs = gridspec.GridSpec(6, 6)
    camera = Camera(fig)
    axs = []
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if source == 'cfsr':
        lims = lims_cfsr
    elif source == 'olam':
        lims = lims_olam
    # Loop for 00, 06, 12 and 18
    if source == 'cfsr':
        times = 4
    elif source == 'olam':
        times = 8
    for t in range(times):
        for WT in range(1,37):
            axs.append(fig.add_subplot(gs[WT-1], projection=proj))
            ax = axs[-1]
            ax.set_extent(lims) 
            # # Plot SLP and wind
            plot_SLP(ax,WT,proj,t,source)
            # Draw boxes for analysis
            if source == 'cfsr':
                draw_box(ax,proj,-70, 0, -62, -25, 'k-',2)
                draw_box(ax,proj,-69, -34, -43, -26, 'k--',1)
                draw_box(ax,proj,-69, -34, -61, -44, 'k--',1)
                draw_box(ax,proj,-33, -1, -43, -26, 'k--',1)
                draw_box(ax,proj,-33, -1,  -61, -44, 'k--',1)
            # Cosmedics
            maps.map_features(ax)
            ax.text(0.05,0.8,str(WT), transform=ax.transAxes, fontsize=16,bbox=props)
            # snap animation
        camera.snap()
    animation = camera.animate(interval = 200, repeat = True,
                            repeat_delay = 500)
    animation.save('../Figures/'+source+'_anim.gif')
    
    

# Assign dates so its needed to acess it only once
dates = get_WT_dates()
# Figure limits for plotting and slice data
lims_cfsr = [-82, 10, -57, 0]
lims_olam = [-54, -44.05, -34, -25.05]

# ---------------    
def main(): 
    # Make gif for supplementary material
    make_gif('cfsr')
    make_gif('olam')
    # Make figures for better visualization
    for wt in range(1,37):
        make_map_CFSR(wt,lims_cfsr)
        make_map_OLAM(wt,lims_olam)              
    
# ---------------
if __name__ == "__main__": 
    main()    
    