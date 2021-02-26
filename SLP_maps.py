#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:18:11 2021

@author: danilocoutodsouza
"""

import cfgrib
import csv

import numpy as np 
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

import maps

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
# Assign dates so its needed to acess it only once
dates = get_WT_dates()

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
def make_map(WT):
    
    proj = ccrs.Orthographic(-60,-20)
    fig = plt.figure(figsize=(9.5,5) , constrained_layout=False)
    ax = (fig.add_subplot(projection=proj))
    
    plot_SLP(ax,WT,proj)
    
    maps.draw_box(ax,ccrs.PlateCarree(),-54, -45, -34, -26)
    
    maps.grid_labels_params(ax)
    maps.map_features(ax)
    maps.Brazil_states(ax)
    
 
# ---------------    
def plot_SLP(ax,WT,proj):
    
    # get data
    slp = get_CFSR_data(WT).prmsl/100
    lat = slp.latitude
    lon = slp.longitude
    # set limits for plotting
    min_ = round(int(np.amin(slp).values))
    max_ = round(int(np.amax(slp).values))
    norm = maps.MidpointNormalize(vmin=min_, vcenter=1014, vmax=max_)
    cf = ax.pcolormesh(lon, lat, slp[0], cmap=cmo.balance,
                norm=norm, shading='nearest', transform=ccrs.PlateCarree())
    
if __name__ == "__main__": 
    make_map(1)
    
    