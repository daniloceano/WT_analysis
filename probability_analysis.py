#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:50:18 2021

Script for computating the annual and seasonal probabilities of the WTs.

It will also calculate the occurence prob. of the WTs in respect to:
    1) Southern Annular Mode indeces
    2) El-Niño Southern Oscilation indeces
    3) Madden Jullian Oscilation phase

@author: danilocoutodsouza
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import (MultipleLocator, FixedLocator)



def get_data():
    ''' 
    Use Pandas to get data from the csv files.
    
    The first file contains all dates from 1979 to 2010
        with all corresponding Wts. The other files contain data
        from the climate modes: Southern Oscillation Index (SOI),
        Marshall Southern Annular Mode (SAM) index (ttation-based) and
        Real-time Multivariate Madden-Julian Oscillation (MJO) series (RMM).
        
    Sources:
        SOI: https://www.ncdc.noaa.gov/teleconnections/enso/indicators/soi/
        SAM: https://climatedataguide.ucar.edu/climate-data/marshall-southern-annular-mode-sam-index-station-based
        RMM: http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
    '''
    
    # WT data
    file = '../all_WT_dates2.txt'
    data = pd.read_csv(file) 
    # ENSO data
    data_soi=  pd.read_csv('../SOI.txt')
    # SAM data
    data_sam=  pd.read_csv('../SAM.txt') 
    # MJO data
    data_mjo = pd.read_csv('../MJO.txt') 

    # Temporal range of data
    years = np.arange(1979,2011)
    mons = np.arange(1,13)
    # month indeces for SAM dataframe indexing
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL",
          "AUG","SEP","OCT","NOV","DEC"]
    # Update the 'data' dataframe to include all climate indeces
    for y in years:
        for m in mons:
            # --- SAM ---
            # Get SAM index for respective year and month
            tmp_sam = data_sam[data_sam['YEAR'] == y][months[m-1]].values[0]
            # Attribute the SAM index to the  original dataframe
            data.loc[(data['year'] == y) & \
                     (data['month'] == m), ['SAM']] = tmp_sam
            # --- SOI ---
            # Get SOI index for respective year and month
            tmp_soi = data_soi[(data_soi['Year'] == y) & \
                           (data_soi['Month'] == m)]['Value'].values[0]
            # Attribute the SOI index to the  original dataframe
            data.loc[(data['year'] == y) & \
                      (data['month'] == m), ['SOI']] = tmp_soi
            # --- Loop through days ----
            days = data[(data['year']==y) & \
                    (data['month']==m)]['day']
            # --- MJO ---
            # Get MJO phase for respective day    
            for d in days:
                tmp_mjo = data_mjo[(data_mjo['year'] == y) & \
                           (data_mjo['month'] == m) & \
                           (data_mjo['day'] == d)]['phase'].values[0]
                data.loc[(data['year'] == y) & \
                      (data['month'] == m) & \
                      (data['day'] == d), ['MJO']] = tmp_mjo
                    
    return data

#------------------------
# ANNUAL PROBABILITY    
#------------------------

def counter_total(data,wt):
    '''
    Counter for a given WT in the whole time series
    '''    
    return len(data[data['WT'] == wt])

def counter_per_year(data,wt,year):
    '''
    Counter for a given WT in a given year
    '''
    return len(data[(data['year'] == year) & (data['WT'] == wt)])


def calc_probability_per_year(data,wt,year):
    '''
    Calculates the occurence probability of a given WT in a given year
    '''
    ct_wt = counter_per_year(data,wt,year)
    ct_tot = len(data[(data['year'] == year)])
    return (ct_wt/ct_tot)*100

def make_annual_ts(data,wt):
    '''
    Make a time series for annual occurence probability of given wt
    '''
    years = np.arange(1979,2011)
    wt_ts = []
    for year in years:
        wt_ts.append(calc_probability_per_year(data,wt,year))
    return wt_ts

#------------------------
# INTERANNUAL PROBABILITY    
#------------------------
    
def calc_interannual_prob(data,wt):
    '''
    Calculates the occurence probability of a given WT across all years in data
    '''
    # counter for all days in data
    ct_tot = len(data)
    # counter for WT occurence within data
    wt_ct_all_years = counter_total(data,wt)
    return (wt_ct_all_years/ct_tot)*100

#------------------------
# SEASONAL PROBABILITY    
#------------------------

def calc_seasonal_prob(data,wt, season):
    '''
    Calculates the occurence probability of a given WT in DJF months
    '''
    # months and seasons
    DJF = [12,1,2]
    MAM = [3,4,5]
    JJA = [6,7,8]
    SON = [9,10,11]
    seasons_dict = {'DJF':DJF,'MAM':MAM,'JJA':JJA,'SON':SON} 
    # select season for indexing
    s = seasons_dict[season]
    # counters
    ct_wt = len(data[(data['month'].isin(s)) & (data['WT'] == wt)])
    ct_tot = len(data[(data['month'].isin(s))])
    return (ct_wt/ct_tot)*100

#------------------------
# ENSO 
#------------------------

def counter_ENSO(data,wt,phase):
    '''
    Count how many times a given WT occur in Nino or Nina "events"
    '''
    if phase == 'Nino':
        ct_wt = len(data[(data['SOI'] < -1) & (data['WT'] == wt)])
    elif phase == 'Nina':
        ct_wt = len(data[(data['SOI'] > 1) & (data['WT'] == wt)])
    return ct_wt

def calc_probability_ENSO(data,wt,phase):
    '''
    Calculates the occurence probability of a given WT for a given ENSO phase
    '''
    # counter for all days in data matching the given ENSO phase
    if phase == 'Nino':
        ct_tot = len(data[(data['SOI'] < -1)])
    elif phase == 'Nina':
        ct_tot = len(data[(data['SOI'] > 1)])
    # counter for WT occurence in data matching the given ENSO phase    
    ct_wt = counter_ENSO(data,wt,phase)
    return (ct_wt/ct_tot)*100

#------------------------
# SAM 
#------------------------

def counter_SAM(data,wt,phase):
    '''
    Count how many times a given WT occur in Positive/Negative SAM "events"
    '''
    if phase == 'Neg':
        ct_wt = len(data[(data['SAM'] < -2) & (data['WT'] == wt)])
    elif phase == 'Pos':
        ct_wt = len(data[(data['SAM'] > 2) & (data['WT'] == wt)])
    return ct_wt

def calc_probability_SAM(data,wt,phase):
    '''
    Calculates the occurence probability of a given WT for a given SAM phase
    '''
    # counter for all days in data matching the given SAM phase
    if phase == 'Neg':
        ct_tot = len(data[(data['SAM'] < -2)])
    elif phase == 'Pos':
        ct_tot = len(data[(data['SAM'] > 2)])
    # counter for WT occurence in data matching the given SAM phase    
    ct_wt = counter_SAM(data,wt,phase)
    return (ct_wt/ct_tot)*100

#------------------------
# MJO 
#------------------------

def counter_MJO(data,wt,phase):
    '''
    Count how many times a given WT occur in a given MJO phase
    '''
    ct_wt = len(data[(data['MJO'] == phase) & (data['WT'] == wt)])
    return ct_wt

def calc_probability_MJO(data,wt,phase):
    '''
    Calculates the occurence probability of a given WT for a given MJO phase
    '''
    # counter for all days in data matching the given MJO phase
    ct_tot = len(data[(data['MJO'] == phase)])
    # counter for WT occurence in data matching the given ENSO phase    
    ct_wt = counter_MJO(data,wt,phase)
    return (ct_wt/ct_tot)*100


#------------------------
# PLOTS 
#------------------------

col1 = ['#f4f0f0','#e7d5be','#afcfdf','#497fc9','#485fb0']
col2 = ['#ffcdb2','#ffb4a2','#e5989b','#b5838d','#6d6875']
col3 = ['#6a040f','#9d0208','#d00000','#dc2f02','#e85d04', '#f48c06','#faa307']
col4 = ['#6a040f','#f3722c','#f8961e','#f9844a','#f9c74f',
        '#90be6d','#43aa8b','#4d908e','#577590','#277da1']
col5 = ['#1a535c','#4ecdc4','#f7fff7','#ff6b6b','#ffe66d']
col6 = ['#d9ed92','#b5e48c','#99d98c','#76c893','#52b69a',
        '#34a0a4','#168aad','#1a759f','#1e6091','#184e77']
col7 = ['#f0ead2','#dde5b6','#adc178','#a98467','#6c584c']


cmap = LinearSegmentedColormap.from_list(
        'MyMap', col1, N=10)


def annotate_wts(ax):
    # Loop over data dimensions and create text annotations.
    ct = 1
    for j in range(0,6):
        for i in range(0,6):
            ax.text(j, i, ct,
                    ha="center", va="center", 
                    color="k", fontsize = 12)
            ct += 1
        
def plot_annual_prob(data,ax,fig):
    probs = []
    for wt in range(1,37):
        probs.append(calc_interannual_prob(data,wt))
    probs = np.reshape(probs,(2,18),order='F')
    cf1 = ax.imshow(probs, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # colorbar
    min_, max_ = np.amin(probs), np.amax(probs)
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0-0.025, pos.width, pos.height/4])
    cbar = plt.colorbar(cf1, ticks=[min_, max_],
                        cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=14)
    # annotate WTs
    ct = 1
    for j in range(0,18):
        for i in range(0,2):
            ax.text(j, i, ct,
                    ha="center", va="center", 
                    color="k", fontsize = 12)
            ct += 1
    
def plot_season_prob(data,season,ax):
    probs = []
    for wt in range(1,37):
        probs.append(calc_seasonal_prob(data,wt, season))
    probs = np.reshape(probs,(6,6),order='F')
    ax.imshow(probs, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    annotate_wts(ax)
    
def plot_mjo_prob(data,phase,ax):
    probs = []
    for wt in range(1,37):
        probs.append(calc_probability_MJO(data,wt, phase))
    probs = np.reshape(probs,(6,6),order='F')
    ax.imshow(probs, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    annotate_wts(ax)
    
def plot_soi_prob(data,phase,ax):
    probs = []
    for wt in range(1,37):
        probs.append(calc_probability_ENSO(data,wt, phase))
    probs = np.reshape(probs,(6,6),order='F')
    ax.imshow(probs, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    annotate_wts(ax)
    
def plot_sam_prob(data,phase,ax):
    probs = []
    for wt in range(1,37):
        probs.append(calc_probability_SAM(data,wt, phase))
    probs = np.reshape(probs,(6,6),order='F')
    ax.imshow(probs, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    annotate_wts(ax)   
    
# Plot time series of probability
#   1) 5-year smoothed 
#   2) All probs

def smooth_ts(ts):
    means = []
    for i in range(0,len(ts)+1,5):
        means.append(np.mean(ts[i:i+5]))
    return means
    
def plot_ts(data,wt,ax):
    years = np.arange(1979,2011)
    ts = make_annual_ts(data,wt)
    ax.plot(years,ts, linewidth=2,linestyle=(0, (5, 1)))
    smoothed = smooth_ts(ts)
    ax.plot(years[::5],smoothed, linewidth=2, color='k',alpha=0.9)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.xaxis.set_major_locator(FixedLocator(years[1::10]))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    plt.xticks(rotation=90)


# ---------------

def main(data):
    # data = get_data()
    fig = plt.figure(figsize=(19.5,12) , constrained_layout=False)
    gs0 = gridspec.GridSpec(1, 2, wspace=0.07,hspace=0)
    gs00 = gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=gs0[0],
                                            wspace=0,hspace=0.2)
    gs01 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs0[1],
                                            wspace=0.05,hspace=0.05)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # === Plot probabilities ===
    axs00 = []
    
    # Annual
    axs00.append(plt.subplot(gs00[0, :4]))
    ax = axs00[-1]
    plot_annual_prob(data,ax,fig)
    ax.text(0.45,1.07, 'Annual', fontsize = 16, transform=ax.transAxes)
    
    # Seasons
    for col,season in zip(range(4),['DJF','MAM','JJA','SON']):
        axs00.append(plt.subplot(gs00[1, col]))
        ax = axs00[-1]
        plot_season_prob(data,season,ax)
        ax.text(0.35,1.05, season, fontsize = 16, transform=ax.transAxes)
        
    # MJO
    phase = 1
    for row in range(2,4):
        for col in range(4):
            axs00.append(plt.subplot(gs00[row, col]))
            ax = axs00[-1]
            plot_mjo_prob(data,phase,ax)
            ax.text(0.3,1.05, 'MJO '+str(phase), fontsize = 16, transform=ax.transAxes)
            phase += 1
            
    # SOI
    titles = ['SOI > 1', 'SOI < -1']
    for col, phase, title in zip(range(2),['Nina','Nino'], titles):
        axs00.append(plt.subplot(gs00[4, col]))
        ax = axs00[-1]
        plot_soi_prob(data,phase,ax)
        ax.text(0.2,1.05, title, fontsize = 16, transform=ax.transAxes)
        
        
    # SAM
    titles = ['SAM > 2', 'SAM < -2']
    for col, phase, title in zip(range(2,4),['Pos','Neg'], titles):
        axs00.append(plt.subplot(gs00[4, col]))
        ax = axs00[-1]
        plot_sam_prob(data,phase,ax)
        ax.text(0.2,1.05, title, fontsize = 16, transform=ax.transAxes)
        
    # === Time sreies ===
    axs01= []
    for wt in range(1,37):
        axs01.append(plt.subplot(gs01[wt-1]))
        ax = axs01[-1]
        plot_ts(data,wt,ax)
        ax.set_ylim(0,10)
        ax.text(0.1,0.8, str(wt), fontsize = 16, transform=ax.transAxes, bbox=props)
        plt.grid(linewidth=0.5,color='gray')
        if wt not in ([1,7,13,19,25,31]):
            ax.set_yticks(range(0,9,2))
            ax.set_yticklabels([])
        else:
            ax.set_yticks([0,2,4,6,8])
        if wt < 31:
            ax.set_xticklabels([])
                
    pl.savefig('../Figures/probabilities/panel.png', format='png')
    pl.savefig('../Figures/probabilities/panel.tiff', format='tiff', dpi=600)
        
    
# ------------------
if __name__ == "__main__": 
    main()
    