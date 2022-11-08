#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:22:18 2021

Export OLAM, INMET and CFSR data for model validation. 
Creates a .csv file using Pandas' DataFrame for each station and WT.
  

@author: danilocoutodsouza
"""

import SLP_maps as smaps
import RAINpanel_validation as rainp
import glob
import cfgrib
import xarray as xr
import numpy as np
import pandas as pd
from metpy.calc import wind_speed
from metpy.units import units

def get_OLAM_data(WT,var):
    if WT < 10:
        file = '0'+str(WT)
    else:
        file = str(WT)
    if var == 'SLP':
        data = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/'+
                                   'OLAM_netcdf_36WT/alltimes/'+
                                   'OLAM_WT'+file+'_full_slp.nc')
    elif var == 'WIND':
        data1 = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/'+
                                   'OLAM_netcdf_36WT/alltimes/'+
                                   'OLAM_WT'+file+'_full_uwnd.nc')
        data2 = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/'+
                                   'OLAM_netcdf_36WT/alltimes/'+
                                   'OLAM_WT'+file+'_full_vwnd.nc')
        data = data1.assign(data2)
    elif var == 'RAIN':
        data = xr.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/'+
                                   'OLAM_netcdf_36WT/snapshot/'+
                                   'OLAM_WT'+file+'_accprecip.nc')
        
    return data

def get_CFSR_data(dstart, dend,var):
    '''
        Open CFSR data for disired range.
    As it is a monthly data, it is needed to open two sets of data for distinct
    months and then merge them. Afterwards, the data is sliced for only the
    desired range and for the OLAM domain.
    '''
    if var == 'SLP' or var == 'RAIN':
        varname = 'prmsl'
    elif var == 'WIND':
        varname = 'wnd1000'
    # Open data 
    # If WT starts and ends in the same year
    if dstart.year == dend.year:
        year = str(dstart.year)
        # If WT starts and ends in the same year and month
        if dstart.month == dend.month:
            if dstart.month < 10:
                month = '0'+str(dstart.month)
            else:
                month = str(dstart.month)
            var_data = cfgrib.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/CFSR/'+
                                   varname+'.l.gdas.'+
                                   '197901-201012.grb2/'+varname+'.l.gdas.'+
                                   year+month+'.grb2',
                                   engine='cfgrib')
        # If WT starts and ends in the same year but different months
        else:
            month1, month2 = dstart.month, dend.month
            data =  []
            for month in [month1,month2]:
                if month < 10:
                    month = '0'+str(month)
                else:
                    month = str(month)
                data.append(cfgrib.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                                   'Mestrado/ROAD/Weather_types/Data/CFSR/'+
                                   varname+'.l.gdas.'+
                                   '197901-201012.grb2/'+varname+'.l.gdas.'+
                                   year+month+'.grb2',
                                   engine='cfgrib'))
            var_data = xr.merge([data[0], data[1]])
    # If WT starts and ends different years and months
    else:
        year1, year2 = str(dstart.year), str(dend.year)
        month1, month2 = dstart.month, dend.month
        data =  []
        for month, year in zip([month1,month2],[year1,year2]):
            if month < 10:
                month = '0'+str(month)
            else:
                month = str(month)
            data.append(cfgrib.open_dataset('/Users/danilocoutodsouza/Documents/UFSC/'+
                               'Mestrado/ROAD/Weather_types/Data/CFSR/'+
                               varname+'.l.gdas.'+
                               '197901-201012.grb2/'+varname+'.l.gdas.'+
                               year+month+'.grb2',
                               engine='cfgrib'))
        var_data = xr.merge([data[0], data[1]])
    var_data = smaps.convert_lon(var_data)   
    var_data = var_data.sel(time=slice(dstart,dend),
                     longitude=slice(-55,-43),
                     latitude=slice(-24,-35))
    return var_data

def get_ERA5_data(dstart, dend,var):
    var = 'SLP'
    # Path to ERA5 data
    path = '/Users/danilocoutodsouza/Documents/UFSC/Mestrado/ROAD/Weather_types/Data/ERA5/'
    suffix = '.grb.spasub.desouza531500'
    ystart, yend = dstart.year,  dend.year
    # For SLP data
    if var == 'SLP':
        prefix = 'e5.oper.an.sfc.128_151_msl.ll025sc.'
        fdir = prefix+'1979010100_1979013123-2008120100_2008123123'+suffix
        if dstart.month == dend.month:
            y,m = str(dstart.year),str(dstart.month)
            if int(m)<10 : m = '0'+m
            file = glob.glob(path+fdir+'/'+prefix+y+m+'0100*[!idx]')
            data = smaps.convert_lon(cfgrib.open_dataset(file[0],
                         engine='cfgrib')).sel(
                         longitude=slice(-55,-43),
                         latitude=slice(-24,-35))
            ## Slice data using timesteps as ERA5 has this stupid indexing
            timeref = pd.Timestamp(data.time.values).to_pydatetime()
            stepstart,stepend = dstart-timeref, dend-timeref
            data = data.sel(step=slice(stepstart,stepend))
        else:
            prefix = 'e5.oper.an.sfc.128_151_msl.ll025sc.'
            ## Open first dataset
            y,m = str(ystart),str(dstart.month)
            if int(m)<10 : m = '0'+m
            file = glob.glob(path+fdir+'/'+prefix+y+m+'0100*[!idx]')
            data1 = smaps.convert_lon(
                    cfgrib.open_dataset(file[0],
                    engine='cfgrib')).sel(
                    longitude=slice(-55,-43),
                    latitude=slice(-24,-35))
            timeref = pd.Timestamp(data1.time.values).to_pydatetime()
            stepstart = dstart-timeref
            data1 = data1.sel(step=slice(stepstart,data1.step[-1]))
            ## Open second dataset
            y,m = str(yend),str(dend.month)
            if int(m)<10 : m = '0'+m
            file = glob.glob(path+fdir+'/'+prefix+y+m+'0100*[!idx]')
            data2 = smaps.convert_lon(
                    cfgrib.open_dataset(file[0],
                    engine='cfgrib')).sel(
                    longitude=slice(-55,-43),
                    latitude=slice(-24,-35))
            timeref = pd.Timestamp(data2.time.values).to_pydatetime()
            stepend = dend-timeref
            data2 = data2.sel(step=slice(data2.step[0],stepend))
            # Concatenate both files
            data = xr.concat([data1,data2],dim='step')
    return data

def GetWTStartEnd():
    files = glob.glob('../*') 
    fname = '../WT_StartEnd.csv'
    if fname in files:
        print('\n**** FOUND FILE WITH WT DATES ****')
        pass
    else:
        print('\n*** FILE WITH WT DATES NOT FOUND ***')
        for WT in range(1,37):
            odata = get_OLAM_data(WT,'SLP')
            dstart = pd.Timestamp(odata.time[0].time.values).to_pydatetime()
            dend =  pd.Timestamp(odata.time[-1].time.values).to_pydatetime()
            if WT == 1:
                df = pd.DataFrame(data=[[dstart,dend]],columns=['Start','End'],index=[WT])
            else:
                tmp = pd.DataFrame(data=[[dstart,dend]],columns=['Start','End'],index=[WT])
                df = df.append(tmp)
        df.to_csv(fname,index_label='WT')
            

def OpenInmetData(station):
    '''
    Opens INMET data for a desired station and set index as a datetime type.
        Also opens the header with station info.
    '''
    parse_dates = ['Data Medicao', 'Hora Medicao'] # Specify which columns are dates
    df_inmet = pd.read_csv(station,delimiter = ';',skiprows=10,decimal=",",
                     index_col=None,keep_date_col=True,parse_dates=parse_dates)
    ## Transform the indeces into datetime arrays
    df_inmet['DATA (YYYY-MM-DD) HORA (UTC)'] = pd.to_datetime(df_inmet['Data Medicao'].astype(str)+ ' ' + df_inmet['Hora Medicao'].astype(str),format='%Y-%m-%d %H%M')
    df_inmet = df_inmet.set_index(pd.to_datetime(df_inmet['DATA (YYYY-MM-DD) HORA (UTC)']))
    # Open header info
    header = pd.read_csv(station,delimiter = ': ',index_col=0,header=None,
                         nrows=9,decimal=",",engine='python').transpose()
    return df_inmet, header

def ProcessINMET(df_inmet,dstart,dend,var):
    '''
    Process INMET SLP data.
        Opens DataFrame input and then:
        1) Slice for desired range
        2) Filter spuirous data
    '''
    # Slice inmet df only for desired range of dates
    df_inmet = df_inmet.loc[str(dstart):str(dend)]
    # Assign variable name as in INMET file to as variable
    if var == 'SLP':
        varname = 'PRESSAO ATMOSFERICA AO NIVEL DO MAR, HORARIA(mB)'
    elif var == 'WIND':
        varname = 'VENTO, VELOCIDADE HORARIA(m/s)'
    elif var == 'RAIN':
        varname = 'PRECIPITACAO TOTAL, HORARIO(mm)'
    # Convert data to float
    df_inmet = df_inmet.astype({varname: float})
    # Mask "strange" values
    if var == 'SLP':
        mask = (df_inmet[varname]  < 1050) & (df_inmet[varname] > 970)
        df_inmet_var = df_inmet[varname].loc[mask] 
    elif var == 'WIND':
        mask = (df_inmet[varname]  > 0) & (df_inmet[varname] < 100)
        df_inmet_var = df_inmet[varname].loc[mask] 
    elif var == 'RAIN':
        df_inmet_var = df_inmet[varname].cumsum()
        mask = (df_inmet[varname]  >= 0)
    # Apply mask
    return df_inmet_var

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

tmpd = {}

    
def create_database(var):
    files = glob.glob('../conventional_stations/dados*') # Get a list with station files
    for station in files[:]:
        ## Now, open INMET data
        tmp = OpenInmetData(station)
        df_inmet, header = tmp[0], tmp[1]
        station_name = header['Nome'][1]
        print('\n-------------------------------------------------')
        print('Processing data for Station: '+(station_name))
        print('-------------------------------------------------')
        print('Inmet data range: '+str(df_inmet.iloc[0]['Data Medicao'])+' to '+str(df_inmet.iloc[-1]['Data Medicao'])+'\n')
        for WT in range(1,37):
            print('WT = '+str(WT))
            ###     Firstly, create a DataFrame using the dates from the OLAM file
            ### Data will not be assigned yet because firstly it's needed to get station info
            dates = pd.read_csv('../WT_StartEnd.csv',index_col=0)
            dstart = pd.Timestamp(dates.loc[WT]['Start']).to_pydatetime()
            dend = pd.Timestamp(dates.loc[WT]['End']).to_pydatetime()
            if var in ['SLP', 'WIND']:
                times = pd.date_range(dstart,dend,freq='3H')
            elif var == 'RAIN':
                times = pd.date_range(dend,dend)
            # Add a column with WT index
            wt_list = pd.Series(np.zeros(len(times))+WT,dtype=int,index=times)
            # In the first WT create a df, for the remaning, just append data
            if WT == 1:
                df = pd.DataFrame(index=times) # create DataFrame with dates
                df['WT'] = wt_list
            else:
                tmp = pd.DataFrame(index=times)
                tmp['WT'] = wt_list
            print('WT start: '+str(dstart))
            print('WT end: '+str(dend))
            # Get INMET data for the desired WT and remove erroneous data
            df_inmet_var = ProcessINMET(df_inmet,dstart,dend,var)
            # Check if there are values
            if len(df_inmet_var) == 0:
                print('*** No matching data for this station and WT! ***\n')
            else:
                ### Populate df with OLAM data
                odata = get_OLAM_data(WT,var) # Open OLAM data
                ## Slice OLAM data to get only data at the station nearest gridpoint
                slat,slon = float(header['Latitude'].values[0]),float(header['Longitude'].values[0])
                olons, olats = odata.lon, odata.lat
                # Check if station is within domain
                if (slat < np.amin(olats) or slat > np.amax(olats)) and (slon < np.amin(olons) or slon > np.amax(olons)):
                    print(' *** Station outside desired domain ***')
                    pass
                else:
                    lon, lat = find_nearest(olons, slon), find_nearest(olats, slat)
                    if var == 'SLP':
                        olam_station = odata.sslp.sel(lon=slice(lon-.1,lon+.1),
                                                    lat=(slice(lat-.1,lat+.1))).mean('lon').mean('lat')/100
                    elif var == 'WIND':
                        olam_stationu = odata.uwnd.sel(lon=slice(lon-.1,lon+.1),
                                                    lat=(slice(lat-.1,lat+.1)))*units.meter/units.second
                        olam_stationv = odata.uwnd.sel(lon=slice(lon-.1,lon+.1),
                                                    lat=(slice(lat-.1,lat+.1)))*units.meter/units.second
                        olam_station = wind_speed(olam_stationu, olam_stationv).mean('lon').mean('lat')
                    elif var == 'RAIN':
                        olam_station = odata.pt.sel(lon=slice(lon-.1,lon+.1),
                                                    lat=(slice(lat-.1,lat+.1))).mean('lon').mean('lat')
                        olam_station = olam_station.expand_dims(time = [dend])
                    # Create DataFrame with OLAM data to easier indexing
                    odata_series = pd.Series(olam_station,
                                             index=olam_station.time.values)
                    # Add OLAM data to df
                    if WT == 1:
                        df['OLAM'] = odata_series
                    else:
                        tmp['OLAM'] = odata_series
                    ### Populate df with INMET data
                    print(df_inmet_var[:5])
                    if WT == 1:
                        df['INMET'] = df_inmet_var 
                    else:
                        tmp['INMET'] = df_inmet_var
                    ### Populate df with CFSR data
                    # Open CFSR data
                    cdata = get_CFSR_data(dstart,dend,var)
                    # Slice CFSR data to get only data at the station nearest gridpoint
                    clons, clats = cdata.longitude, cdata.latitude
                    lon, lat = find_nearest(clons, slon), find_nearest(clats, slat)
                    if var == 'SLP':
                        cfsr_station = cdata.prmsl.sel(longitude=lon,latitude=(lat))/100
                    elif var == 'WIND': 
                        cfsr_stationu = cdata.u.sel(longitude=lon,latitude=(lat))*units.meter/units.second
                        cfsr_stationv = cdata.u.sel(longitude=lon,latitude=(lat))*units.meter/units.second
                        cfsr_station = wind_speed(cfsr_stationu, cfsr_stationv)
                    elif var == 'RAIN':
                        cfsr_station = cdata.prmsl.sel(longitude=lon,latitude=(lat))*np.nan                   
                    cdata_series = pd.Series(cfsr_station,index=cfsr_station.time.values)
                    if WT == 1:
                        df['CFSR'] = cdata_series
                    else:
                        tmp['CFSR'] = cdata_series
                    ### Populate with ERA5 data
                    # Open ERA5 data
                    edata = get_ERA5_data(dstart, dend,var)
                    # Slice ERA5 data to get only data at the station nearest gridpoint
                    elons, elats = edata.longitude, edata.latitude
                    lon, lat = find_nearest(elons, slon), find_nearest(elats, slat)
                    if var == 'SLP':
                        era_station = edata.msl.sel(longitude=lon,latitude=(lat))/100
                    else:
                        era_station = edata.msl.sel(longitude=lon,latitude=(lat))*np.nan
                    if era_station.time.size == 1:
                        estart = era_station.time.values+era_station.step[0]
                        eend = era_station.time.values+era_station.step[-1]
                        etimes =pd.date_range(estart.values,eend.values, freq='H')
                        edata_series = pd.Series(era_station,index=etimes)
                    else:
                        estart = era_station.time[0].values+era_station.step[0]
                        eend = era_station.time[-1].values+era_station.step[-1]
                        etimes =pd.date_range(estart.values,eend.values, freq='H')
                        edata_series = pd.Series(era_station,index=etimes)
                    if WT == 1:
                        df['ERA5'] = edata_series
                    else:
                        tmp['ERA5'] = edata_series
                    ## Populate with CHIRPS data
                    chdata = rainp.open_CHIRPS(WT)
                    # Slice ERA5 data to get only data at the station nearest gridpoint
                    chlons, chlats = chdata.longitude, chdata.latitude
                    lon, lat = find_nearest(chlons, slon), find_nearest(chlats, slat)
                    if var == 'RAIN':
                        ch_station = chdata.precip.sel(longitude=lon,latitude=(lat)).cumsum('time')[-1]
                    else:
                        ch_station = chdata.precip.sel(longitude=lon,latitude=(lat))*np.nan
                    chdata_series = pd.Series(ch_station.values,index=[ch_station.time.values])
                    if WT == 1:
                        df['CHIRPS'] = chdata_series
                    else:
                        tmp['CHIRPS'] = chdata_series
                        # Append temporary df to df so we have a full
                        # list of dates an values
                        df = df.append(tmp)
        # Save file if there is data on it
        if 'OLAM' in df.columns:
            fname = '../database_'+var+'_validation/'+station_name+'.csv'
            df.to_csv(fname, index_label=['Time'])
            print('saved file: '+fname)
    

if __name__ == '__main__': 
    GetWTStartEnd()
    # create_database('SLP')
#     # create_database('WIND')
    # create_database('RAIN')
    