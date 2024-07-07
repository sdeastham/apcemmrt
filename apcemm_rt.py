#!/usr/bin/env python3

import numpy as np
import os
import tqdm.auto as tqdm
from netCDF4 import Dataset
from datetime import datetime, timedelta

c0s = -6024.5282
c1s = 24.7219
c2s = 0.010613868
c3s = -1.3198825E-5
c4s = -0.49382577
def p_sat_ice_sonntag(T: float) -> float:
    #return 100.0 * np.exp(-6024.5282 / T + 24.7219 + 0.010613868 * T - 1.3198825E-5 * T * T - 0.49382577 * np.log(T))
    return 100.0 * np.exp(c0s/T + c1s + c2s*T + c3s*T*T + c4s*np.log(T))

# From https://stackoverflow.com/a/44271357
def convert_datetime64_to_datetime( usert: np.datetime64 )->datetime:
    t = np.datetime64( usert, 'us').astype(datetime)
    return t

from typing import List, Optional
# For reading from CG data:
#  -> 2D ['aluvd','aluvp','alnid','alnip']:
#  --> aluvd:     Albedo (UV direct) [-]
#  --> aluvp:     Albedo (UV diffuse) [-]
#  --> alnid:     Albedo (near-IR direct) [-]
#  --> alnip:     Albedo (near-IR diffuse) [-]
#  -> 3D (NB: ciwc and clwc are grid cell averages, and the divisor is dry air + water vapour + 
#         cloud water + cloud ice + precipitating water + precipitating ice)
#  --> cc:        3D cloud fraction [-]
#  --> ciwc:      Cloud ice water content [kg ice/kg grid cell air]
#  --> clwc:      Cloud liquid water content [kg ice/kg grid cell air]
#  --> q:         Specific humidity [kg water/kg total air] (https://codes.ecmwf.int/grib/param-db/247)
#  --> t:         Temperature [K]
def extract_era5(time_vec: List[float],lon_vec: List[float],lat_vec: List[float],
                 var_list_sfc: List[str], var_list_plev: List[str],
                 sfc_file_format: str, plev_file_format: str,
                 lon_edge: Optional[List[float]] = None,lat_edge: Optional[List[float]] = None,
                 lon_var: str = 'longitude', lat_var: str = 'latitude', lev_var: str = 'isobaricInhPa'):
    # time_vec:         List of datetimes
    # lon_vec:          List of longitudes (degrees)
    # lat_vec:          List of latitudes (degrees)
    # var_list_sfc:     List of variable names to be read from ERA5 surface data file
    # var_list_plev:    List of variable names to be read from ERA5 pressure level file
    # lon_edge:         Longitude edges of the input met data (degrees)
    # lat_edge:         Latitude edges of the input met data (degrees)
    
    # Read dimension data if not supplied
    n_times = len(time_vec)
    n_2D_vars = len(var_list_sfc)
    n_3D_vars = len(var_list_plev)
    data_2D = {}
    data_3D = {}
    for var in var_list_sfc:
        data_2D[var] = np.zeros(n_times)
    n_levs = None
    lon_idx = None
    lat_idx = None
    for time in time_vec:
        year  = time.year
        month = time.month
        day   = time.day
        with Dataset(sfc_file_format.format(year,month,day)) as nc:
            t_file_int = nc['time'][...] # Usually seconds since 1970-01-01, but this should be checked...
            
            if lon_era5 is None or lat_era5 is None:
                lon_era5 = nc[lon_var][...]
                lat_era5 = nc[lat_var][...]
                nlon = len(lon_era5)
                nlat = len(lat_era5)
                lon_edge = np.zeros(nlon+1)
                lat_edge = np.zeros(nlat+1)
                dlon = np.median(np.diff(lon_era5))
                dlat = np.median(np.diff(lat_era5))
                lon_edge[:-1] = lon_era5 - (dlon*0.5)
                lon_edge[-1] = lon_edge[-2] + dlon
                lat_edge[1:-1] = lat_era5 - (dlat*0.5)
                lat_edge[0] = lat_edge[1] - dlat
                lat_edge[-1] = lat_edge[-2] + dlat
            
            if lon_idx is None or lat_idx is None:
                # Get the indices to be returned
                lon_idx, lat_idx = get_lonlat_idx(lon_vec,lat_vec,lon_edge,lat_edge)
            
            #for var in var_list_sfc:
                
        with Dataset(plev_file_format.format(year,month,day)) as nc:
            if n_levs is None:
                n_levs = len(nc[lev_var])
                for var in var_list_plev:
                    data_3D[var] = np.zeros((n_times,n_levs))
    return data_2D, data_3D, lon_edge, lat_edge

def get_lonlat_idx(lon,lat,lon_edge,lat_edge):
    # Are the latitudes order south to north?
    s_to_n = lat[1] > lat[0]
    # Make sure that the longitudes are using the same "edges" as the globe
    lon_mod = lon
    i_minlon = np.argmin(lon_mod)
    while lon_mod[i_minlon] < lon_edge[0]:
        lon_mod += 360.0
    i_maxlon = np.argmax(lon_mod)
    while lon_mod[i_maxlon] >= lon_edge[-1]:
        lon_mod -= 360.0
    dlon = np.median(np.diff(lon_edge))
    dlat = np.median(np.diff(lat_edge))
    
    lon_idx = np.floor((lon_mod - lon_edge)/dlon)
    if s_to_n:
        lat_idx = np.floor((lat - lat_edge)/dlat)
    else:
        lat_idx = np.ceil((lat - lat_edge)/(-1.0*dlat))
    return np.int32(lon_idx), np.int32(lat_idx)

def process_apcemm_data(f,column_width=None,weight_by_area=True,collapse_vertical=True):
    with Dataset(f,'r') as nc:
        x_mid_native = nc['x'][:] # m
        p_mid = nc['Pressure'][:] # Pa
        z_mid = nc['Altitude'][:] # m, from surface (y)
        temperature_mid = nc['Temperature'][:,:] # K, 2D (y,x)
        effective_radius = nc['Effective radius'][:,:] # m, 2D (y,x)
        ice_water_content_native = nc['IWC'][:,:] # kg/m3, 2D (y,x)
        if (weight_by_area):
            weighting = nc['Ice aerosol surface area'][:,:] # m2/m3, 2D (y,x)
        else:
            weighting = nc['Ice aerosol particle number'][:,:] # num/m3, 2D (y,x)
    n_lev = z_mid.size
    n_x_native = x_mid_native.size
    z_edge = np.zeros(n_lev+1)
    z_edge[1:-1] = (z_mid[:-1] + z_mid[1:]) / 2.0
    z_edge[0]  = z_edge[1]  - (0.5*(z_edge[1] - z_mid[0]  ))
    z_edge[-1] = z_edge[-2] + (0.5*(z_mid[-1] - z_edge[-2]))
    p_edge = np.zeros(n_lev+1)
    p_edge[0]  = p_edge[1]  - (0.5*(p_edge[1] - p_mid[0]  ))
    p_edge[-1] = p_edge[-2] + (0.5*(p_mid[-1] - p_edge[-2]))
    x_edge_native = np.zeros(n_x_native+1)
    x_edge_native[1:-1] = (x_mid_native[1:] + x_mid_native[:-1]) * 0.5
    x_edge_native[0]  = x_edge_native[1]  - (0.5*(x_edge_native[1]  - x_mid_native[0] ))
    x_edge_native[-1] = x_edge_native[-2] + (0.5*(x_edge_native[-2] - x_mid_native[-1]))
    width_native = np.diff(x_edge_native)

    # Find the edges of the contrail
    iwc_sum = np.sum(ice_water_content_native,axis=0).squeeze()
    idx_nonzero = np.nonzero(iwc_sum)[0]
    i0 = idx_nonzero[0]
    i1 = idx_nonzero[-1]
    if column_width is None:
        x_edge = np.array([x_edge_native[i0],x_edge_native[i1+1]])
    else:
        n_columns = int(np.ceil((x_edge_native[i1+1] - x_edge_native[i0])/column_width))
        x_edge = np.arange(0,(n_columns*column_width) + (column_width/10.0),column_width) + x_edge_native[i0]
        #print(f'{n_columns} columns of width {column_width} m')
    width = np.diff(x_edge)
    n_x = width.size # Should be the same as n_columns if column width is not None
    layer_weights = np.zeros((n_lev,n_x))
    crystal_radius = np.zeros((n_lev,n_x))
    ice_water_content = np.zeros((n_lev,n_x))
    x_mid = (x_edge[1:] + x_edge[:-1]) * 0.5
    for i_x in range(n_x):
        idx_nonzero_local = np.nonzero(np.logical_and(x_mid_native > x_edge[i_x],x_mid_native < x_edge[i_x + 1]))[0]
        if len(idx_nonzero_local) > 0:
            i0 = idx_nonzero_local[0]
            i1 = idx_nonzero_local[-1] + 1 # Add 1 because this one wil be excluded
            #print(f'{i_x:5d}: {i0:6d} -> {i1:6d}')
            # For consisteny isolate the cells which we are using 
            local_width_vec = width_native[i0:i1]
            local_width_sum = np.sum(local_width_vec)
            crystal_radius[:,i_x] = np.sum(local_width_vec * weighting[:,i0:i1] * effective_radius[:,i0:i1],axis=1)/np.sum(local_width_vec * weighting[:,i0:i1],axis=1)
            ice_water_content[:,i_x] = np.sum(local_width_vec * ice_water_content_native[:,i0:i1],axis=1)/local_width_sum
            layer_weights[:,i_x] = np.sum(weighting[:,i0:i1] * local_width_vec) / local_width_sum
            width[i_x] = local_width_sum
        else:
            crystal_radius[:,i_x] = 1.0e-9
            ice_water_content[:,i_x] = 0.0
            width[i_x] = 0.0
    if collapse_vertical:
        z_edge = np.array([z_edge[0],z_edge[-1]])
        p_edge = np.array([p_edge[0],p_edge[-1]])
        ice_water_content = np.mean(ice_water_content,axis=0,keepdims=True)  # Assumes layer thicknesses are constant
        crystal_radius = np.sum(crystal_radius * layer_weights,axis=0,keepdims=True)/np.sum(layer_weights,axis=0,keepdims=True)
    return x_edge, z_edge, p_edge, crystal_radius, ice_water_content

def read_config(apcemm_dir,config_file):
    import yaml
    with open(os.path.join(apcemm_dir,config_file),'r') as stream:
        config = yaml.safe_load(stream)
        met_file = config['METEOROLOGY MENU']['METEOROLOGICAL INPUT SUBMENU']['Met input file path (string)']
        met_file_location = os.path.join(apcemm_dir,met_file)
    return {'met_file_location': met_file_location}

def read_met_file(met_file):
    import xarray as xr
    # Read in and return the necessary data subset
    data_out = {}
    with xr.open_dataset(met_file) as nc:
        var_list = list(nc.variables)
        n_times = nc['time'].size
        data_out['time'] = np.array([convert_datetime64_to_datetime(t) for t in nc['time'].values])
        for var in ['lon','lat','altitude','temperature','iwc','lwc','relative_humidity_ice','pressure']:
            if var not in var_list:
                #print(f'{var} not present in met data; skipping')
                continue
            data_raw = nc[var].values.copy()
            # If this is a data variable (i.e. not a dimension) but we don't have time variation,
            # add time "variation" by duplicating the data into every time
            if var not in nc.dims and 'time' not in nc[var].dims:
                #print(f'Duplicating {var} data to span {n_times} times')
                data = np.zeros(list(data_raw.shape) + [n_times])
                for i_time in range(n_times):
                    data[...,i_time] = data_raw[...]
            else:
                data = data_raw
            data_out[var] = data
        # Derive specific humidity
        rhi = data_out['relative_humidity_ice'][...] * 0.01 # Convert from % to fraction
        p_h2o = rhi * p_sat_ice_sonntag(data_out['temperature'][...]) * 0.01 # Convert from Pa to hPa
        vmr_h2o = p_h2o / data_out['pressure'][...] # mol h2o per mol total air
        mmrdry_h2o = (vmr_h2o / (1.0 - vmr_h2o)) * (18.0/28.97) # kg h2o per kg dry air
        data_out['qv'] = mmrdry_h2o / (1.0 + mmrdry_h2o) # kg h2o per kg total air

        # Convert time from numpy.timedelta64 to datetime timedelta
        
    return data_out
