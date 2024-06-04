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
            # WARNING: Assuming here all grid boxes are the same width
            crystal_radius[:,i_x] = np.sum(weighting[:,i0:i1] * effective_radius[:,i0:i1],axis=1)/np.sum(weighting[:,i0:i1],axis=1)
            ice_water_content[:,i_x] = np.mean(ice_water_content_native[:,i0:i1],axis=1)
            layer_weights[:,i_x] = np.sum(weighting[:,i0:i1])
        else:
            crystal_radius[:,i_x] = 1.0e-9
            ice_water_content[:,i_x] = 0.0
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
