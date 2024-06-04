#!/usr/bin/env python3

import os
from pyLRT import RadTran, get_lrt_folder
import copy
import numpy as np
from tqdm import tqdm
from . import apcemm_rt
from . import sza_calc
from datetime import datetime, timedelta

def process_apcemm_data(apcemm_dir, met_data=None, f_list_in=None, column_width=None, **kwargs):
    # met_data:     pre-processed file containing meteorological conditions. Derived from APCEMM directory if not supplied.
    #apcemm_dir = sys.argv[1]
    apcemm_out_dir = os.path.join(apcemm_dir,'APCEMM_out')
    
    if f_list_in is None:
        f_list = [f for f in np.sort(os.listdir(apcemm_out_dir)) if f.endswith('.nc') and f.startswith('ts_aerosol')]
    elif type(f_list_in) != list:
        f_list = [f_list_in]
    else:
        f_list = f_list_in
    
    # Try to read the configuration file - assume it is present in the output directory
    config_file = 'input.yaml'
    config_options = apcemm_rt.read_config(apcemm_dir,config_file)
    # Currently only the met file location is returned
    # For this script, the met file must include:
    #  --> time (fully formatted!) [t]
    #  --> longitude (degrees East) [t]
    #  --> latitude (degrees North) [t]
    #  --> iwc (kg/m3) [z,t]
    #  --> lwc (kg/m3) [z,t]
    #  --> altitude_edges (m) [z+1,t]
    # For APCEMM, the met file must include:
    #  --> time (assumed time interval) [t]
    #  --> altitude (km, midpoints) [z]
    #  --> pressure (hPa, midpoints) [z]
    #  --> shear (s-1) [z,t]
    #  --> relative_humidity_ice (%) [z,t]
    #  --> temperature (K) [z,t]
    #  --> w (m s-1, based on comments) [z,t]
   
    if met_data is None: 
        met_data = apcemm_rt.read_met_file(config_options['met_file_location'])
   
    if 'lon' not in met_data:
        print('Using dummy longitude data')
        n_times_met = met_data['time'].size
        met_data['lon'] = np.zeros(n_times_met) + 0.0

    if 'lat' not in met_data:
        print('Using dummy latitude data')
        n_times_met = met_data['time'].size
        met_data['lat'] = np.zeros(n_times_met) + 30.0
 
    # Seconds since start for each sample
    t_offset = np.zeros(len(f_list))
    for i_f, f in enumerate(f_list):
        hhmm = int(f.split('_')[-1].split('.')[0])
        time_parsed = 3600.0 * np.floor(hhmm/100.0) + 60.0 * (hhmm%100)
        t_offset[i_f] = time_parsed
    
    # Get longitude and latitude at the output times
    t_base = met_data['time'][0]
    t_met_data = np.array([(t - t_base).total_seconds() for t in met_data['time'][:]])
    lon_vec = np.interp(t_offset,t_met_data,met_data['lon'])
    lat_vec = np.interp(t_offset,t_met_data,met_data['lat'])
    
    # Also get the actual datetime objects and generate SZA
    t_vec = []
    sza_vec = []
    for i_t, t in enumerate(t_offset):
        t_vec.append(t_base + timedelta(seconds=t))
        sza_vec.append((180.0/np.pi) * np.arccos(sza_calc.global_csza(t_vec[-1],lon_vec[i_t],lat_vec[i_t])))
    
    #TODO: Met processing
    liquid_cloud = 'lwc' in met_data and np.sum(met_data['lwc'][:,i_time_met]) > 0.1e-9
    
    quiet = True
    verbose = False
    debug = False
    
    #quiet = False
    #verbose = True
    #debug = True
    f_out_list = []
    out_data_list = []
    
    with tqdm(total=len(f_list)) as pbar:
        for i_file, short_file in enumerate(f_list):
            apcemm_file = os.path.join(apcemm_out_dir,short_file)
            # Get the solar zenith angle in degrees
            sza = sza_vec[i_file]
            
            f_save = apcemm_file.replace('.nc','.npz')
            
            LIBRADTRAN_FOLDER = get_lrt_folder()
            gsl_lib_path = '/sw-eb/software/GSL/2.7-GCC-11.3.0/lib'
            conda_lib_path = '/rds/general/user/seastham/home/anaconda3/envs/gcpy/lib'
            env = {'LD_LIBRARY_PATH': gsl_lib_path + ':' + conda_lib_path}
            
            #TODO: Get the ambient conditions as a vector, then make the below a function which is run once per APCEMM output file
            
            # Set up output
            albedo_solar = 0.2
            albedo_thermal = 0.1
            spectrum_solar   = 'kato2'
            spectrum_thermal = 'fu'
            
            ice_data = 'yang'
            # For Yang
            habit_vec = ['solid-column','hollow-column','rough-aggregate','plate','droxtal']#,'spheroid']
            habit = 'rough-aggregate'
            r_minimum = 3.55e-6 # Data-specific
            r_maximum = 108.10e-6 # Data-specific
            
            # Options for a liquid cloud - representative
            # This would be a cloud with optical depth 15 (!)
            # NEED A MET PROCESSING SCRIPT
            #cld_base = 1.0 # km
            #cld_top  = 2.0 # km
            #cld_lwc  = 0.1 # g/m3
            #cld_r    = 10.0 # um
            
            # Contrail properties (this one should have OD of ~0.1)
            #contrail_altitude_edges, contrail_iwc, contrail_r, contrail_qv = process_apcemm_data(apcemm_file)
            #processed_apcemm = apcemm_rt.process_apcemm_data(apcemm_file) # Store the data in a single list/dict for convenience?
            #ice_altitude_edges, iwc, ice_r, liquid_altitude_edges, lwc, liquid_r, data_for_profile  = merge_contrail_with_met(processed_apcemm, processed_met)
            #
            #contrail_base = 9.0 # km
            #contrail_top  = 10.0 # km
            #contrail_iwc  = 0.015 # g/m3
            #contrail_r    = 20.0 # um
            #column_width = None # Just use one column (!)
            column_x_edge, contrail_z_edge, contrail_p_edge, contrail_r_m, contrail_iwc = apcemm_rt.process_apcemm_data(apcemm_file,column_width,**kwargs)
        
            contrail_r_m[contrail_r_m <= r_minimum] = r_minimum * 1.0001
            contrail_r_m[contrail_r_m >= r_maximum] = r_maximum * 0.9999
            n_columns = column_x_edge.size - 1
            
            # CODE STARTS HERE
            
            # Notes:
            #   yang:     works
            #   yang2013: needs interpolation to wavelengths; this takes a LOT of memory
            #   baum:     needs interpolation; also not available for 2200 to 3076.92 nm
            #   fu:       only available for 250 - 4990 nm
            #   hey:      needs interpolation; only available for 200 - 5000 nm
            #   key:      only available for 200 - 5000 nm
            #   baum_v36: needs interpolation; data files missing?
            
            for i_x in range(n_columns):
                # Options common to both the solar and thermal calculation
                xopts = {}
                xopts['pseudospherical'] = ''
                xopts['data_files_path'] = '/rds/general/user/seastham/home/libRadtran/lrt_2.0.5/share/libRadtran/data'
                xopts['rte_solver'] = 'disort'
                xopts['pressure_out'] = 'toa'
            
                if spectrum_solar == 'kato2':
                    solar_wl   = '250 2600'
                else:
                    solar_wl   = '200 2600'
                thermal_wl = '2500 80000'
            
                #TODO: Read in surface conditions
                #TODO: Transmit relative humidity field
            
                # Set up the contrail
                #TODO: Multiple layers
                #contrail_data = {'z':   np.array([contrail_top,contrail_base]),
                #                 'iwc': np.array([0, contrail_iwc]), # g/m3
                #                 're':  np.array([0, contrail_r])}
        
                # Need to pad (zero IWC at top) and flip (top -> bottom)
                contrail_iwc_offset = np.zeros(contrail_z_edge.size)
                contrail_r_m_offset = np.zeros(contrail_z_edge.size)
                contrail_iwc_offset[1:] = np.flip(contrail_iwc[:,i_x])
                contrail_r_m_offset[1:] = np.flip(contrail_r_m[:,i_x])
           
                contrail_data = {'z':   np.flip(contrail_z_edge * 0.001),
                                 'iwc': contrail_iwc_offset * 1000.0,
                                 're':  contrail_r_m_offset * 1.0e6}

                #TODO: Add liquid clouds with multiple layers
                #if liquid_cloud:
                #    cloud_data = {'z':   np.array([cld_top, cld_base]),
                #                  'lwc': np.array([0,cld_lwc]),
                #                  're':  np.array([0,cld_r])}
                ic_opts = {}
                if ice_data == 'yang2013':
                    ice_data = ice_data + ' interpolate'
                    ic_opts['ic_habit_yang2013'] = '{:s} {:s}'.format(habit, roughness)
                elif ice_data == 'hey':
                    ice_data = ice_data + ' interpolate'
                    ic_opts['ic_habit'] = habit
                elif ice_data == 'yang':
                    ic_opts['ic_habit'] = habit
                elif ice_data == 'key':
                    ic_opts['ic_habit'] = habit
                elif ice_data == 'baum':
                    ice_data = ice_data + ' interpolate'
                    #ic_opts['ic_habit'] = habit
                elif ice_data == 'baum_v36':
                    ice_data = ice_data + ' interpolate'
            
                slrt = RadTran(LIBRADTRAN_FOLDER,env=env)
                slrt.options['sza'] = sza
                slrt.options['albedo'] = albedo_solar
                slrt.options['source'] = 'solar'
                slrt.options['wavelength'] = solar_wl
                #slrt.options['output_user'] = 'lambda eglo eup edn edir'
                slrt.options['output_user'] = 'p edir edn eup'
                slrt.options['mol_abs_param'] = spectrum_solar
                if spectrum_solar in ['fu','kato2']:
                    slrt.options['output_process'] = 'sum'
                else:
                    slrt.options['output_process'] = 'integrate'
                    
                if liquid_cloud:
                    slrt.cloud = cloud_data 
            
                #slrt.options['umu'] = '-1.0 1.0'
                for key, val in xopts.items():
                    slrt.options[key] = val
            
                tlrt = RadTran(LIBRADTRAN_FOLDER,env=env)
                tlrt.options['source'] = 'thermal'
                tlrt.options['albedo'] = albedo_thermal
                #tlrt.options['data_files_path'] = '/home/seastham/libRadtran/lrt_2.0.5/share/libRadtran/data'
                #tlrt.options['output_user'] = 'lambda edir eup uu'
                tlrt.options['output_user'] = 'p edir edn eup'
                tlrt.options['wavelength'] = thermal_wl
                tlrt.options['mol_abs_param'] = spectrum_thermal
                if spectrum_thermal in ['fu','kato2']:
                    tlrt.options['output_process'] = 'sum'
                else:
                    tlrt.options['output_process'] = 'integrate'
                for key, val in xopts.items():
                    tlrt.options[key] = val
            
                if liquid_cloud:
                    tlrt.cloud = cloud_data 
           
                # Only run SW calculation if there is sunlight
                # Nighttime calculation/sunset/sunrise not currently functioning
                if sza > 0.0 and sza <= 90.0:
                    slrt_cld = copy.deepcopy(slrt)
                    slrt_cld.icecloud = contrail_data
                    slrt_cld.options['ic_properties'] = ice_data
                    for key, val in ic_opts.items():
                        slrt_cld.options[key] = val

                    scdata = slrt_cld.run(verbose=verbose,quiet=quiet,debug=debug)
                    sdata = slrt.run(verbose=verbose,quiet=quiet,debug=debug)
                    sw = (scdata[3] - scdata[2]) - (sdata[3] - sdata[2])
                else:
                    sw = 0.0
            
                tlrt_cld = copy.deepcopy(tlrt)
                tlrt_cld.icecloud = contrail_data
                tlrt_cld.options['ic_properties'] = ice_data
                for key, val in ic_opts.items():
                    tlrt_cld.options[key] = val
            
                # Run the RT
                tcdata = tlrt_cld.run(verbose=verbose,quiet=quiet,debug=debug)
                tdata = tlrt.run(verbose=verbose,quiet=quiet,debug=debug)
            
                # Upwelling - downwelling, difference between cloudy and clear
                lw = (tcdata[3] - tcdata[2]) - (tdata[3] - tdata[2])
                net = lw + sw
                result = np.array([net,lw,sw])
            
                if (i_x == 0):
                    width = np.diff(column_x_edge)
                    output_data = {'habit':            habit,
                                   'albedo':           albedo_solar,
                                   'albedo_thermal':   albedo_thermal,
                                   'sza':              sza,
                                   'lw':               np.zeros(n_columns),
                                   'sw':               np.zeros(n_columns),
                                   'net':              np.zeros(n_columns),
                                   'width':            width,
                                   'x':                column_x_edge[:-1]}
                output_data['lw'][i_x] = lw
                output_data['sw'][i_x] = sw
                output_data['net'][i_x] = net

            f_out_list.append(f_save)
            out_data_list.append(output_data)
            pbar.update(1)
    return f_out_list, out_data_list

if __name__ == '__main__':
    import sys
    assert len(sys.argv) > 1, f'Need at least 1 arguments: APCEMM directory and, unless you want to process the whole directory, the file list. Only received {len(sys.argv)}'
    if len(sys.argv) < 3:
        f_list = None
    else:
        f_list = sys.argv[2:]
    f_out_list, out_data_list = process_apcemm_data(sys.argv[1], f_list_in=f_list, column_width=100)
    for f_save, output_data in zip(f_out_list,out_data_list):
        np.savez(f_save,**output_data)
        print('Saved to ' + f_save)
