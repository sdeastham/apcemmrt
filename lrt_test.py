#!/usr/bin/env python3

import os
from pyLRT import RadTran, get_lrt_folder
import matplotlib.pyplot as plt
import copy
import numpy as np
from tqdm import tqdm

LIBRADTRAN_FOLDER = get_lrt_folder()
gsl_lib_path = '/sw-eb/software/GSL/2.7-GCC-11.3.0/lib'
conda_lib_path = '/rds/general/user/seastham/home/anaconda3/envs/gcpy/lib'
env = {'LD_LIBRARY_PATH': gsl_lib_path + ':' + conda_lib_path}

# Set up output
albedo_solar_vec = np.linspace(0.0,1.0,11) # Shortwave
sza_vec = np.linspace(0.0,90.0,20)
albedo_thermal = 0.1
spectrum_solar   = 'kato2'
spectrum_thermal = 'fu'

ice_data = 'yang'
# For Yang
habit_vec = ['solid-column','hollow-column','rough-aggregate','plate','droxtal']#,'spheroid']

# Options for a liquid cloud - representative
# This would be a cloud with optical depth 15 (!)
cld_base = 1.0 # km
cld_top  = 2.0 # km
cld_lwc  = 0.1 # g/m3
cld_r    = 10.0 # um

# Contrail properties (this one should have OD of ~0.1)
contrail_base = 9.0 # km
contrail_top  = 10.0 # km
contrail_iwc  = 0.015 # g/m3
contrail_r    = 20.0 # um

# CODE STARTS HERE
#liquid_cloud = True
#for liquid_cloud in [True,False]:
for liquid_cloud in [False]:
    if liquid_cloud:
        f_save = 'lrt_test_cloudy'
    else:
        f_save = 'lrt_test_clear'
    quiet = True
    
    # Notes:
    #   yang:     works
    #   yang2013: needs interpolation to wavelengths; this takes a LOT of memory
    #   baum:     needs interpolation; also not available for 2200 to 3076.92 nm
    #   fu:       only available for 250 - 4990 nm
    #   hey:      needs interpolation; only available for 200 - 5000 nm
    #   key:      only available for 200 - 5000 nm
    #   baum_v36: needs interpolation; data files missing?
    
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
    
    # Set up the contrail
    contrail_data = {'z':   np.array([contrail_top,contrail_base]),
                     'iwc': np.array([0, contrail_iwc]), # g/m3
                     're':  np.array([0, contrail_r])}
    
    if liquid_cloud:
        cloud_data = {'z':   np.array([cld_top, cld_base]),
                      'lwc': np.array([0,cld_lwc]),
                      're':  np.array([0,cld_r])}
    
    n_albedo = len(albedo_solar_vec)
    n_sza = len(sza_vec)
    n_habit = len(habit_vec)
    output_data = np.zeros((3,len(albedo_solar_vec),len(sza_vec),len(habit_vec)))
    
    i_run = 0
    with tqdm(total=n_albedo*n_sza*n_habit) as pbar:
        for i_albedo, albedo_solar in enumerate(albedo_solar_vec):
            for i_sza, sza in enumerate(sza_vec):
                for i_habit, habit in enumerate(habit_vec):
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
    #                slrt.options['output_user'] = 'lambda eglo eup edn edir'
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
    #                tlrt.options['data_files_path'] = '/home/seastham/libRadtran/lrt_2.0.5/share/libRadtran/data'
    #                tlrt.options['output_user'] = 'lambda edir eup uu'
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
                    
                    slrt_cld = copy.deepcopy(slrt)
                    slrt_cld.icecloud = contrail_data
                    slrt_cld.options['ic_properties'] = ice_data
                    for key, val in ic_opts.items():
                        slrt_cld.options[key] = val
    
                    tlrt_cld = copy.deepcopy(tlrt)
                    tlrt_cld.icecloud = contrail_data
                    tlrt_cld.options['ic_properties'] = ice_data
                    for key, val in ic_opts.items():
                        tlrt_cld.options[key] = val
                    
                    # Run the RT
                    scdata = slrt_cld.run(verbose=False,quiet=quiet)
                    tcdata = tlrt_cld.run(verbose=False,quiet=quiet)
                    
                    sdata = slrt.run(verbose=False,quiet=quiet)
                    tdata = tlrt.run(verbose=False,quiet=quiet)
    
                    # Upwelling - downwelling, difference between cloudy and clear
                    lw = (tcdata[3] - tcdata[2]) - (tdata[3] - tdata[2])
                    sw = (scdata[3] - scdata[2]) - (sdata[3] - sdata[2])
                    net = lw + sw
                    result = np.array([net,lw,sw])
                    output_data[:,i_albedo,i_sza,i_habit] = result
                    i_run += 1
                    pbar.update(1)
    
    np.savez(f_save,output_data=output_data,sza=sza_vec,albedo=albedo_solar_vec,habit=habit_vec)
    print('Saved to ' + f_save)

