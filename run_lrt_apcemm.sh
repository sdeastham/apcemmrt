#!/bin/bash
#PBS -N pyLRT
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=1:00:00
#PBS -j oe

if [[ "z$PBS_O_WORKDIR" != "z" ]]; then
    cd $PBS_O_WORKDIR
fi

source ~/.bashrc
module load anaconda3/personal
conda activate gcpy

#apcemm_output_dir=/rds/general/user/seastham/home/simulations/C2024-04-10_APCEMM/Run/test_001/APCEMM_out
apcemm_output_dir=/rds/general/user/seastham/home/simulations/C2024-04-10_APCEMM/Run_rhisub20/long_rhisub20_0500m_140pct_s002_000up_nvpm0001
#python3 lrt_apcemm.py $apcemm_output_dir ts_aerosol_case0_0400.nc ts_aerosol_case0_0200.nc
python3 lrt_apcemm.py $apcemm_output_dir
