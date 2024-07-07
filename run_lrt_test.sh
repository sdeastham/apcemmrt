#!/bin/bash
#PBS -N pyLRT
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=1:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

source ~/.bashrc
module load anaconda3/personal
conda activate gcpy

python3 lrt_test.py
