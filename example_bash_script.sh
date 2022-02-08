#!/bin/bash
#
#SBATCH --job-name=lkl_prof
#SBATCH --output=j_lcdm_lkl_prof.txt
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=2-0:00:00
#SBATCH --partition=highcore
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karwal@sas.upenn.edu

cd ~
source .bash_profile

which python
which gcc
which mpirun

cd /home2/karwal/lkl_prof/

python LCDM_tests.py
