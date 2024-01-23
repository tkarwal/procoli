#!/bin/bash
#
#SBATCH --job-name=lkl_prof
#SBATCH --output=j_lkl_prof__%j.txt
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=2-0:00:00
#SBATCH --partition=your_partition
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=procoli-user@procoli.edu

# source your bash profile, bash rc, or any environments 
source .bash_profile

cd /directory/where/you/want/to/run/things/

# Ideally, name your job with the increment direction too
# Remember to run two jobs, each with a + and - increment 
# Or you can have both in the same python script and they
# will run one after the other 

python example_run.py
