
# Code to extract profile likelihoods using MontePython

### Written by Tanvi Karwal



This code is still being actively developed.  
Prerequisites are an edited branch of [**MontePython**](https://github.com/tkarwal/montepython_lkl_prof) and everything that entails and [**GetDist**](https://getdist.readthedocs.io/en/latest/). Both can be pip installed, but I refer you to their own documentation to see specific installation instructions.  
The rest of the required python modules are likely already installed, particularly if you're running anaconda. They include **numpy, subprocess, os, copy, time, glob**. 

General logic of the code is as follows: 

1. We begin by either running or inheriting from a global mcmc and then a minimization run, where all cosmological and nuisance parameters are varied. The result of this, the global maximum likelihood (or `global_ML`), becomes the starting point for the profile likelihood of the parameter of choice, `prof_param`.  
The code then increments `prof_param` by input increment `prof_incr` and fixes it. This increment should be updated in later versions to be calculated from pre-run mcmc chains if possible.  
2. Then, a minimization is run on all other parameters, beginning from the reference point of the previous (in this case, global) minimization run. Giving a good starting guess helps the minimizer. I am working on quantitatively understanding exactly how much this starting guess helps.  
3. The result of this is stored, and the result provides the new reference starting points for the next run. We again increment `prof_param` by `prof_incr` and run the next minimizer. 

In this way, we go until we hit the bounds `prof_min` or `prof_max` depending on whether `prof_incr` is positive or negative. The code then terminates itself.   

Ideally, the code should be run on several processors to speed up the minimization, i.e., `cpus-per-task` in your job script should be ~4. 
The code should also be run using several parallel chains for robustness, i.e. input `processes` >= 4, say.
For more details, see the **example_run** files. 

The output is the file `<name>_<+/-><prof_param>_lkl_profile.txt` that contains the values of all parameters at the minimized points for each iteration of `prof_param`, plus derived params, and -logL.  
This file can then be plotted as wanted. 

The repo contains functions to do all this, and a code that has the functions in the right order in a jupyter notebook as an example or for tests. 
For actual jobs, I run something like the **example_run.py** file through something like the **example_bash_script.sh** script. 
No need to run this file through mpi. All parallel commands are internally run as subprocesses. 
So just execute 

> `python <name_of_script>.py` 

in the SLURM scipt. Of course, assign the right number of total processors and cpus-per-task. See `example_bash_script.sh`.

I'll eventually update the repo also with timed run of how many processors is optimal, how important good starting guesses are, etc. Benchmarking yet to be done. 

This repo is currently by invite only. 

Cheers!
