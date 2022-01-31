-----------------------------------------------
Code to extract profile likelihoods from Cobaya
-----------------------------------------------
Written by Tanvi Karwal
-----------------------


This code is still being actively developed. 
Prerequisites are Cobaya and everything that entails and GetDist. The rest of the required python modules are likely already installed, particularly if you're running anaconda. 

The code currently relies on pre-run mcmc chains, although this is not a requirement and can be removed by the user / will be in an updated version. 

We begin by either running or inheriting from a global minimization run, where all cosmological and nuisance parameters are varied. The result of this, the global maximum likelihood (or global ML), becomes the starting point for the profile likelihood of the parameter of choice, prof_param. 
The code then increments prof_param by input increment prof_incr and fixes it. This increment should be updated in later versions to be calculated from pre-run mcmc chains if available. 
Then, a minimization is run on all other parameters, beginning from the reference point of the previous (in this case, global) minimization run. Giving a good starting guess helps the minimizer. I am working on quantitatively understanding exactly how much this starting guess helps. 
The result of this is stored, the result provides the new reference, starting points for the next run. We again increment prof_param by prof_incr and run the next minimizer. 
In this way, we go until we hit the bounds prof_min or prof_max depending on whether prof_incr is positive or negative. 
The code then terminates itself. 
Ideally, the code should be run on several processors to speed up the minimization.

The output is a file of <name>_lkl_profile.txt that contains the values of all parameters at the minimized points for each iteration of prof_param, plus derived params, and -logL and chi-sq. 
This file can then be plotted as wanted. 

The repo contains functions to do all this, and a code that has the functions in the right order in a jupyter notebook. 
I turn this notebook into a python .py file and run that through a job script. No need to run this file through mpi. All parallel commands are internally run as subprocesses. 
So just execute 

python <name_of_script>.py 

in the SLURM scipt. Of course, assign the right number of total processors and cpus-per-task. 

I'll eventually update the repo also with timed run of how many processors is optimal, how important good starting guesses are, etc. 

And if you use this code without my permission, I will track you down and throw rotten eggs at you, filled with food colouring that will not wash off. 

Cheers!
