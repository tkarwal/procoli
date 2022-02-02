#!/usr/bin/env python
# coding: utf-8

# In[1]:


from getdist import mcsamples, plots, chains
from getdist.mcsamples import MCSamplesError
import numpy as np
from subprocess import run
import os
from yaml import dump
from cobaya.yaml import yaml_load_file
from copy import deepcopy


# ### Variables used for functions and testing them 

# In[ ]:


# chains_dir = "/Users/tanvikarwal/Desktop/Early_dark_energy/likelihood_profile/chains/lcdm_base/"
# chain_file = 'lcdm_cmb_bao_sne_'
# chains_dir = '/home2/karwal/mcmc_chains/ede_lkl_profile/lcdm_base/'
# chain_file = 'lcdm_cmb_bao_sne_'
# os.chdir(chains_dir)

# settings = {'ignore_rows' : 0.2}
# mcmc_chains = mcsamples.loadMCSamples(chains_dir+chain_file, settings=settings)


# ### Read minimum file and save parameter names and values lists and MLs dictionary 

# In[ ]:


def read_minimum(chains_dir, chain_file, extension='_lkl_prof'):
    param_ML, param_names = np.loadtxt(chains_dir + chain_file + extension + '.minimum', skiprows=3, usecols = (1,2), dtype=str, unpack=True)
    param_ML = param_ML.astype(float)

    with open(chains_dir + chain_file + extension + '.minimum') as min_file:
        loglkl_and_chi = [next(min_file) for x in range(2)] # reading in the first two lines separately for -log(lkl) and chi^2
    for line in loglkl_and_chi:
        param_names = np.append(param_names, line.split("=")[0])
        param_ML = np.append(param_ML, float(line.split("=")[1]))
    MLs = dict(zip(param_names, param_ML))
    return param_names, param_ML, MLs


# In[ ]:


# param_names, param_values, MLs = read_minimum(chains_dir=chains_dir, chain_file=chain_file)
# MLs


# ### Read last line of lkl prof output file into list and update MLs

# In[ ]:


def read_lkl_output(chains_dir, chain_file, extension='_lkl_profile.txt', loc=-1):
    lkl_prof_table = np.loadtxt(chains_dir + chain_file + '_lkl_profile.txt')
    try:
        lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
        lkl_prof_table = lkl_prof_table[-1, :]
    except IndexError:
        pass
    return lkl_prof_table


# ### Write params from MLs dict into txt file in append mode

# In[ ]:


def write_MLs(param_names, MLs, chains_dir, chain_file, extension='_lkl_profile.txt'):
    with open(chains_dir + chain_file + extension, 'a') as lkl_txt:
        for param in param_names:
            lkl_txt.write("\t %s" % str(MLs[param]))
        lkl_txt.write("\n")
    lkl_prof_table = np.loadtxt(chains_dir + chain_file + extension)
    return lkl_prof_table.shape


# In[ ]:


# write_MLs(chains_dir=chains_dir, chain_file=chain_file)


# ### Check that param names match in target file and MLs dictionary

# In[ ]:


def match_param_names(param_names, chains_dir, chain_file, extension='_lkl_profile.txt'):
    with open(chains_dir + chain_file + extension, 'r') as lkl_txt:
        params_recorded = lkl_txt.readline()
    # params_recorded = params_recorded
    # define the expected first row of this file
    expected_string = '#'
    for param in param_names:
        expected_string += "\t %s" % param
    expected_string += "\n"
    if expected_string == params_recorded:
        print("match_param_names: Found existing file with correct name and parameters / parameter sequence. Will append to it. \n" 
                 + chains_dir + chain_file + extension)
        return True
    else:
        print("match_param_names: Error: existing file found at " + chains_dir + chain_file + extension 
             + "\n but parameters / parameter sequence does not match expected.")
        print("--> parameters found: \n" + params_recorded)
        print("--> parameters expected: \n" + expected_string)
        return False


# In[ ]:


# match_param_names()


# ### Check if some location in lkl_prof output file matches current MLs

# In[ ]:


def match_param_line(param_names, MLs, chains_dir, chain_file, extension='_lkl_profile.txt', loc=-1):
    lkl_prof_table = np.loadtxt(chains_dir + chain_file + '_lkl_profile.txt')
    if lkl_prof_table.size==0:
        print("match_param_line: File empty ")
        return False
    else: 
        try:
            lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
            if False in [lkl_prof_table[loc, np.where(param_names == param)] == MLs[param] for param in param_names]:
                return False
            else:
                return True 
        except IndexError:
            print("match_param_line: Only one entry in file, checking that entry ")
            if False in [lkl_prof_table[np.where(param_names == param)] == MLs[param] for param in param_names]:
                return False 
            else:
                return True    


# In[ ]:


# match_param_line(param_names, MLs, loc=-1)


# ### Updated yaml info to next increment 

# In[ ]:


def increment_update_yaml(chains_dir, chain_file, MLs, lkl_pro_yaml, prof_param, prof_incr, yaml_ext = '_lkl_prof'):
    # update profile lkl param 
    latex_info = lkl_pro_yaml['params'][prof_param]['latex']
    lkl_pro_yaml['params'][prof_param] = {'value': MLs[prof_param]+prof_incr, 'latex': latex_info}
    lkl_pro_yaml['output'] = chain_file + yaml_ext
    # update all other independent parameters 
    for param in lkl_pro_yaml['params']:
        if 'prior' in lkl_pro_yaml['params'][param]:
            lkl_pro_yaml['params'][param]['ref'] = MLs[param]
    # dump yaml to file for running 
    with open(chains_dir+chain_file+yaml_ext+'.minimize.input.yaml', 'w') as yaml_file:
        dump(lkl_pro_yaml, yaml_file, default_flow_style=False)    
    return lkl_pro_yaml['params'][prof_param]


# ### Run minimizer 

# In[ ]:


def run_minimizer(chain_file, yaml_ext='_lkl_prof', debug=False, processes=4):
    """
    For the parameter we want to vary, remove all but latex and value. 
    The latex is as before from the MCMC yaml file. 
    The value is ML $\pm$ increment. 
    """
    if debug==True:
        run("mpirun -np "+str(processes)+" cobaya-run "+chain_file+yaml_ext+".minimize.input.yaml -f -d", shell=True)
    else:
        run("mpirun -np "+str(processes)+" cobaya-run "+chain_file+yaml_ext+".minimize.input.yaml -f", shell=True)   
    return True


# ### Check if minimizer was run

# In[ ]:


def check_global_min(mcmc_chains, chains_dir, chain_file):
    try:
        mcmc_chains.getParamBestFitDict()
        min_yaml = yaml_load_file(chains_dir+chain_file+'.minimize.updated.yaml')
        print("check_global_min: Found previously run MCMC chains and global minimizer. ")
        return True
    except MCSamplesError:
        print("check_global_min: Need to first run a minimizer on the full MCMC chains before beginning 1d profile lkl code.")
        return False 
    except FileNotFoundError:
        print("check_global_min: Found best-fit but not the file "+chains_dir+chain_file+".minimize.updated.yaml. Something has gone wrong. ")
        return FileNotFoundError


# # Overarching run_prof_lkl function

# In[ ]:


def run_prof_lkl(chains_dir, chain_file, mcmc_chains, processes  ):
    ### Check if minimizer was run on all chains previously 
        # And if not, use input minimizer settings to run it. 
        # Ideally, TBC, get number of chains from mcmc and run that many minimizer processes. 

    if not check_global_min(mcmc_chains=mcmc_chains, chains_dir=chains_dir, chain_file=chain_file):
        mcmc_yaml = yaml_load_file(chains_dir+chain_file+'.input.yaml')
        mcmc_yaml['sampler'] = minimizer_settings
        min_yaml = deepcopy(mcmc_yaml)
        with open(chains_dir + chain_file + '.minimize.input.yaml', 'w') as yaml_file:
            dump(min_yaml, yaml_file, default_flow_style=False)

        run_minimizer(chain_file=chain_file, yaml_ext='', debug=False, processes=processes)  

        
    ### Add minimized point to lkl profile text file 
        # So: 
        # 1) load the global minimum file. 
        # 2) check if we have a file with prof lkl values. 
        #     * If yes, check that it has the same parameters and in the right order. Proceed. 
        #     * If no file, start it and write the first line as param names. Proceed. 
        #     * If file yes, but parameters don't match, then print an error. Stop. 
        # 2) check if global minimum params have already been written (first line of file)
        #     * If parameters are written, check that they match global minimum. Don't write them again
        #     * If parameters are written but don't match, spit out error. 
        #     * If no params written, add this current ML values for all parameters in append mode
        
    param_names, param_ML, MLs = read_minimum(chains_dir, chain_file, extension='')

    global_ML = deepcopy(MLs)
    param_order = param_names

    try: 
        if not match_param_names(param_names, chains_dir=chains_dir, chain_file=chain_file):
            raise FileExistsError
    except FileNotFoundError:
        print("File not found. Starting a new file now: " + chains_dir + chain_file + '_lkl_profile.txt \n')
        with open(chains_dir + chain_file + '_lkl_profile.txt', 'w') as lkl_txt:
            lkl_txt.write("#")
            for param_recorded in param_names:
                lkl_txt.write("\t %s" % param_recorded)
            lkl_txt.write("\n")

    lkl_prof_table = np.loadtxt(chains_dir + chain_file + '_lkl_profile.txt')
    if lkl_prof_table.shape!=(0,):
        if not match_param_line(param_names, global_ML, chains_dir=chains_dir, chain_file=chain_file, loc=0):
            print("Something went wrong. The first line of the lkl_profile.txt file which should be global ML does not match the global ML in file \n"
                 +chains_dir + chain_file + '.minimum')
            raise FileExistsError
    else: 
        write_MLs(param_names, MLs, chains_dir=chains_dir, chain_file=chain_file)


    ## Likelihood profile 
        ### Set up lkl profile minimum input yaml file 
            # Only this should be read and manipulated by the rest of the code

            # Here I check that a "_lkl_prof" file has been created for the minimizer input yaml.

    try:
        lkl_pro_yaml = yaml_load_file(chains_dir+chain_file+'_lkl_prof.minimize.input.yaml')
    except FileNotFoundError:
        run("cp "+chains_dir+chain_file+'.minimize.updated.yaml'+" "+chains_dir+chain_file+'_lkl_prof.minimize.input.yaml', shell=True)
        lkl_pro_yaml = yaml_load_file(chains_dir+chain_file+'_lkl_prof.minimize.input.yaml')

            # We already have param_names and MLs saved. Update the param_ML with the values from the last entry in _lkl_prof.txt in case we're restarting / continuing a run. 

    param_ML = read_lkl_output(chains_dir=chains_dir, chain_file=chain_file, loc=-1)
    MLs = dict(zip(param_names, param_ML))


    ## Run loop over increments of profile lkl param 
        # While we are within the bounds of the profile param we want to explore: 

        # 1) check if the point we are currently at i.e. param_ML and MLs, matches the last entry in the lkl_prof table.
        #     - if it does, the last minimum was run and saved successfully. 
        #     - if not, check if a minimum file exists. 
        #         - if it does, read it in and save it in the lkl prof txt. minimum run successfully. 
        #         - if not, this happens when we have updated the yaml but the minimizer didn't finish. Run the yaml again without updating. 
        # 2) check if minimum was run and saved. 
        #     - if yes, update the yaml and increment the prof lkl param, update all other params to new values from current ML. Assign the MLs values for the independent params in the yaml as new reference starting points. 
        # 3) run the minimizer 
        # 4) save minimizer output 
    
    while MLs[prof_param] < prof_max:
        last_entry_matches_current_params = match_param_line(param_names, MLs, chains_dir=chains_dir, chain_file=chain_file, loc=-1)
        if last_entry_matches_current_params:
            run('rm '+chains_dir + chain_file + '_lkl_prof.minimum*', shell=True)
            minimum_successfully_run_and_saved = True
        else:
            try:
                param_names, param_ML, MLs = read_minimum(chains_dir, chain_file)
                write_MLs(param_order, MLs, chains_dir=chains_dir, chain_file=chain_file)
                run('rm '+chains_dir + chain_file + '_lkl_prof.minimum*', shell=True)
                minimum_successfully_run_and_saved = True 
                print("-----> Minimizer run successfully for "+prof_param+" = "+str(MLs[prof_param]))
            except OSError:
                minimum_successfully_run_and_saved = False
                print("-----> Minimizer not run for "+prof_param+" = "+str(MLs[prof_param]))
                print("       Rerunning this point")

        if minimum_successfully_run_and_saved:
            increment_update_yaml(chains_dir, chain_file, MLs, lkl_pro_yaml, prof_param, prof_incr)
            run('rm '+chains_dir + chain_file + '_lkl_prof.minimize.updated.yaml', shell=True)

        run_minimizer(chain_file=chain_file, yaml_ext='_lkl_prof', debug=False, processes=6)

        param_names, param_ML, MLs = read_minimum(chains_dir, chain_file)
        
    param_names, param_ML, MLs = read_minimum(chains_dir, chain_file)
    write_MLs(param_order, MLs, chains_dir=chains_dir, chain_file=chain_file)

