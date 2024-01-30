import os
from copy import deepcopy
from glob import glob
from subprocess import run
from time import time
import re

import numpy as np
from getdist import mcsamples

import procoli.procoli_io as pio 
from procoli.procoli_errors import (ParamDifferenceError, GlobalMLDifferenceError, 
                                    LogParamUpdateError, ExperimentNotFoundError)


class lkl_prof:
    """
    Class for profiling likelihoods from MontePython MCMC chains.

    Parameters:
    - chains_dir (str): Directory containing MCMC chains and log.param, OR containing .covmat, .bestfit and log.param files.
    - prof_param (str): The parameter for which the likelihood will be profiled as recognised by MontePython in the log.param.
    - prof_incr (float, optional): Increment for profiling prof_param. 
    - prof_min (float, optional): Minimum value of prof_param for profiling. 
    - prof_max (float, optional): Maximum value of prof_param for profiling. 
    - info_root (str, optional): Information root for the chains directory. Defaults to the last part of the chains directory path. 
      Provide this if your .covmat and .bestit files have a different filename than the name of the parent directory. 
    - processes (int, optional): Number of parallel processes to use. Defaults to 5.
    - R_minus_1_wanted (float, optional): The target R-1 value for chains. Defaults to 0.05. If this R-1 is not attained, 
      the code prints a warning, but continues anyway. 
    - mcmc_chain_settings (dict, optional): Settings for MCMC chains as understood by GetDist mcsamples.loadMCSamples. 
      Defaults to {'ignore_rows': 0.3}.
    - jump_fac (list, optional): List of jump factors for profile-likelihood simulated-annealing temperature ladder. 
      Defaults to [0.15, 0.1, 0.05].
    - temp (list, optional): List of temperature values for profile-likelihood simulated-annealing temperature ladder. 
      Defaults to [0.1, 0.005, 0.001].
    - global_jump_fac (list, optional): List of jump factors for global minimum temperature ladder. 
      Defaults to [1, 0.8, 0.5, 0.2, 0.1, 0.05].
    - global_min_temp (list, optional): List of minimum temperatures for global minimum temperature ladder. 
      Defaults to [0.3333, 0.25, 0.2, 0.1, 0.005, 0.001].

    Attributes:
    - chains_dir (str): Directory containing MCMC chains.
    - info_root (str): Information root for the chains directory.
    - processes (int): Number of parallel processes to use.
    - R_minus_1_wanted (float): The target R-1 value.
    - mcmc_chain_settings (dict): Settings for MCMC chains.
    - mcmc_chains (None or list): Placeholder for storing MCMC chains.
    - prof_param (str): The parameter for which the likelihood will be profiled.
    - prof_incr (float or None): Increment for profiling.
    - prof_min (float or None): Minimum value for profiling.
    - prof_max (float or None): Maximum value for profiling.
    - jump_fac (list): List of jump factors for local temperature ladder.
    - temp (list): List of temperature values for local temperature ladder.
    - global_jump_fac (list): List of jump factors for global temperature ladder.
    - global_min_temp (list): List of minimum temperatures for global temperature ladder.
    - covmat_file (str): Full path to the covariance matrix file.

    Example:
    ```
    profiler = lkl_prof(chains_dir='path/to/chains', prof_param='theta', processes=4)
    ```

    """
    
    def __init__(self, chains_dir, prof_param, info_root=None, processes=5, 
                 R_minus_1_wanted=0.05, mcmc_chain_settings={'ignore_rows' : 0.3}, 
                 prof_incr=None, prof_min=None, prof_max=None, 
                 jump_fac=[0.15, 0.1, 0.05], temp=[0.1, 0.005, 0.001], 
                 global_jump_fac=[1, 0.8, 0.5, 0.2, 0.1, 0.05], 
                 global_min_temp=[0.3333, 0.25, 0.2, 0.1, 0.005, 0.001]
                ):
        
        self.chains_dir = chains_dir
        chains_full_path = os.path.abspath(self.chains_dir)
        if info_root is None: 
            info_root = [x for x in chains_full_path.split('/') if x][-1]
        self.info_root = info_root 
        
        self.processes = processes
        self.R_minus_1_wanted = R_minus_1_wanted
        self.mcmc_chain_settings = mcmc_chain_settings
        self.mcmc_chains = None
        
        self.prof_param = prof_param
        self.prof_incr = prof_incr
        self.prof_min = prof_min
        self.prof_max = prof_max
        
        self.jump_fac = jump_fac
        self.temp = temp
        
        self.global_min_jump_fac = global_jump_fac
        self.global_min_temp = global_min_temp
        
        self.covmat_file = f'{self.chains_dir}{self.info_root}.covmat'

    def set_jump_fac(self, jump_fac):
        """
        Setter function for the jump factor for the likelihood profile

        :jump_fac: A list of jump factors

        :return: Nothing
        """

        self.jump_fac = jump_fac

    def set_temp(self, temp):
        """
        Setter function for the jump factor for the likelihood profile

        :temp: A list of temperatures

        :return: Nothing
        """

        self.temp = temp

    def set_global_jump_fac(self, global_jump_fac):
        """
        Setter function for the jump factor for the global mimimum

        :global_jump_fac: A list of jump factors

        :return: Nothing
        """

        self.global_min_jump_fac = global_jump_fac

    def set_global_temp(self, global_min_temp):
        """
        Setter function for the jump factor for the global mimimum

        :global_min_temp: A list of temperatures

        :return: Nothing
        """

        self.global_min_temp = global_min_temp
    
    def check_mcmc_chains(self, read_all_chains=False):
        """
        Check if mcmc chains chains exist. 
        
        If read_all_chains = False
            This explicitly uses the longest chain root in the folder. 
            That is, if the self.chains_dir contains files with the roots:
            1993-10-05_500_
            1993-10-05_5000000_
            1991-08-15_1000000_
            The code will pick out the longest chain root name, so 1993-10-05_5000000_
        
        If read_all_chains = True
            This sets up an MCMCSamples instance using the longest chain
            It then replaces the chains in that instance with all the chains in the 
                folder 
            No duplication of chains occurs. 
            
        :read_all_chains: boolean for whether to read all the chains in the chains 
            directory 
        
        :return: True if files found, else False 
        """
        max_steps_in_chain = str( max( [ int(i[len(self.chains_dir)+11:-7]) for i in 
                                        glob(f'{self.chains_dir}*__1.txt') ] ) )
        for file_root in glob(f'{self.chains_dir}*__1.txt'):
            if max_steps_in_chain in file_root:
                self.chain_root = file_root[len(self.chains_dir):-6]
        print('check_mcmc_chains: Looking for files: '\
              f'{self.chains_dir}{self.chain_root}')

        try:
            self.mcmc_chains = mcsamples.loadMCSamples(self.chains_dir+self.chain_root, 
                                                       settings=self.mcmc_chain_settings)
            self.covmat_file = self.chains_dir+self.info_root+'.covmat'
        except OSError:
            return False 
        
        if read_all_chains is True:
            chain_root_list = glob(f'{self.chains_dir}*__*.txt')
            print("check_mcmc_chains: Reading all chains:")
            for chain_x in chain_root_list:
                print(chain_x)
            try:
                self.mcmc_chains.readChains(chain_root_list)
            except OSError:
                return False 
        
        return True
        
    def run_mcmc(self, N_steps=30000):
        """
        Run MCMC chains 
        Requires the folder chains_dir to already be popualted with a log.param file 
        :N_steps: number of steps to take for the MCMC 
        
        :return: True if files found, else False 
        """
        
        with open(f'{self.chains_dir}log.param', 'r'):
            pass
        
        try:
            with open(f'{self.chains_dir}{self.info_roo}.bestfit', 'r'):
                        pass
            bf_exists = True    
        except FileNotFoundError:
            bf_exists = False
        try:
            with open(f'{self.chains_dir}{self.info_root}.covmat', 'r'):
                        pass
            covmat_exists = True    
        except FileNotFoundError:
            covmat_exists = False

        if (bf_exists and covmat_exists):
            run_command = 'mpirun -np {procs} MontePython.py run -p {param} '\
                '-o {output} -b {bf} -c {covmat} -N {steps} '\
                '--update 50 --superupdate 20'.format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                bf=self.chains_dir+self.info_root+'.bestfit', 
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=N_steps
            )
        elif bf_exists:
            run_command = 'mpirun -np {procs} MontePython.py run -p {param} '\
                '-o {output} -b {bf} -N {steps} --update 50 --superupdate 20'.format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                bf=self.chains_dir+self.info_root+'.bestfit', 
                steps=N_steps
            )
        elif covmat_exists:
            run_command = 'mpirun -np {procs} MontePython.py run -p {param} '\
                '-o {output} -c {covmat} -N {steps} --update 50 --superupdate 20'.format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=N_steps
            )
        else:
            run_command = 'mpirun -np {procs} MontePython.py run -p {param} '\
                '-o {output} -N {steps} --update 50 --superupdate 20'.format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                steps=N_steps
            )
            
        run(run_command, shell=True)
            
        return True
        
    def check_mcmc_convergence(self, mcmc_chains=None):
        """
        Check if MCMC converged 
        
        :mcmc_chains: getdist MCSamples instance 
        
        :return: True if MCMC chains have converged to the desired R-1, 
            default is R-1=0.05. Else False 
        """
        if mcmc_chains is None:
            mcmc_chains=self.mcmc_chains
            
        current_R_minus_1 = mcmc_chains.getGelmanRubin()
        if current_R_minus_1 < self.R_minus_1_wanted:
            print("check_mcmc_convergence: Chains converged sufficiently. '\
                  'Current R-1 = {:.3f} satisfies R-1 wanted = {:.3f}. '\
                  '\nMove on to checking minimum.".format(current_R_minus_1, 
                                                         self.R_minus_1_wanted))
            return True
        else: 
            print("check_mcmc_convergence: Chains not converged. '\
                  'Current R-1 = {:.3f} while R-1 wanted = {:.3f}. '\
                  '\nResume MCMC. ".format(current_R_minus_1,self.R_minus_1_wanted))
            return False 
    
    def mcmc(self):
        """
        Check MCMC and run if needed 
        
        /!\ THIS FUNCTION DOESN'T ACTUALLY LEAD TO CONVERGENCE. NEEDS IMPROVEMENT. 
        
        :return: True once finished 
        """
        if not self.check_mcmc_chains(read_all_chains=True):
            self.run_mcmc()
        while not self.check_mcmc_convergence():
            run(f'mpirun -np 1 MontePython.py info {self.chains_dir} '\
                '--keep-non-markovian --noplot --want-covmat --minimal', shell=True)
            self.run_mcmc(N_steps=50000)
            self.check_mcmc_chains(read_all_chains=True)
        return True

    def check_global_min_has_lower_loglike(self, existing_min):
        """
        Check the negative log likelihoods of the global minimum bestfit and the 
            info root bestfit to see which is better (lower negative log likelihood)
        
        :existing_min: True if global minimum was run and relevant files are accesible. 
            Else False
        
        :return: If a global bestfit already exists with a lower 
            log likehood than the info root bestfit
        """

        global_path = 'global_min/global_min'
        global_min_is_better = False
        if existing_min:
            if os.path.exists(f'{self.chains_dir}{global_path}.bestfit'):
                global_min_point = pio.get_MP_bf_dict(f'{self.chains_dir}{global_path}.bestfit')
                info_root_point = pio.get_MP_bf_dict(f'{self.chains_dir}{self.info_root}.bestfit')
                
                if info_root_point['-logLike'] != global_min_point['-logLike']:
                    print('check_global_min: WARNING!!!: global_min folder found with '\
                            'a global_min.bestfit that is different from '\
                            f'{self.info_root}.bestfit. Code will use the better '\
                            'chi^2 of the two going forward.')

                if info_root_point['-logLike'] >= global_min_point['-logLike']:
                    _ = pio.file_copy(f'{self.chains_dir}{global_path}.bestfit', 
                                        f'{self.chains_dir}{self.info_root}.bestfit')
                    global_min_is_better = True
                    print(f'check_global_min: WARNING!!!: global_min folder found '\
                            'with a global_min.bestfit that was found to be as good '\
                            f'or a better chi^2 than the {self.info_root}.bestfit file. '\
                            f'Code will replace the {self.info_root}.bestfit and '\
                            f'{self.info_root}.log files with ones from the '\
                            'global_min/global_min.bestfit and .log going forward.')
                    
        return global_min_is_better


    def check_global_min(self, mcmc_chains=None):
        """
        Check for .bestfit file. This does not necessarily indicate a global 
            minimum run!!! 
        It only indicates that there exists a file storing some bf in the 
            'info_root' file. 
        This also resets the info_root to the current directory name to avoid 
            errors later in the code. 
        
        :mcmc_chains: getdist MCSamples instance 
        
        :return: True if global minimum was run and relevant files are accesible. 
            Else False, If a global bestfit already exists with a lower 
            log likehood than the info root bestfit
        """
        
        if mcmc_chains is None:
            mcmc_chains=self.mcmc_chains

        global_min_exists = False
        existing_min = False
            
        try:
            # TODO can probably check if it exists with the os module
            pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}.bestfit')
            print(f'check_global_min: Found minimum with file name {self.info_root}')
            pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}.covmat')
            print(f'check_global_min: Found covmat with file name {self.info_root}')
            
            new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
            if self.info_root != new_info_root:
                _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.bestfit', 
                                  f'{self.chains_dir}{new_info_root}.bestfit')
                _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.covmat', 
                                  f'{self.chains_dir}{new_info_root}.covmat')
                try:
                    _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.log', 
                                      f'{self.chains_dir}{new_info_root}.log')
                except FileNotFoundError:
                    self.make_log_file( bf_file=f'{self.chains_dir}{self.info_root}.bestfit',
                                        output_loc=self.chains_dir
                                      )
                    _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.log', 
                                      f'{self.chains_dir}{new_info_root}.log')
                self.info_root = new_info_root
            else:
                if not os.path.exists(f'{self.chains_dir}{self.info_root}.log'):
                    self.make_log_file( bf_file=f'{self.chains_dir}{self.info_root}.bestfit',
                                        output_loc=self.chains_dir
                                      )
                
            global_min_exists = True
            existing_min = True
        except OSError:
            try:
                new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
                # TODO can we run montepython with mpirun directly from python?
                run(f'mpirun -np 1 MontePython.py info {self.chains_dir} '\
                    '--keep-non-markovian --noplot --want-covmat --minimal', 
                    shell=True, check=True)
                # TODO can probably check if it exists with the module
                pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}.bestfit')
                # TODO why change the info root?
                self.info_root = new_info_root
                print('check_global_min: Found minimum with file name '\
                      f'{self.info_root}')
                global_min_exists = True 
            except OSError:
                print('check_global_min: Cannot run MP info for global minimum. '\
                      'Something went wrong. Either provide chains or provide .bestfit and .covmat file.')
                global_min_exists = False 

        global_min_is_better = self.check_global_min_has_lower_loglike(existing_min)

        return global_min_exists, global_min_is_better
        
    def global_min(self, run_glob_min=None, N_min_steps=4000, run_minuit=False):
        """
        Check global minizer, run if wanted (default False), then write if not 
            already written 

        So: 
        1) Load / create the global minimum file. 
        2) Check if the previous global minimum bestfit is 
            better than the info_root bestfit
        3) If we want a global min run, run the minimizer 
        4) grab the global minimizer results 
        5) check if we have a file with prof lkl values. 
            * If yes, check that it has the same parameters and in the right order. 
                Proceed. 
            * If no file, start it and write the first line as param names. Proceed. 
            * If file yes, but parameters don't match, then print an error. Stop. 
        6) check if global minimum params have already been written (first line of file)
            * If parameters are written, check that they match global minimum. 
                Don't write them again
            * If parameters are written but don't match, spit out error. 
            * If no params written, add this current ML values for all parameters 
                in append mode
            
        :run_glob_min: Boolean for whether to run a global minimizer.  
            If True or False are given then choose to run the minimzer accordingly
            If no value is given then let check_global_min decide by checking
                if the global min has already been run and has a better bestfit
        :N_min_steps: The number of steps the minimizer should use for each run
        :run_minuit: Flag for the minimizer to use minuit

        :return: global maximum lkl dictionary 
        """

        # check to see if the global min exists already 
        # and decide to run the minimizer accordingly
        global_min_exists, global_min_is_better = self.check_global_min()
        if run_glob_min is None:
            if global_min_is_better:
                run_glob_min = False
            else:
                run_glob_min = True

        if run_glob_min:
            pio.make_path(f'{self.chains_dir}global_min', exist_ok=True)
            _ = pio.file_copy(f'{self.chains_dir}log.param', 
                              f'{self.chains_dir}global_min/log.param')
            
            self.run_minimizer(min_folder='global_min', N_steps=N_min_steps, 
                               run_minuit=run_minuit, 
                               jump_fac=self.global_min_jump_fac, 
                               temp=self.global_min_temp)

            _ = pio.file_copy(f'{self.chains_dir}global_min/global_min.bestfit', 
                              f'{self.chains_dir}{self.info_root}.bestfit')
            _ = pio.file_copy(f'{self.chains_dir}global_min/global_min.log', 
                              f'{self.chains_dir}{self.info_root}.log')

        param_names, param_ML, MLs = self.read_minimum(extension='')
        # Additional code to get chi2 per experiment 
        MLs_and_chi2 = self.update_MLs_chi2_per_exp(MLs)
        self.param_order = [key for key in MLs_and_chi2]
        
        self.global_ML = deepcopy(MLs_and_chi2)
        # self.param_order = param_names.tolist()

        extension = '_lkl_profile.txt' 
        extension = self.pn_ext(extension)
        
        try:
            self.match_param_names(self.param_order)
        except FileNotFoundError:
            print('global_min: File not found. Starting a new file now: '\
                  f'{self.chains_dir}{self.info_root}{extension}\n') 
            with open(f'{self.chains_dir}{self.info_root}{extension}', 'w') as lkl_txt: 
                lkl_txt.write('#')
                for param_recorded in self.param_order:
                    lkl_txt.write(f'\t {param_recorded}')
                lkl_txt.write("\n")

        lkl_prof_table = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{extension}') 

        # TODO param order should inherit from file header, param order not matching 
        #   should never cause the code to fail
        if lkl_prof_table.shape!=(0,):
            if not self.match_param_line(self.global_ML, loc=0):
                raise GlobalMLDifferenceError(f'{self.chains_dir}{self.info_root}')
        else: 
            self.write_MLs(MLs_and_chi2)

        return self.global_ML
        
        
    def pn_ext(self, extension):
        """
        Prefix the file extension string input with 
        the sign of the profile lkl parameter, 
        and its name to track files correctly. 
        
        :extension: A string of the file name extension, eg. "_good_pupper"
        :return: String of extension prefixed with the sign and name of the 
            profile lkl parameter "_+height_good_pupper"
        """
        if len(extension)>0:
            if self.prof_incr > 0:
                extension = '_+'+self.prof_param+extension
            if self.prof_incr < 0:
                extension = '_-'+self.prof_param+extension
        return extension
        
    def read_minimum(self, extension='_lkl_prof'):
        """
        Read minimum file and save parameter names list, parameter values list 
            and MLs dictionary 
        Also update the dictionary object self.MLs 
        
        :extension: The extension of the life type being read in. Leave this as is, the 
            rest of the code assumes the same naming conventions. Otherwise, specify to 
            read a specific file, but know that this will update the self.MLs dict too. 
        
        :return: List of parameter names, list of parameter ML values, 
            dictionary of {'param_names': param_ML_value}
        """

        prefix_extension = self.pn_ext(extension)
        
        # TODO can probably make this a single read to dict
        param_ML = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{prefix_extension}.bestfit')
        param_names = pio.read_header_as_list(f'{self.chains_dir}{self.info_root}{prefix_extension}.bestfit')

        MLs = dict(zip(param_names, param_ML))

        with open(f'{self.chains_dir}{self.info_root}{prefix_extension}.log') as log_file:
            last_line = log_file.readlines()[-1]
            neg_logLike = float(last_line.split(":")[-1])

        MLs['-logLike'] = neg_logLike
        param_names = np.append(param_names, '-logLike')
        param_ML = np.append(param_ML, MLs['-logLike'])
        
        self.MLs = MLs
        
        # TODO do we want to remove param_ML from the output?  
        #   It's never used as an output
        return param_names, param_ML, MLs
    
    def read_lkl_output(self, extension='_lkl_profile.txt', loc=-1):
        """
        Read (default = last) line of lkl prof output file into list
        
        :extension: Leave this alone, thank you. 
        :loc: integer location of line in file to read. Default is last line 
        
        :return: Dict of parameters
        """

        prefix_extension = self.pn_ext(extension)

        lkl_prof_table = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{prefix_extension}') 
        try:
            lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
            lkl_prof_table = lkl_prof_table[loc, :]
        except IndexError:
            pass
        
        self.param_names = pio.read_header_as_list(f'{self.chains_dir}{self.info_root}{prefix_extension}')
        
        MLs = dict(zip(self.param_names, lkl_prof_table))
        
        return MLs
    
    def write_MLs(self, MLs=None, extension='_lkl_profile.txt'):
        """
        Write params from MLs dict into txt file in append mode
        Note that to write, we use self.param_order, not self.param_names. 
        This is because the global param_names list is the one that has the 
            correct order. 
        
        :extension: Leave it alone, thank you.
        
        :return: new length of the saved lkl profile table
        
        """
        if MLs is None:
            MLs = self.MLs
        prefix_extension = self.pn_ext(extension)
        
        with open(f'{self.chains_dir}{self.info_root}{prefix_extension}', 'a') as lkl_txt: 
            for param in self.param_order:
                lkl_txt.write(f'\t {str(MLs[param])}')
            lkl_txt.write('\n')
        lkl_prof_table = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{prefix_extension}') 
        return lkl_prof_table.shape
    
    
    def match_param_names(self, param_names, extension='_lkl_profile.txt'):
        """
        Check that param names match in target file and MLs dictionary
        Matches combination, not permutation. 
        
        :param_names: List of param_names to check against the file 
        :extension: Leave it alone, thank you. 
        
        :return: True if param_names match, else False
        """
        
        prefix_extension = self.pn_ext(extension)
        
        params_recorded = pio.read_header_as_list(f'{self.chains_dir}{self.info_root}{prefix_extension}')
        
        # TODO this can probably be combined so you don't have to compare them 
        #   against each other twice
        mismatched_params = [param for param in param_names 
                             if param not in params_recorded]
        if not mismatched_params:
            param_names_in_rec = True
        else: 
            print("\nmatch_param_names: Params don't match. \nPassed param_names has "\
                  "the following params that are not expected by params recorded:")
            print(mismatched_params)
            param_names_in_rec = False

        mismatched_params = [param for param in params_recorded 
                             if param not in param_names]
        if not mismatched_params:
            rec_params_in_param_names = True
        else: 
            print("\nmatch_param_names: Params don't match. \nRecorded params expected "\
                  "that are absent in passed param_names: ")
            print(mismatched_params)
            rec_params_in_param_names = False

        if (param_names_in_rec and rec_params_in_param_names):
            print("match_param_names: Params match - the recorded params contain the "\
                  "same params as param_names passed. ")
            self.param_order = params_recorded
            return True
        else:
            raise ParamDifferenceError(f'{self.chains_dir}{self.info_root}{prefix_extension}')
    
    
    def match_param_line(self, MLs, param_names=None, extension='_lkl_profile.txt', 
                         loc=-1):
        """
        Check if specified (default: last) location in lkl_prof output file matches 
            current MLs

        :param_names: list of parameter names in the same order as that printed in 
            the file. 
            This is usually the global param_order list. Note: LIST not array! 
        :MLs: dictionary of {'param_name': ML_value }
        :extension: Leave it alone, thank you. 
        :loc: integer location of row in file to check, default is the last line

        :return: True if match, else False 
        """

        prefix_extension = self.pn_ext(extension)

        if param_names is None:
            # param_names=self.param_order
            try:
                param_names = [i for i in self.param_order if i not in self.likelihoods]
                param_names.remove('Total')
            except AttributeError:
                pass
            except ValueError:
                pass

        # print('match_param_line: checking file '\
        #       f'{self.chains_dir}{self.info_root}{prefix_extension}')
        lkl_prof_table = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{prefix_extension}') 
        
        if lkl_prof_table.size==0:
            print("match_param_line: File empty ")
            return False
        else: 
            try:
                lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
                if False in [ lkl_prof_table[loc, param_names.index(param)]
                              == MLs[param] for param in param_names ]:
                    return False
                else:
                    return True 
            except IndexError:
                print("match_param_line: Only one entry in file, checking that entry ")
                if False in [ lkl_prof_table[param_names.index(param)] 
                             == MLs[param] for param in param_names ]:
                    return False 
                else:
                    return True
                
    def update_neg_log_likelhood_chains(self, path, temp, chain_length, 
                                        previous_chains=[]):
        """
        Update the chain files to show the correct negative log likelihood
            when using the temperature parameter
        Ignore chain files that have already been updated

        :path: Path to the directory with the chain files
        :temp: Float of the temperature parameter used 
        :chain_length: Int of how many points are in each chain file 
        :previous_chains: List of the previous chain files that have been updated
            The dfault is an empty list

        :return: All chain files that have been updated previously, 
            or that needed no updates if temp == 1 for that step. 
        """

        chains = glob(f'{path}*_{chain_length}__*.txt')
        if temp != 1:
            for chain in chains:
                if chain not in previous_chains:

                    # read in the current chain, update the negative log likelihood
                    # and then save the new data back to the original file
                    chain_file = pio.load_mp_info_files(chain)
                    row_size = chain_file.shape[1]
                    frmt_list = ['%.4g', '%.6g'] + ['%.6e']*(row_size-2)
                    chain_file[:,1] = chain_file[:,1]*temp
                    pio.save_mp_info_files(chain, chain_file, fmt=frmt_list, 
                                           delimiter='\t')
                    
        return chains
    
    def run_minimizer(self, min_folder="lkl_prof", prev_bf=None, N_steps=3000, 
                      run_minuit=False, jump_fac=None, temp=None):
        """
        Run minimizer as described in 2107.10291, by incrementally running a finer MCMC 
        with a more discrening lklfactor that increases preference for moving towards 
            higher likelihoods 
        and taking smaller jumps between points so we sample a finer grid in parameter 
            space. 
        This can be modified as wanted, changing step sizes per rung, lkl factor and 
            jumping factors. 
        You would need to change this function, then import the class and run a lkl 
            prof as usual. 
        Default:
        N_steps = input parameter, same for all rungs
        lklfactor = 10, 200, 1000
        jumping factor (-f) = 0.5, 0.1, 0.05

        Requires the folder min_folder to already be popualted with a log.param file 

        :min_folder: Folder to run minimizer in 
        :prev_bf: Starting best-fit file. 
                This is the MAP of the MCMC chains for the global bf run, 
                and the previous point for the lkl prof. 
                The function init_lkl_prof takes care of designating this, 
                set interanlly. 
        :N_steps: Number of steps each minimizer rung takes, same for all rungs 

        :return: True
        """
        
        if min_folder=="lkl_prof":
            min_folder += self.pn_ext('/')

        # Prep the command 
        if not min_folder:
            min_folder = '.'
        elif min_folder[-1] == '/':
            min_folder = min_folder[:-1]

        if not prev_bf:
            prev_bf = self.info_root
        elif '.bestfit' in prev_bf:
            prev_bf = prev_bf[:-8]

        # TODO clean this up
        # Check if jumping factors and lkl factors are defined, otherwise set to 
        # lkl profile run defaults 
        if jump_fac is None:
            jump_fac = self.jump_fac
        if temp is None:
            temp = self.temp
        if len(jump_fac) != len(temp):
            jump_fac = [0.15, 0.1, 0.05] 
            # TODO: DEFAULTS HARD CODED HERE. 
            # Move to init as defaults that the user should not know about 
            # and cannot change. 
            temp = [0.1, 0.005, 0.001]
            print('!!!!!!!!!\n!!!!!!!!!\n!!!!!!!!!')
            print('Error in run_minimizer: Lists passed for jumping factor and lkl '\
            'factor are of different lengths. \n'\
            'Setting to defaults!!! \n'\
            f'jumping factor list = {jump_fac} \n'\
            f'temperature list = {temp}')
            print("!!!!!!!!!\n!!!!!!!!!\n!!!!!!!!!")

        ##### First rung #####

        # MCMC
        mp_run_command = 'mpirun -np {procs} MontePython.py run -p {param} '\
            '-o {output} -b {bf} -c {covmat} -N {steps} -f {f} '\
            '-T {temp}'.format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=N_steps, 
            f = jump_fac[0], 
            temp = temp[0]
        )
        previous_chains = glob(f'{self.chains_dir}{min_folder}/*_{N_steps}__*.txt')
        run(mp_run_command, shell=True, check=True)
        previous_chains = self.update_neg_log_likelhood_chains(f'{self.chains_dir}{min_folder}/', 
                                             temp[0], N_steps, previous_chains=previous_chains)
        # analyse 
        mp_info_command = 'mpirun -np 1 MontePython.py info {folder} '\
            '--keep-non-markovian --noplot --minimal'.format(
            folder=self.chains_dir+min_folder+'/'
        )
        run(mp_info_command, shell=True)
        # print output 
        if min_folder=='.':
            prev_bf = [x for x in str(os.getcwd()).split('/') if x][-1]
            # switch to current directory as bf root, 
            #   ensures that we're using the most recent file 
        else:
            prev_bf = min_folder+'/'+min_folder 
            # switch to most recently produced bf file 
            #   in the minimizer directory as bf root 
        # set new minimum 
        new_min_point = pio.get_MP_bf_dict(f'{self.chains_dir}{prev_bf}.bestfit')
        print('\n\n------------------> After minimizer rung 1, -logL minimized to '\
              '{logL} \n\n'.format(logL=new_min_point['-logLike']))


        ##### Loop over other rungs #####

        num_itrs = len(jump_fac)

        for i in range(1,num_itrs):
            # MCMC
            run_command = 'mpirun -np {procs} MontePython.py run -p {param} '\
                    '-o {output} -b {bf} -c {covmat} -N {steps} -f {f} '\
                    '-T {temp}'.format(
                procs=self.processes,
                param=self.chains_dir+min_folder+'/log.param', 
                output=self.chains_dir+min_folder+'/',
                bf=self.chains_dir+prev_bf+'.bestfit', 
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=N_steps, 
                f = jump_fac[i], 
                temp = temp[i]
            )
            run(run_command, shell=True)
            previous_chains = self.update_neg_log_likelhood_chains(f'{self.chains_dir}{min_folder}/', 
                                             temp[i], N_steps, previous_chains=previous_chains)
            # analyse 
            run_command = 'mpirun -np 1 MontePython.py info {folder} '\
                '--keep-non-markovian --noplot --minimal'.format(
                folder=self.chains_dir+min_folder+'/'
            )
            run(run_command, shell=True)
            # set new minimum 
            new_min_point = pio.get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
            print('\n\n------------------> After minimizer rung {ith}, '\
                  '-logL minimized to  {logL} \n\n'.format(
                ith=i+1, logL=new_min_point['-logLike']))
            
        ##### Bonus step: run Minuit minimizer #####
        
        if run_minuit:
            # NOTE: this will only work well if the minimizer outputs results that 
            #   have correctly scaled params 
            # MP rescales some params: omega_b (1e-2), A_s (1e-9) 
            #   and a couple of PLC params 
            # So this output needs to scale them back to normal, 
            #   with omega_b of O(0.01), etc. 
            # TK will update MP to do this correctly. 
            #   Fix already there, need to turn print into replacement 

            # Run minuit minimizer 
            run_command = 'mpirun -np 1 MontePython.py run -p {param} -o {output} '\
                '-b {bf} -c {covmat} --minimize'.format(
                param=self.chains_dir+min_folder+'/log.param', 
                output=self.chains_dir+min_folder+'/',
                bf=self.chains_dir+prev_bf+'.bestfit', 
                covmat=self.chains_dir+self.info_root+'.covmat'
            )
            run(run_command, shell=True)
            # run a fake MCMC point at this minimum 
                # Here, we need to run a fake chain at this minimized point first
                # in order to create the .bestfit and .log files that play well with 
                #   the rest of the code. 
            run_command = 'mpirun -np 1 MontePython.py run -p {param} -o {output} '\
                '-b {bf} -c {covmat} -N {steps} -f {f}'.format(
                param=self.chains_dir+min_folder+'/log.param', 
                output=self.chains_dir+min_folder+'/',
                bf=self.chains_dir+min_folder+'/results.minimized', 
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=1, 
                f = 0
            )
            run(run_command, shell=True)
            # analyse 
            run_command = 'mpirun -np 1 MontePython.py info {folder} '\
                '--keep-non-markovian --noplot --minimal'.format(
                folder=self.chains_dir+min_folder+'/'
            )
            run(run_command, shell=True)
            # update and print minimum 
            new_min_point = pio.get_MP_bf_dict(f'{self.chains_dir}{prev_bf}.bestfit')
            print('\n\n------------------> After Minuit run, -logL minimized to '\
                  '{logL} \n\n'.format(logL=new_min_point['-logLike']))

        return True

    def first_jump_fac_less_than_prof_incr(self):
        """
        Checks if the first element of the 'jump_fac' sequence in the simulated-annealing 
        minimizer is less than the absolute value of the ratio between the step size
        in the profile parameter 'prof_incr', and its 1sigma error from the covariance 
        matrix. 
    
        :return: True if j < \Delta \theta_i / \sigma_i
        """
        covmat_header = pio.read_header_as_list(f'{self.chains_dir}{self.info_root}.covmat')
        covmat = np.loadtxt(f'{self.chains_dir}{self.info_root}.covmat')    
        prof_param_index = covmat_header.index(self.prof_param)
        param_1sigma = covmat[prof_param_index,prof_param_index]**0.5
    
        is_j_less_than_incr = (self.jump_fac[0] < np.abs(self.prof_incr/param_1sigma) )
        # we want j < \Delta \theta_i / \sigma_i
    
        if not is_j_less_than_incr:
            print('   ___\n  // \\\\\n // ! \\\\\n//_____\\\\\n')
            print('Warning: Increments in profile parameter are smaller than '\
            'the first jumping factor in simulated annealing sequence. \n'\
            'Ideally, we want first jumping factor < '\
            '(increment in profile parameter)/(error in profile parameter) \n'\
            'Currently we have j > Delta theta_i/ sigma_i with \n'\
            f'{self.jump_fac[0]} > {np.abs(self.prof_incr)}/{param_1sigma} = {np.abs(self.prof_incr/param_1sigma)} \n'\
            'This will result in a poor profile. \n'\
            'Either increase the profile increment prof_incr, \n'\
            'or decrease the first jumping factor in the simulated annealing sequence list jump_fac\n'\
                 )
        
        return is_j_less_than_incr
    
    
    def init_lkl_prof(self, lkl_dir = "lkl_prof"):
        """
        Initialise profile lkl yaml:
        1) create profile lkl folder and copy the global log.param into it
        2) read in last .bestfit file and set that as the current MLs dictionary, 
                as self.MLs 
            this updates the location in prof_param that we're at for running prof lkls.
            Under MP, decided to do this instead of updating to the last line of the 
                    lkl output file
        3) copy global bf into prof lkl output bf if the file doesn't exist 

        :lkl_dir: Leave the extension alone, thank you. 
        
        :return: the current lkl prof_param value 
        """
        
        global_lp = f'{self.chains_dir}log.param'
        full_lkl_dir = f'{self.chains_dir}{lkl_dir}{self.pn_ext("/")}'

        pio.make_path(full_lkl_dir, exist_ok=True)
        _ = pio.file_copy(global_lp, full_lkl_dir)
                
        try: 
            self.read_minimum()
        except OSError:
            # the lkl prof bf and lof files don't exist
            # copy global bf 
            _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.bestfit', 
                              f'{self.chains_dir}{self.info_root}{self.pn_ext("_lkl_prof")}.bestfit')
            # copy global log 
            _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.log', 
                              f'{self.chains_dir}{self.info_root}{self.pn_ext("_lkl_prof")}.log')

            # now this should work 
            self.read_minimum()
        
        # # /!\ Used to be initialised to last entry of lkl prof txt file 
        # self.MLs = self.read_lkl_output()
        # # Copy last lkl profile txt point into the bestfit file:
        # lkl_prof_header = pio.read_header_as_list(self.info_root+self.pn_ext('_lkl_profile.txt'))
        # update_bf_to_last_point = self.info_root+self.pn_ext("_lkl_prof")+".bestfit"
        # with open(update_bf_to_last_point, 'w') as lkl_txt: 
        #     lkl_txt.write("#       ")
        #     lkl_txt.write((",      ").join(lkl_prof_header))
        #     lkl_txt.write("\n")
        #     lkl_txt.write(str(self.MLs[lkl_prof_header[0]]))
        #     for param in lkl_prof_header[1:]:
        #         lkl_txt.write("    "+str(self.MLs[param]) )

        _ = self.first_jump_fac_less_than_prof_incr()
        
        return self.MLs[self.prof_param]


    def increment_update_logparam(self, lkl_dir = "lkl_prof"):
        """
        Update log.param value for prof_param to next increment = current + increment
        
        
        :lkl_dir: Leave the extension alone, thank you. 
        
        :return: new value of prof_param that the log.param was updated to, 
                    string of the line containing the prof_param in the log.param
        """
        extension = self.pn_ext('/')
        lkl_lp = f'{self.chains_dir}{lkl_dir}{extension}log.param'

        with open(lkl_lp, 'r') as f:
            lkl_lp_lines = f.readlines()

        line_modified = False
        lp_prof_param_string = f"'{self.prof_param}'"
        with open(lkl_lp, 'w') as f:
            for line in lkl_lp_lines:
                if lp_prof_param_string in line:
                    # print("Original: \t"+line)
                    prof_param_lp_line = line.split('=')
                    prof_param_lp_data = prof_param_lp_line[1].split(',')
                    
                    updated_prof_param = self.MLs[self.prof_param]+self.prof_incr
                    prof_param_lp_data[0] = f'[{updated_prof_param}'
                    prof_param_lp_data[3] = '0.'
                
                    prof_param_lp_data_str = ','.join(prof_param_lp_data)
                    prof_param_lp_line = f'{prof_param_lp_line[0]} '\
                        f'= {prof_param_lp_data_str}'
                    # print("Modified: \t"+line)
                    f.write(prof_param_lp_line)
                    line_modified = True
                else:
                    f.write(line)

        if line_modified is False:
            raise LogParamUpdateError(self.prof_param, lkl_lp)

        return updated_prof_param, prof_param_lp_line
    
    
    def get_prof_param_value_from_lp(self, lp_dir = "lkl_prof"):
        """
        Get current value of the prof lkl parameter from the lop param file 
        
        :lp_dir: directory of the log.param file to read. Default set for using 
            function internally. 
        
        :return: 'mean' of prof lkl parameter in the log.param as float 
        """
        if lp_dir:
            lp_dir += self.pn_ext('/')
        lp_file = f'{lp_dir}log.param'

        with open(lp_file, 'r') as f:
            lkl_lp_lines = f.readlines()

        lp_prof_param_string = "'"+self.prof_param+"'"
        for line in lkl_lp_lines:
            if lp_prof_param_string in line:
                prof_param_line = line
                prof_param_line = prof_param_line.split("=")
                prof_param_line = prof_param_line[1].split(",")
                prof_param_line = prof_param_line[0].strip()[1:]

                prof_param_value = float(prof_param_line)
                break

        return prof_param_value

    
    def get_experiments(self):
        """
        Extracts a list of likelihoods from the log.param file in the specified 
        chains directory.
    
        Returns:
        list or None: A list of likelihoods if found, otherwise None.
    
        Raises:
        FileNotFoundError: If the log.param file is not found in the specified 
        chains directory.
    
        Example:
        >>> profile = lkl_prof(chains_dir='/path/to/chains/', ...)
        >>> likelihoods = profile.get_experiments()
        >>> print(likelihoods)
        ['Planck_highl_TTTEEE', 'Planck_lowl_EE', 'Planck_lowl_TT']
        """
        # Read the text file
        lp_file_content = pio.read_file(f'{self.chains_dir}log.param')
        # Define the pattern for finding likelihoods
        experiments_line = re.compile(r"data\.experiments=\[([^\]]+)\]")
        # Use regular expression to find the likelihoods
        match = experiments_line.search(lp_file_content)
        if match:
            # Extract the likelihoods from the matched group
            likelihoods_str = match.group(1)
            # Split the likelihoods string into a list
            likelihoods = [lkl.strip(' ').strip("'") 
                           for lkl in likelihoods_str.split(',')]
            # Print the extracted likelihoods
            # print("Likelihoods of interest:")
            # for lkl in likelihoods:
            #     print(lkl)
        else:
            likelihoods = None
            print(f'Error in get_experiments: No likelihoods found in the file ' \
                  f'{self.chains_dir}log.param.')
        self.likelihoods = likelihoods
        return likelihoods

    
    def MP_run_chi2_per_exp_at_point(self, output_dir, param_point_bf_file):
        """
        Run MontePython to calculate and display chi^2 values for each likelihood 
        at a specific parameter point.
    
        Args:
        output_dir (str): The directory to store the output files from the MontePython run.
        param_point_bf_file (str): The file containing the best-fit parameter point 
          for which chi^2 values will be calculated.
    
        Returns:
        str: The output string from the MontePython run, 
          containing effective chi^2 values for different likelihoods.
    
        Example:
        >>> output_directory = '/path/to/output/'
        >>> best_fit_param_file = '/path/to/best_fit.bestfit'
        >>> chi2_output_str = profile.MP_run_chi2_per_exp_at_point(output_directory, best_fit_param_file)
        >>> print(chi2_output_str)
        "... -> for Planck_highl_TTTEEE : ... chi2eff= 123.45 ... -> for Planck_lowl_EE : ... chi2eff= 67.89 ..."
        """
        mp_run_command = 'mpirun -np 1 MontePython.py -N 1 -f 0 --display-each-chi2 '\
        f'-o {output_dir} -p {self.chains_dir}log.param -b {param_point_bf_file}' 
        captured_output = run(mp_run_command, shell=True, check=True, 
                              capture_output=True).stdout
        # Turn our b-string into a normal string 
        chi2_per_exp_output = captured_output.decode('utf-8')
        return chi2_per_exp_output

    
    def get_chi2_per_exp_dict(self, chi2_per_exp_output, likelihoods=None):
        """
        Extract and return chi^2 values for each likelihood from a given output string.
    
        This method parses the output string generated by a 
        MontePython --display-each-chi2 
        run to extract effective chi^2 values for different likelihoods.
    
        Args:
        likelihoods (list, optional): A list of likelihood names. 
            Defaults to None, in which case it uses the likelihoods attribute of 
            the class instance.
        chi2_per_exp_output (str): The output string from a 
            MontePython --display-each-chi2 
            run containing effective chi^2 values for different likelihoods.
    
        Returns:
        dict: A dictionary where keys are likelihood names and values are 
            corresponding chi^2 values.
    
        Raises:
        ExperimentNotFoundError: If the effective chi^2 value is not found 
            for a given likelihood or 'Total'.
    
    
        Example:
        >>> profile = lkl_prof(chains_dir='/path/to/chains/', ...)
        >>> likelihood_list = profile.get_experiments()
        >>> chi2_output_str = "... -> for Planck_highl_TTTEEE : ... chi2eff= 123.45 
                               ... -> for Planck_lowl_EE : ... chi2eff= 67.89 ..."
        >>> chi2_dict = profile.get_chi2_per_exp_dict(chi2_output_str)
        >>> print(chi2_dict)
        {'Planck_highl_TTTEEE': 2400.00, 'Planck_lowl_EE': 400.00, 
            'Planck_lowl_TT': 25.00}
        """    
        # get any likelihood cross references for MP
        # where the log.param and output refer to the same likelihood
        # but with different names 
        exp_crosslist = pio.get_experiment_crosslist()
        
        if likelihoods is None:
            likelihoods = self.likelihoods
        # Initialize a dictionary to store chi2eff values for each likelihood
        chi2eff_values = {}
        # Iterate over likelihoods
        for lkl in likelihoods:
            # Define the reg expression pattern for finding chi2eff value
            pattern = re.compile(fr"-> for  {lkl} : .* chi2eff= ([0-9.-]+)")
            # Use regular expression to find the chi2eff value
            match = pattern.search(chi2_per_exp_output)
            if match:
                # Convert the matched value to float and store in the dictionary
                chi2eff_values[lkl] = float(match.group(1))
                
            # if the experiment is missing from the output, check the MP experiment 
            # crosslistings for any name swaps and repeat the above 
            else:
                if lkl in exp_crosslist:
                    lkl_cross = exp_crosslist[lkl]
                    pattern = re.compile(fr"-> for  {lkl_cross} : .* chi2eff= ([0-9.-]+)")
                    match = pattern.search(chi2_per_exp_output)
                    if match:
                        chi2eff_values[lkl] = float(match.group(1))
                    else:
                        raise ExperimentNotFoundError(lkl_cross)
                else:
                    raise ExperimentNotFoundError(lkl)
    
        # Total chi2 
        pattern = re.compile(r"-> Total:.*chi2eff= ([0-9.-]+)")
        # Use regular expression to find the chi2eff value
        match = pattern.search(chi2_per_exp_output)
        if match:
            # Convert the matched value to float and store in the dictionary
            chi2eff_values['Total'] = float(match.group(1))
        else:
            raise ExperimentNotFoundError('Total')
            
        return chi2eff_values

    def update_MLs_chi2_per_exp(self, param_point):
        """
        Update a parameter point with corresponding chi^2 values for 
        each likelihood using MontePython.
        We run MontePython at the param_point dictionary, get chi2 per 
        experiment and output a dictionary of the param and chi2 values. 
    
        Args:
        param_point (dict): A dictionary representing a point in the parameter space.
    
        Returns:
        dict: A dictionary containing the original parameter point along with the 
        chi^2 values for each likelihood.
    
        Example:
        >>> parameter_point = {'param1': 1.0, 'param2': 2.0, 'param3': 3.0}
        >>> updated_point = update_MLs_chi2_per_exp(parameter_point)
        >>> print(updated_point)
        {'param1': 1.0, 'param2': 2.0, 'param3': 3.0, 'Planck_highl_TTTEEE': 2400.00, 'Planck_lowl_EE': 400.00, 'Planck_lowl_TT': 25.00}
        """
        # set likelihoods variable from log.param 
        self.get_experiments()
        
        # set location for running this point 
        save_output_bf_loc = f'{self.chains_dir}chi2_per_exp/'
        pio.make_path(save_output_bf_loc)
        # set bf file for running file 
        save_output_bf_file = save_output_bf_loc+'chi2_per_exp.bestfit'
    
        # Write this point to a MP style .bestfit file 
        pio.write_bf_dict_to_file(param_point, save_output_bf_file)
        # Run MP at this point and get chi2 values from output as dict 
        chi2_per_exp_output = self.MP_run_chi2_per_exp_at_point(output_dir=save_output_bf_loc, param_point_bf_file=save_output_bf_file)
        # print(chi2_per_exp_output)
        chi2eff_values = self.get_chi2_per_exp_dict(chi2_per_exp_output)
        # new dictionary with passed parameter points and chi2eff values 
        params_and_chi2s = deepcopy(param_point)
        params_and_chi2s.update(chi2eff_values)
    
        return params_and_chi2s

    def make_log_file(self, bf_file, output_loc=None, output_log_file=None):
        """
        Generate a .log file containing the minimum of -logLike for a given .bestfit file
        using MontePython, following the usual syntax of MP. 
    
        This function sets the likelihoods variable from log.param, runs MontePython at the 
        specified parameter point given by bf_file, obtains chi^2 values for each experiment, 
        and appends the total -logLike to a .log file.
    
        Args:
        bf_file (str): File path to the MontePython best-fit parameter file.
        output_loc (str, optional): Output location for the .log file. Defaults to 
        self.chains_dir.
    
        Returns:
        str: File path to the generated .log file.
    
        Example:
        >>> log_file_path = make_log_file('my_procoli.bestfit')
        >>> print(log_file_path)
        'my_procoli.log'
        """
        # set output defaults 
        if not output_loc:
            output_loc = self.chains_dir
        if not output_log_file:
            output_log_file = f'{bf_file[:-8]}.log'
        
        # set likelihoods variable from log.param 
        self.get_experiments()
        
        # Run MP at this point and get chi2 values from output as dict 
        chi2_per_exp_output = self.MP_run_chi2_per_exp_at_point(
            output_dir=output_loc, 
            param_point_bf_file=bf_file
        )
        # print(chi2_per_exp_output)
        chi2eff_values = self.get_chi2_per_exp_dict(chi2_per_exp_output)
        # only relevant line to add to .log file 
        save_loglike_in_log = f"--> Minimum of -logLike           : {chi2eff_values['Total']/2}"
    
        # save the .log file
        pio.save_file( output_log_file, 
                       lines=save_loglike_in_log
                     )
    
        return output_log_file
        
    def update_and_save_min_output(self, extension='_lkl_prof'):
        """
        Function to add the profile lkl param to the output bf file, 
        by copying the file to the main folder with this addition. 
        /!\ File naming scheme hard-coded within this function. 
        
        :extension: Leave it alone, thank you. 
        
        :return: current value of the prof lkl param as a float 

        """
        prefix_extension = self.pn_ext(extension)

        pn_ext = self.pn_ext('/')
        min_output_bf = f'{self.chains_dir}lkl_prof{pn_ext}'\
            f'lkl_prof{pn_ext[:-1]}.bestfit'

        bf_lines = pio.readlines_file(min_output_bf)

        bf_lines[0] = f'{bf_lines[0][:-1]},        {self.prof_param}\n'
        bf_lines[1] = f'{bf_lines[1][:-1]} {self.current_prof_param}\n'

        save_output_bf = f'{self.chains_dir}{self.info_root}{prefix_extension}.bestfit'

        pio.save_file(save_output_bf, bf_lines)

        from_file = f'{self.chains_dir}lkl_prof{pn_ext}lkl_prof{pn_ext[:-1]}.log'
        to_file = f'{self.chains_dir}{self.info_root}{prefix_extension}.log'
        _ = pio.file_copy(from_file, to_file)

        return self.current_prof_param

    
    def run_lkl_prof(self, time_mins=False, N_min_steps=3000, run_minuit=False):
        """
        Run the likelihood profile loop. 
        Initialise time-keeping file if wanted. 
        
        While we are within the bounds of the profile param we want to explore: 
        1) check if the point we are currently at i.e. param_ML and MLs, matches the 
                last entry in the lkl_prof table.
            - if it does, the last minimum was run and saved successfully.
            - if not, check if a minimum file exists. 
                - if it does, read it in and save it in the lkl prof txt. minimum run 
                    successfully. 
        2) check if minimum was run and saved. 
            - if yes, increment the prof lkl param and update the log.param, 
              remove all previous chains and analysis files created from the previous 
                increment. 
              Remember, this increment means prof_param = current_MLs + increment. 
              So we are simply one increment away from the most recent .bestfit file. 
            - With this setup, we really shouldn't wind up in a mininmum-not-saved 
                regime. 
              But in this case, grab the current value of prof_param so we have it, 
              and still remove files from any previous run. 
        3) run the minimizer, pointing to the bf file (usually the one we just wrote) 
            as starting bf
        4) save minimizer output + value of the prof param into a new bf in the 
            main folder. 
        
        Finally, outside the loop, save the output of the last minimizer. 

        :time_mins: boolean for whether you want to time each minimiser increment or not
        :N_min_steps: Number of steps to run per rung of the minimizer 
        
        :return: the value of the profile lkl parameter at the end of this loop 
        """
        _ = self.first_jump_fac_less_than_prof_incr()

        if time_mins is True:
            time_extension = self.pn_ext('_time_stamps.txt')
            with open(f'{self.chains_dir}{self.info_root}{time_extension}', 'a') as lkl_txt:
                lkl_txt.write("#")
                lkl_txt.write(f' {self.prof_param} \t step_size \t minimizer_time ')
                lkl_txt.write("\n")

        while ((self.MLs[self.prof_param] <= self.prof_max) 
               and (self.MLs[self.prof_param] >= self.prof_min)):
            last_entry_matches_current_params = self.match_param_line(self.MLs)
            if not last_entry_matches_current_params:
                param_names, param_ML, self.MLs = self.read_minimum()
                # read_min updates self.MLs 
                MLs_and_chi2 = self.update_MLs_chi2_per_exp(self.MLs)
                self.write_MLs(MLs_and_chi2)
                print(f'run_lkl_prof: -----> Minimizer run successfully for '\
                        f'{self.prof_param} = {self.MLs[self.prof_param]}')

            # TODO see about re-writing this function to not need to go line by 
            #   line somehow.  I don't know if there is anything better
            self.current_prof_param, _ = self.increment_update_logparam()  

            # break out of the loop if the parameter is outside it's range
            if not ((self.current_prof_param <= self.prof_max) 
                    and (self.current_prof_param >= self.prof_min)):
                break
                
            pn_ext_str = self.pn_ext('/')
            rm_chain_path = f'{self.chains_dir}lkl_prof{pn_ext_str}20*'
            rm_info_path = f'{self.chains_dir}lkl_prof{pn_ext_str}'\
                                f'lkl_prof{pn_ext_str[:-1]}*'
            pio.rm_files_wildcared(rm_chain_path)
            pio.rm_files_wildcared(rm_info_path)

            time_start = time()

            print(f'run_lkl_prof: -----> Running point {self.prof_param} '\
                    f'= {self.current_prof_param}')
            self.run_minimizer(prev_bf=self.info_root+self.pn_ext("_lkl_prof"), 
                               min_folder="lkl_prof" + self.pn_ext('/')[:-1],
                               N_steps=N_min_steps, 
                               run_minuit=run_minuit)
            self.update_and_save_min_output() 

            time_end = time()
            time_taken = time_end - time_start
            
            if time_mins is True:
                with open(f'{self.chains_dir}{self.info_root}{time_extension}', 'a') as lkl_txt:
                    lkl_txt.write(f'{self.current_prof_param:.4g} '\
                                  f'\t {self.prof_incr:.2g} \t {time_taken:.2f} \n')
                print(f'run_lkl_prof:        Time taken for minimizer '\
                      f'= {time_taken:.2f}')

            param_names, param_ML, self.MLs = self.read_minimum()


            # prof_incr *= 2. # Readjust prof lkl increment if wanted by copying this 
            #   function and adding such a line 

        # outside loop now 
        # TODO do we need this?  Does it every actually need to run this?  
        #   Could it be before the loop begins and then put at the end of the loop, 
        #       but still insdie it
        last_entry_matches_current_params = self.match_param_line(self.MLs)
        if not last_entry_matches_current_params:
            param_names, param_ML, self.MLs = self.read_minimum()
            MLs_and_chi2 = self.update_MLs_chi2_per_exp(self.MLs)
            self.write_MLs(MLs_and_chi2)
            self.write_MLs(self.MLs)
            print(f'run_lkl_prof: -----> Minimizer run successfully for '\
                  f'{self.prof_param} = {self.MLs[self.prof_param]}')
        
        return self.MLs[self.prof_param]
    



    
    def full_lkl_prof_array(self):
        """
        Combine positive and negative increment files into one array 
        But first check that they have the same param order. 

        :return: full likelihood profile array 
        """

        pos_filename = f'{self.chains_dir}{self.info_root}_+'\
            f'{self.prof_param}_lkl_profile.txt'
        neg_filename = f'{self.chains_dir}{self.info_root}_-'\
            f'{self.prof_param}_lkl_profile.txt'

        
            
        try:
            pos_header = pio.read_header_as_list(pos_filename)
            all_MLs_p = pio.load_mp_info_files(pos_filename)
            pos_file = True
        except FileNotFoundError:
            pos_file = False
        try:
            neg_header = pio.read_header_as_list(neg_filename)
            all_MLs_n = pio.load_mp_info_files(neg_filename)
            if pos_file is True:
                if pos_header==neg_header:
                    all_MLs = np.concatenate( (np.flip(all_MLs_n, 0),all_MLs_p) )
                else:
                    print('full_lkl_prof_array: the positive and negative files either '\
                            'have different parameters '\
                            'or have them in different orders. \n'\
                            'Either way, this function cannot correctly combine them. ')
                    return 0
            else:
                all_MLs = np.flip(all_MLs_n, 0)
        except FileNotFoundError:
            if pos_file is True:
                all_MLs = all_MLs_p
            else:
                print('full_lkl_prof_array: could not find files '\
                      f'\n{pos_filename} \n{neg_filename} ')
        return all_MLs   


    def full_lkl_prof_dict(self):
        """
        Combine positive and negative increment files into one dictionary with 
        keys = param names 
        values = 1D array of profile likelihood values 
        
        :return: full likelihood profile dictionary 
        """
        full_prof_dict = {}
        
        full_lkl_prof_array = self.full_lkl_prof_array()

        try: 
            pos_filename = f'{self.chains_dir}{self.info_root}_+'\
                f'{self.prof_param}_lkl_profile.txt'
            lkl_prof_header = pio.read_header_as_list(pos_filename)
        except FileNotFoundError: 
            neg_filename = f'{self.chains_dir}{self.info_root}_-'\
                f'{self.prof_param}_lkl_profile.txt'
            lkl_prof_header = pio.read_header_as_list(neg_filename)

        for param_num in range(len(lkl_prof_header)):
            full_prof_dict[lkl_prof_header[param_num]] = full_lkl_prof_array[:,param_num]
        
        # # Commented out following. Using file header to get param order
        # for param_num in range(len(self.param_order)):
        #     full_prof_dict[self.param_order[param_num]] = full_lkl_prof_array[:,param_num]

        return full_prof_dict

    def sum_params(self, params_to_sum):
        """
        Sum list of params and return array of summed params.
        Useful for adding up chi^2's post profile lkl run 
        
        :params_to_sum: list of parameter names that you want to sum. 
        
        :return: array of summed parameters 
        """
        prof_lkl = self.full_lkl_prof_dict()

        param_vectors = [prof_lkl[param] for param in params_to_sum]
        param_stack = np.stack(param_vectors, axis=0)
        summed_params = param_stack.sum(axis=0)

        return summed_params
            
