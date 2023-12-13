import os, sys
from copy import deepcopy
from glob import glob
from subprocess import run
from time import time

import numpy as np
from getdist import chains, mcsamples, plots
from getdist.mcsamples import MCSamplesError

import procoli.procoli_io as pio 
from procoli.procoli_errors import *


class lkl_prof:
    
    def __init__(self, chains_dir, info_root, prof_param, processes=6, R_minus_1_wanted=0.05, 
                 mcmc_chain_settings={'ignore_rows' : 0.3}, 
                 prof_incr=None, prof_min=None, prof_max=None, 
                 jump_fac=None, lkl_fac=None
                ):
        
        self.chains_dir = chains_dir
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
        self.lkl_fac = lkl_fac
        
        self.covmat_file = self.chains_dir+self.info_root+'.covmat'
    
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
            It then replaces the chains in that instance with all the chains in the folder 
            No duplication of chains occurs. 
            
        :read_all_chains: boolean for whether to read all the chains in the chains directory 
        
        :return: True if files found, else False 
        """
        max_steps_in_chain = str( max( [ int(i[len(self.chains_dir)+11:-7]) for i in glob(f'{self.chains_dir}*__1.txt') ] ) )
        for file_root in glob(f'{self.chains_dir}*__1.txt'):
            if max_steps_in_chain in file_root:
                self.chain_root = file_root[len(self.chains_dir):-6]
        print(f"check_mcmc_chains: Looking for files: {self.chains_dir}{self.chain_root}")

        try:
            self.mcmc_chains = mcsamples.loadMCSamples(self.chains_dir+self.chain_root, settings=self.mcmc_chain_settings)
            self.covmat_file = self.chains_dir+self.info_root+'.covmat'
        except OSError:
            return False 
        
        if read_all_chains==True:
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
        
        with open(self.chains_dir+'log.param', 'r') as log:
            pass
        
        try:
            with open(self.chains_dir+self.info_root+'.bestfit', 'r') as log:
                        pass
            bf_exists = True    
        except FileNotFoundError:
            bf_exists = False
        try:
            with open(self.chains_dir+self.info_root+'.covmat', 'r') as log:
                        pass
            covmat_exists = True    
        except FileNotFoundError:
            covmat_exists = False

        if (bf_exists and covmat_exists):
            run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} --update 50 --superupdate 20".format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                bf=self.chains_dir+self.info_root+'.bestfit', 
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=N_steps
            )
        elif bf_exists:
            run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -N {steps} --update 50 --superupdate 20".format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                bf=self.chains_dir+self.info_root+'.bestfit', 
                steps=N_steps
            )
        elif covmat_exists:
            run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -c {covmat} -N {steps} --update 50 --superupdate 20".format(
                procs=self.processes,
                param=self.chains_dir+'log.param', 
                output=self.chains_dir,
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=N_steps
            )
        else:
            run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -N {steps} --update 50 --superupdate 20".format(
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
        
        :return: True if MCMC chains have converged to the desired R-1, default is R-1=0.05. Else False 
        """
        if mcmc_chains==None:
            mcmc_chains=self.mcmc_chains
            
        current_R_minus_1 = mcmc_chains.getGelmanRubin()
        if current_R_minus_1 < self.R_minus_1_wanted:
            print("check_mcmc_convergence: Chains converged sufficiently. Current R-1 = {:.3f} satisfies R-1 wanted = {:.3f}. \nMove on to checking minimum.".format(current_R_minus_1,self.R_minus_1_wanted))
            return True
        else: 
            print("check_mcmc_convergence: Chains not converged. Current R-1 = {:.3f} while R-1 wanted = {:.3f}. \nResume MCMC. ".format(current_R_minus_1,self.R_minus_1_wanted))
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
            run("mpirun -np 1 MontePython.py info "+self.chains_dir+" --keep-non-markovian --noplot --want-covmat", shell=True)
            self.run_mcmc(N_steps=50000)
            self.check_mcmc_chains(read_all_chains=True)
        return True

    def check_global_min(self, mcmc_chains=None):
        """
        Check for .bestfit file. This does not necessarily indicate a global minimum run!!! 
        It only indicates that there exists a file storing some bf in the 'info_root' file. 
        This also resets the info_root to the current directory name to avoid errors later in the code. 
        
        :mcmc_chains: getdist MCSamples instance 
        
        :return: True if global minimum was run and relevant files are accesible. Else False 
        """
        if mcmc_chains==None:
            mcmc_chains=self.mcmc_chains
            
        try:
            # TODO can probably check if it exists with the os module
            pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}.bestfit')
            print(f'check_global_min: Found minimum with file name {self.info_root}')
            pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}.covmat')
            print(f'check_global_min: Found covmat with file name {self.info_root}')
            
            new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
            if self.info_root != new_info_root:
                _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.bestfit', f'{self.chains_dir}{new_info_root}.bestfit')
                _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.log', f'{self.chains_dir}{new_info_root}.log')
                _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.covmat', f'{self.chains_dir}{new_info_root}.covmat')
                self.info_root = new_info_root
                
            return True
        except OSError:
            try:
                new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
                # TODO can we run montepython with mpirun directly from python?
                run(f'mpirun -np 1 MontePython.py info {self.chains_dir} --keep-non-markovian --noplot --want-covmat', shell=True, check=True)
                # TODO can probably check if it exists with the module
                pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}.bestfit')
                # TODO why change the info root?
                self.info_root = new_info_root
                print(f'check_global_min: Found minimum with file name {self.info_root}')
                return True 
            except OSError:
                print('check_global_min: Cannot run MP info for global minimum. Something went wrong. ')
                return False 
        
    def global_min(self, run_glob_min=False, N_min_steps=3000, run_minuit=False):
        """
        Check global minizer, run if wanted (default False), then write if not already written 

        So: 
        1) load / create the global minimum file. 
        2) If we want a global min run, run the minimizer 
        3) grab the global minimizer results 
        4) check if we have a file with prof lkl values. 
            * If yes, check that it has the same parameters and in the right order. Proceed. 
            * If no file, start it and write the first line as param names. Proceed. 
            * If file yes, but parameters don't match, then print an error. Stop. 
        5) check if global minimum params have already been written (first line of file)
            * If parameters are written, check that they match global minimum. Don't write them again
            * If parameters are written but don't match, spit out error. 
            * If no params written, add this current ML values for all parameters in append mode
            
        :run_glob_min: Boolean for whether to run a global minimizer 

        :return: global maximum lkl dictionary 
        """

        self.check_global_min()

        if run_glob_min:
            
            # Check if jumping factors and lkl factors are defined, 
            # otherwise set to defaults for global min 
            # provided by Yashvi Patel
            if self.jump_fac == None:
                self.jump_fac = [1, 0.8, 0.5, 0.2, 0.1, 0.05]
            if self.lkl_fac == None:
                self.lkl_fac =  [3, 4, 5, 10, 200, 1000]

            pio.makedirs(f'{self.chains_dir}global_min', exist_ok=True)
            _ = pio.file_copy(f'{self.chains_dir}log.param', f'{self.chains_dir}global_min/log.param')
            
            self.run_minimizer(min_folder='global_min', N_steps=N_min_steps, run_minuit=run_minuit)

            _ = pio.file_copy(f'{self.chains_dir}global_min/global_min.bestfit', f'{self.chains_dir}{self.info_root}.bestfit')
            _ = pio.file_copy(f'{self.chains_dir}global_min/global_min.log', f'{self.chains_dir}{self.info_root}.log')

        param_names, param_ML, MLs = self.read_minimum(extension='')
        self.global_ML = deepcopy(MLs)
        self.param_order = param_names.tolist()

        extension = '_lkl_profile.txt' 
        extension = self.pn_ext(extension)
        
        try:
            self.match_param_names(self.param_order)
        except FileNotFoundError:
            print(f'global_min: File not found. Starting a new file now: {self.chains_dir}{self.info_root}{extension}\n') 
            with open(f'{self.chains_dir}{self.info_root}{extension}', 'w') as lkl_txt: 
                lkl_txt.write('#')
                for param_recorded in self.param_order:
                    lkl_txt.write(f'\t {param_recorded}')
                lkl_txt.write("\n")

        lkl_prof_table = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{extension}') 

        # TODO param order should inherit from file header, param order not matching should never cause the code to fail
        if lkl_prof_table.shape!=(0,):
            if not self.match_param_line(self.global_ML, loc=0):
                raise GlobalMLDifferenceError(f'{self.chains_dir}{self.info_root}')
        else: 
            self.write_MLs()

        return self.global_ML
        
        
    def pn_ext(self, extension):
        """
        Prefix the file extension string input with 
        the sign of the profile lkl parameter, 
        and its name to track files correctly. 
        
        :extension: A string of the file name extension, eg. "_good_pupper"
        :return: String of extension prefixed with the sign and name of the profile lkl parameter "_+height_good_pupper"
        """
        if len(extension)>0:
            if self.prof_incr > 0:
                extension = '_+'+self.prof_param+extension
            if self.prof_incr < 0:
                extension = '_-'+self.prof_param+extension
        return extension
        
    def read_minimum(self, extension='_lkl_prof'):
        """
        Read minimum file and save parameter names list, parameter values list and MLs dictionary 
        Also update the dictionary object self.MLs 
        
        :extension: The extension of the life type being read in. Leave this as is, the rest of the code assumes the same naming conventions. Otherwise, specify to read a specific file, but know that this will update the self.MLs dict too. 
        
        :return: List of parameter names, list of parameter ML values, dictionary of {'param_names': param_ML_value}
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
        
        # TODO do we want to remove param_ML from the output?  It's never used as an output
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
        This is because the global param_names list is the one that has the correct order. 
        
        :extension: Leave it alone, thank you.
        
        :return: new length of the saved lkl profile table
        
        """
        if MLs == None:
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
        
        # TODO this can probably be combined so you don't have to compare them against each other twice
        mismatched_params = [param for param in param_names if param not in params_recorded]
        if not mismatched_params:
            param_names_in_rec = True
        else: 
            print("\nmatch_param_names: Params don't match. \nPassed param_names has the following params that are not expected by params recorded:")
            print(mismatched_params)
            param_names_in_rec = False

        mismatched_params = [param for param in params_recorded if param not in param_names]
        if not mismatched_params:
            rec_params_in_param_names = True
        else: 
            print("\nmatch_param_names: Params don't match. \nRecorded params expected that are absent in passed param_names: ")
            print(mismatched_params)
            rec_params_in_param_names = False

        if (param_names_in_rec and rec_params_in_param_names):
            print("match_param_names: Params match - the recorded params contain the same params as param_names passed. ")
            self.param_order = params_recorded
            return True
        else:
            raise ParamDifferenceError(f'{self.chains_dir}{self.info_root}{prefix_extension}')
            return False
    
    
    def match_param_line(self, MLs, param_names=None, extension='_lkl_profile.txt', loc=-1):
        """
        Check if specified (default: last) location in lkl_prof output file matches current MLs

        :param_names: list of parameter names in the same order as that printed in the file. 
                        This is usually the global param_order list. Note: LIST not array! 
        :MLs: dictionary of {'param_name': ML_value }
        :extension: Leave it alone, thank you. 
        :loc: integer location of row in file to check, default is the last line

        :return: True if match, else False 
        """

        prefix_extension = self.pn_ext(extension)

        if param_names==None:
            param_names=self.param_order

#         print("match_param_line: checking file {file}".format(file=self.chains_dir + self.info_root + prefix_extension) )
        lkl_prof_table = pio.load_mp_info_files(f'{self.chains_dir}{self.info_root}{prefix_extension}') 
        
        if lkl_prof_table.size==0:
            print("match_param_line: File empty ")
            return False
        else: 
            try:
                lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
                if False in [lkl_prof_table[loc, param_names.index(param) ] == MLs[param] for param in param_names]:
                    return False
                else:
                    return True 
            except IndexError:
                print("match_param_line: Only one entry in file, checking that entry ")
                if False in [lkl_prof_table[param_names.index(param) ] == MLs[param] for param in param_names]:
                    return False 
                else:
                    return True
    
    
    def run_minimizer(self, min_folder="lkl_prof", prev_bf=None, N_steps=5000, run_minuit=False):
        """
        Run minimizer as described in 2107.10291, by incrementally running a finer MCMC 
        with a more discrening lklfactor that increases preference for moving towards higher likelihoods 
        and taking smaller jumps between points so we sample a finer grid in parameter space. 
        This can be modified as wanted, changing step sizes per rung, lkl factor and jumping factors. 
        You would need to change this function, then import the class and run a lkl prof as usual. 
        Default:
        N_steps = input parameter, same for all rungs
        lklfactor = 10, 200, 1000
        jumping factor (-f) = 0.5, 0.1, 0.05

        Requires the folder min_folder to already be popualted with a log.param file 

        :min_folder: Folder to run minimizer in 
        :prev_bf: Starting best-fit file. 
                This is the MAP of the MCMC chains for the global bf run, 
                and the previous point for the lkl prof. 
                The function init_lkl_prof takes care of designating this, set interanlly. 
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
        # Check if jumping factors and lkl factors are defined, otherwise set to defaults, provided by Yashvi Patel
        if self.jump_fac == None:
            self.jump_fac = [0.15, 0.1, 0.05]
        if self.lkl_fac == None:
            self.lkl_fac = [10, 200, 1000]
        if len(self.jump_fac) != len(self.lkl_fac):
            print("!!!!!!!!!\n!!!!!!!!!\n!!!!!!!!!")
            print("Error in run_minimizer: Lists passed for jumping factor and lkl factor are of different lengths. \
            Setting to defaults!!! ")
            print("!!!!!!!!!\n!!!!!!!!!\n!!!!!!!!!")
            self.jump_fac = [0.5, 0.2, 0.1, 0.05]
            self.lkl_fac = [1, 10, 200, 1000]

        ##### First rung #####

        # MCMC
        mp_run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=N_steps, 
            f = self.jump_fac[0], 
            lkl = self.lkl_fac[0]
        )
        run(mp_run_command, shell=True, check=True)
        # analyse 
        mp_info_command = "mpirun -np 1 MontePython.py info {folder} --keep-non-markovian --noplot".format(
            folder=self.chains_dir+min_folder+'/'
        )
        run(mp_info_command, shell=True)
        # print output 
        if min_folder=='.':
            prev_bf = [x for x in str(os.getcwd()).split('/') if x][-1]
            # switch to current directory as bf root, ensures that we're using the most recent file 
        else:
            prev_bf = min_folder+'/'+min_folder 
            # switch to most recently produced bf file in the minimizer directory as bf root 
        # set new minimum 
        new_min_point = pio.get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
        print("\n\n------------------> After minimizer rung 1, -logL minimized to  {logL} \n\n".format(
            logL=new_min_point['-logLike']))


        ##### Loop over other rungs #####

        num_itrs = len(self.jump_fac)

        for i in range(1,num_itrs):
            # MCMC
            run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
                procs=self.processes,
                param=self.chains_dir+min_folder+'/log.param', 
                output=self.chains_dir+min_folder+'/',
                bf=self.chains_dir+prev_bf+'.bestfit', 
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=N_steps, 
                f = self.jump_fac[i], 
                lkl = self.lkl_fac[i]
            )
            run(run_command, shell=True)
            # analyse 
            run_command = "mpirun -np 1 MontePython.py info {folder} --keep-non-markovian --noplot".format(
                folder=self.chains_dir+min_folder+'/'
            )
            run(run_command, shell=True)
            # set new minimum 
            new_min_point = pio.get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
            print("\n\n------------------> After minimizer rung {ith}, -logL minimized to  {logL} \n\n".format(
                ith=i+1, logL=new_min_point['-logLike']))
            
        ##### Bonus step: run Minuit minimizer #####
        
        if run_minuit:
            # NOTE: this will only work well if the minimizer outputs results that have correctly scaled params 
            # MP rescales some params: omega_b (1e-2), A_s (1e-9) and a couple of PLC params 
            # So this output needs to scale them back to normal, with omega_b of O(0.01), etc. 
            # TK will update MP to do this correctly. Fix already there, need to turn print into replacement 

            # Run minuit minimizer 
            run_command = "mpirun -np 1 MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} --minimize".format(
                param=self.chains_dir+min_folder+'/log.param', 
                output=self.chains_dir+min_folder+'/',
                bf=self.chains_dir+prev_bf+'.bestfit', 
                covmat=self.chains_dir+self.info_root+'.covmat'
            )
            run(run_command, shell=True)
            # run a fake MCMC point at this minimum 
                # Here, we need to run a fake chain at this minimized point first
                # in order to create the .bestfit and .log files that play well with the rest of the code. 
            run_command = "mpirun -np 1 MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f}".format(
                param=self.chains_dir+min_folder+'/log.param', 
                output=self.chains_dir+min_folder+'/',
                bf=self.chains_dir+min_folder+'/results.minimized', 
                covmat=self.chains_dir+self.info_root+'.covmat',
                steps=1, 
                f = 0
            )
            run(run_command, shell=True)
            # analyse 
            run_command = "mpirun -np 1 MontePython.py info {folder} --keep-non-markovian --noplot".format(
                folder=self.chains_dir+min_folder+'/'
            )
            run(run_command, shell=True)
            # update and print minimum 
            new_min_point = pio.get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
            print("\n\n------------------> After Minuit run, -logL minimized to  {logL} \n\n".format(
                logL=new_min_point['-logLike']))

        return True
    
    
    def init_lkl_prof(self, lkl_dir = "lkl_prof"):
        """
        Initialise profile lkl yaml:
        1) create profile lkl folder and copy the global log.param into it
        2) read in last .bestfit file and set that as the current MLs dictionary, as self.MLs 
            this updates the location in prof_param that we're at for running prof lkls. 
            Under MP, decided to do this instead of updating to the last line of the lkl output file
                TK: fixed different bug in code which now no longer needs this update, from what I can see.
                Leaving this comment here for posterity / future fixes:
                    I don't know why I did that. Ideally, we should be at the last point that was acually saved..
                    Oh, right. I did that because MP needs a .bf file to start the next minimizer run with... 
                    /!\ AMMEND THIS FUNCTION: should take last step in lkl prof txt file and copy that into the bf to be used. 
                    This also works for the very first step which should be the global bf,
                    which would be the last point in the lkl prof txt file if this is the first step anyway 
        3) copy global bf into prof lkl output bf if the file doesn't exist 

        :lkl_dir: Leave the extension alone, thank you. 
        
        :return: the current lkl prof_param value 
        """
        
        global_lp = f'{self.chains_dir}log.param'
        full_lkl_dir = f'{self.chains_dir}{lkl_dir}{self.pn_ext("/")}'

        pio.makedirs(full_lkl_dir, exist_ok=True)
        _ = pio.file_copy(global_lp, full_lkl_dir)
                
        try: 
            self.read_minimum()
        except OSError:
            # the lkl prof bf and lof files don't exist
            # copy global bf 
            _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.bestfit', f'{self.chains_dir}{self.info_root}{self.pn_ext("_lkl_prof")}.bestfit')
            # copy global log 
            _ = pio.file_copy(f'{self.chains_dir}{self.info_root}.log', f'{self.chains_dir}{self.info_root}{self.pn_ext("_lkl_prof")}.log')

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
                    prof_param_lp_line = f'{prof_param_lp_line[0]} = {prof_param_lp_data_str}'
                    # print("Modified: \t"+line)
                    f.write(prof_param_lp_line)
                    line_modified = True
                else:
                    f.write(line)

        if line_modified == False:
            raise LogParamUpdateError(self.prof_param, lkl_lp)

        return updated_prof_param, prof_param_lp_line
    
    
    def get_prof_param_value_from_lp(self, lp_dir = "lkl_prof"):
        """
        Get current value of the prof lkl parameter from the lop param file 
        
        :lp_dir: directory of the log.param file to read. Default set for using function internally. 
        
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
        min_output_bf = f'{self.chains_dir}lkl_prof{pn_ext}lkl_prof{pn_ext[:-1]}.bestfit'

        with open(min_output_bf, 'r') as f:
            bf_lines = f.readlines()

        bf_lines[0] = f'{bf_lines[0][:-1]},        {self.prof_param}\n'
        bf_lines[1] = f'{bf_lines[1][:-1]} {self.current_prof_param}\n'

        save_output_bf = f'{self.chains_dir}{self.info_root}{prefix_extension}.bestfit'

        with open(save_output_bf, 'w') as f:
            for line in bf_lines:
                f.write(line)

        from_file = f'{self.chains_dir}lkl_prof{pn_ext}lkl_prof{pn_ext[:-1]}.log'
        to_file = f'{self.chains_dir}{self.info_root}{prefix_extension}.log'
        _ = pio.file_copy(from_file, to_file)

        return self.current_prof_param
        
    def run_lkl_prof(self, time_mins=False, N_min_steps=5000, run_minuit=False):
        """
        Run the likelihood profile loop. 
        Initialise time-keeping file if wanted. 
        
        While we are within the bounds of the profile param we want to explore: 
        1) check if the point we are currently at i.e. param_ML and MLs, matches the last entry in the lkl_prof table.
            - if it does, the last minimum was run and saved successfully.
            - if not, check if a minimum file exists. 
                - if it does, read it in and save it in the lkl prof txt. minimum run successfully. 
        2) check if minimum was run and saved. 
            - if yes, increment the prof lkl param and update the log.param, 
              remove all previous chains and analysis files created from the previous increment. 
              Remember, this increment means prof_param = current_MLs + increment. 
              So we are simply one increment away from the most recent .bestfit file. 
            - With this setup, we really shouldn't wind up in a mininmum-not-saved regime. 
              But in this case, grab the current value of prof_param so we have it, 
              and still remove files from any previous run. 
        3) run the minimizer, pointing to the bf file (usually the one we just wrote) as starting bf
        4) save minimizer output + value of the prof param into a new bf in the main folder. 
        
        Finally, outside the loop, save the output of the last minimizer. 

        :time_mins: boolean for whether you want to time each minimiser increment or not 
        :N_min_steps: Number of steps to run per rung of the minimizer 
        
        :return: the value of the profile lkl parameter at the end of this loop 
        """
        if time_mins == True:
            time_extension = self.pn_ext('_time_stamps.txt')
            with open(f'{self.chains_dir}{self.info_root}{time_extension}', 'a') as lkl_txt:
                lkl_txt.write("#")
                lkl_txt.write(f' {self.prof_param} \t step_size \t minimizer_time ')
                lkl_txt.write("\n")

        while ((self.MLs[self.prof_param] <= self.prof_max) and (self.MLs[self.prof_param] >= self.prof_min)):
            last_entry_matches_current_params = self.match_param_line(self.MLs)
            if not last_entry_matches_current_params:
                param_names, param_ML, self.MLs = self.read_minimum()
                # read_min updates self.MLs 
                self.write_MLs(self.MLs)
                print(f'run_lkl_prof: -----> Minimizer run successfully for {self.prof_param} = {self.MLs[self.prof_param]}')

            # TODO see about re-writing this function to not need to go line by line somehow.  I don't know if there is anything better
            self.current_prof_param, prof_param_string_in_logparam = self.increment_update_logparam()  

            # break out of the loop if the parameter is outside it's range
            if not ((self.current_prof_param <= self.prof_max) and (self.current_prof_param >= self.prof_min)):
                break
                
            pn_ext_str = self.pn_ext('/')
            rm_chain_path = f'{self.chains_dir}lkl_prof{pn_ext_str}20*'
            rm_info_path = f'{self.chains_dir}lkl_prof{pn_ext_str}lkl_prof{pn_ext_str[:-1]}*'
            pio.rm_files_wildcared(rm_chain_path)
            pio.rm_files_wildcared(rm_info_path)

            time_start = time()

            print(f'run_lkl_prof: -----> Running point {self.prof_param} = {self.current_prof_param}')
            self.run_minimizer(prev_bf=self.info_root+self.pn_ext("_lkl_prof"), 
                               min_folder="lkl_prof" + self.pn_ext('/')[:-1],
                               N_steps=N_min_steps, 
                               run_minuit=run_minuit)
            self.update_and_save_min_output() 

            time_end = time()
            time_taken = time_end - time_start
            
            if time_mins == True:
                with open(f'{self.chains_dir}{self.info_root}{time_extension}', 'a') as lkl_txt:
                    lkl_txt.write(f'{self.current_prof_param:.4g} \t {self.prof_incr:.2g} \t {time_taken:.2f} \n')
                print(f'run_lkl_prof:        Time taken for minimizer = {time_taken:.2f}')

            param_names, param_ML, self.MLs = self.read_minimum()


            # prof_incr *= 2. # Readjust prof lkl increment if wanted by copying this function and adding such a line 

        # outside loop now 
        # TODO do we need this?  Does it every actually need to run this?  Could it be before the loop begins and then put at the end of the loop, but still insdie it
        last_entry_matches_current_params = self.match_param_line(self.MLs)
        if not last_entry_matches_current_params:
            param_names, param_ML, self.MLs = self.read_minimum()
            self.write_MLs(self.MLs)
            print(f'run_lkl_prof: -----> Minimizer run successfully for {self.prof_param} = {self.MLs[self.prof_param]}')
        
        return self.MLs[self.prof_param]
    



    
    def full_lkl_prof_array(self):
        """
        Combine positive and negative increment files into one array 
        But first check that they have the same param order. 

        :return: full likelihood profile array 
        """

        pos_filename = f'{self.chains_dir}{self.info_root}_+{self.prof_param}_lkl_profile.txt'
        neg_filename = f'{self.chains_dir}{self.info_root}_-{self.prof_param}_lkl_profile.txt'

        
            
        try:
            pos_header = pio.read_header_as_list(pos_filename)
            all_MLs_p = pio.load_mp_info_files(pos_filename)
            pos_file = True
        except FileNotFoundError:
            pos_file = False
        try:
            neg_header = pio.read_header_as_list(neg_filename)
            all_MLs_n = pio.load_mp_info_files(neg_filename)
            if pos_file==True:
                if pos_header==neg_header:
                    all_MLs = np.concatenate( (np.flip(all_MLs_n, 0),all_MLs_p) )
                else:
                    print('full_lkl_prof_array: the positive and negative files either have different parameters \
                            or have them in different orders. \
                            \nEither way, this function cannot correctly combine them. ')
                    return 0
            else:
                all_MLs = np.flip(all_MLs_n, 0)
        except FileNotFoundError:
            if pos_file == True:
                all_MLs = all_MLs_p
            else:
                print(f'full_lkl_prof_array: could not find files \n{pos_filename} \n{neg_filename} ')
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
            pos_filename = f'{self.chains_dir}{self.info_root}_+{self.prof_param}_lkl_profile.txt'
            lkl_prof_header = pio.read_header_as_list(pos_filename)
        except FileNotFoundError: 
            neg_filename = f'{self.chains_dir}{self.info_root}_-{self.prof_param}_lkl_profile.txt'
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
            