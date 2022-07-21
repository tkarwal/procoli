from getdist import mcsamples, plots, chains
from getdist.mcsamples import MCSamplesError
import numpy as np
from subprocess import run
import os
from yaml import dump
from cobaya.yaml import yaml_load_file
from copy import deepcopy
from time import time
from glob import glob


def read_header_as_list(filename):
    """
    Read in header of file and save as list. 
    Header can be comma or space delimited. 


    :filename: full path to file, ideally

    :return: list of elements of header 
    """
    with open(filename, 'r') as fc:
        header = fc.readline()
    if ',' in header:
        out_list = [elem.strip() for elem in header[1:].split(',')]
    else:
        out_list = [elem.strip() for elem in header[2:].split(' ')]
    out_list = list(filter(None, out_list))
    
    return out_list

def get_MP_bf_dict(MP_bf_file):
    """
    Read in MontePython style best fit file and save as dictionary. 
    
    :MP_bf_file: full path to MP best fit file, ideally

    :return: dictionary of {'param_names': best_fit_point}
    """

    MP_param_values = np.loadtxt(MP_bf_file)

    MP_param_names = read_header_as_list(MP_bf_file)

    MP_bf = dict(zip(MP_param_values, MP_param_names))

    try:
        with open(MP_bf_file[:-8]+'.log') as log_file:
            last_line = log_file.readlines()[-1]
            neg_logLike = float(last_line.split(":")[-1])
            MP_bf['-logLike'] = neg_logLike
    except FileNotFoundError:
        pass

    return MP_bf


class lkl_prof:
    
    def __init__(self, chains_dir, info_root, prof_param, processes=6, R_minus_1_wanted=0.05, 
                 mcmc_chain_settings={'ignore_rows' : 0.3}, 
                 minimizer_settings={'minimize': {'method': 'bobyqa','covmat' : 'auto',}}, # Remove 
                 prof_incr=None, prof_min=None, prof_max=None
                ):
        
        self.chains_dir = chains_dir
        self.info_root = info_root # this should change. Maybe info_root 
        
        self.processes = processes
        self.R_minus_1_wanted = R_minus_1_wanted
        self.mcmc_chain_settings = mcmc_chain_settings
        self.minimizer_settings = minimizer_settings # Remove 
        
        self.prof_param = prof_param
        self.prof_incr = prof_incr
        self.prof_min = prof_min
        self.prof_max = prof_max
        
        self.covmat_file = minimizer_settings['minimize']['covmat'] # Change 
        
        os.chdir(self.chains_dir)
    
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
        
        :return: True if files found, else False 
        """
        max_steps_in_chain = str( max( [ int(i[11:-7]) for i in glob('*__1.txt') ] ) )
        for file_root in glob('*__1.txt'):
            if max_steps_in_chain in file_root:
                self.chain_root = file_root[:-6]
        print("check_mcmc_chains: Looking for files: "+self.chains_dir+self.chain_root)

        try:
            self.mcmc_chains = mcsamples.loadMCSamples(self.chains_dir+self.chain_root, settings=self.mcmc_chain_settings)
            self.covmat_file = self.chains_dir+self.info_root+'.covmat'
        except OSError:
            return False 
        
        if read_all_chains==True:
            chain_root_list = glob('*__*.txt')
            print("check_mcmc_chains: Reading all chains:")
            for chain_x in chain_root_list:
                print(chain_x)
            try:
                self.mcmc_chains.readChains(chain_root_list)
            except OSError:
                return False 
        
        return True
        
    def run_mcmc(self, resume=False, N_steps=30000):
        """
        Run or resume MCMC chains 
        Requires the folder chains_dir to already be popualted with a log.param file 
        
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
            
            
#         if resume==False:
        run(run_command, shell=True)
#         else:
#             run(run_command+"  -r", shell=True)
            
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
        
        /!\ UNTESTED FUNCTION 
        
        :return: True once finished 
        """
        if not self.check_mcmc_chains(read_all_chains=True):
            self.run_mcmc()
        while not self.check_mcmc_convergence():
            run("mpirun -np 1 MontePython.py info "+self.chains_dir+" --keep-non-markovian --noplot --want-covmat", shell=True)
            self.run_mcmc(resume=True, N_steps=50000)
            self.check_mcmc_chains(read_all_chains=True)
        return True

    def check_global_min(self, mcmc_chains=None):
        """
        Check for .bestfit file. This does not necessarily indicate a global minimum run!!! 
        It only indicates that there exists a file storing the MAP. 
        
        :mcmc_chains: getdist MCSamples instance 
        
        :return: True if global minimum was run and relevant files are accesible. Else False 
        """
        if mcmc_chains==None:
            mcmc_chains=self.mcmc_chains
            
        try:
            np.loadtxt(self.chains_dir+self.info_root+'.bestfit')
            print("check_global_min: Found global minimizer with file name "+self.info_root)
            return True
        except OSError:
            try:
                new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
                run("MontePython.py info "+self.chains_dir+" --keep-non-markovian --noplot")
                np.loadtxt(self.chains_dir+new_info_root+'.bestfit')
                self.info_root = new_info_root
                print("check_global_min: Found global minimizer with file name "+self.info_root)
                return True 
            except OSError:
                print("check_global_min: Cannot run MP info for global minimum. Something went wrong. ")
                return False 
        
    def global_min(self, run_glob_min=True):
        """
        Check global minizer, run if wanted (default True), then write if not already written 

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

        :return: global maximum lkl dictionary 
        """
        self.check_global_min()

        if run_glob_min:
            self.run_minimizer()

        param_names, param_ML, MLs = self.read_minimum(extension='')
        self.global_ML = deepcopy(MLs)
        self.param_order = param_names

        try:
            self.match_param_names(self.param_order)
        except FileNotFoundError:
            extension = '_lkl_profile.txt' # TK change lkl prof file names to include prof param!! 
    #             extension = '_'+self.prof_param+'_lkl_profile.txt' # new file names to include prof param 
            extension = self.pn_ext(extension)

            print("File not found. Starting a new file now: " + self.chains_dir + self.info_root + extension + '\n') 
            with open(self.chains_dir + self.info_root + extension, 'w') as lkl_txt: 
                lkl_txt.write("#")
                for param_recorded in self.param_order:
                    lkl_txt.write("\t %s" % param_recorded)
                lkl_txt.write("\n")

        extension = '_lkl_profile.txt' # TK change lkl prof file names to include prof param!! 
    #         extension = '_'+self.prof_param+'_lkl_profile.txt' # new file names to include prof param 
        extension = self.pn_ext(extension)

        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) 

        if lkl_prof_table.shape!=(0,):
            if not self.match_param_line(self.global_ML, loc=0):
                print("Something went wrong. The first line of the lkl_profile.txt file which should be global ML does not match the global ML in file \n"
                     +self.chains_dir + self.info_root + '.bestfit') 
                raise FileExistsError
        else: 
            self.write_MLs()

        return self.global_ML
        
        
    def pn_ext(self, extension):
        """
        Prefix the file extension string input with the sign of the profile lkl parameter increment to track files correctly. 
        
        :extension: A string of the file name extension without the sign of the prof_parameter 
        :return: String of extension prefixed with the sign of the increment of profile lkl parameter 
        """
        if len(extension)>0:
            if self.prof_incr > 0:
                extension = '_p'+extension
            if self.prof_incr < 0:
                extension = '_n'+extension
        return extension
        
    def read_minimum(self, extension='_lkl_prof'):
        """
        Read minimum file and save parameter names list, parameter values list and MLs dictionary 
        
        :extension: The extension of the life type being read in. Leave this as is, the rest of the code assumes the same naming conventions. 
        
        :return: List of parameter names, list of parameter ML values, dictionary of {'param_names': param_ML_value}
        """
        extension=self.pn_ext(extension)
        
        param_ML = np.loadtxt(self.chains_dir + self.info_root + extension + '.bestfit')

        param_names = read_header_as_list(self.chains_dir + self.info_root + extension + '.bestfit')

        MLs = dict(zip(param_names, param_ML))

        with open(self.chains_dir + self.info_root + extension + '.log') as log_file:
            last_line = log_file.readlines()[-1]
            neg_logLike = float(last_line.split(":")[-1])

        MLs['-logLike'] = neg_logLike
        param_names = np.append(param_names, '-logLike')
        param_ML = np.append(param_ML, MLs['-logLike'])
        
        self.MLs = MLs
        
        return param_names, param_ML, MLs
    
    def read_lkl_output(self, extension='_lkl_profile.txt', loc=-1):
        """
        Read (default = last) line of lkl prof output file into list
        
        :extension: Leave this alone, thank you. 
        :loc: integer location of line in file to read. Default is last line 
        
        TK: SHOULD THIS BE PROMOTED TO A DICT?? 
        Already have read_header func to get the param names 
        
        :return: List of parameters
        """

        extension=self.pn_ext(extension)

        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) # change 
        try:
            lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
            lkl_prof_table = lkl_prof_table[loc, :]
        except IndexError:
            pass
        return lkl_prof_table
    
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
        extension=self.pn_ext(extension)
        
        with open(self.chains_dir + self.info_root + extension, 'a') as lkl_txt: 
            for param in self.param_order:
                lkl_txt.write("\t %s" % str(MLs[param]))
            lkl_txt.write("\n")
        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) 
        return lkl_prof_table.shape
    
    
    def match_param_names(self, param_names, extension='_lkl_profile.txt'):
        """
        Check that param names match in target file and MLs dictionary
        
        TK: THIS DOESN'T NEED TO MATCH THE ORDER THOUGH!! 
        WE CAN KEEP THE ORDER AS THE ONE READ IN FROM THE GLOBAL BF IF THE LKL PROF FILE IS NOT YET CREATED
        OR THE ORDER OF THE LKL PROF TXT FILE IF IT DOES EXIST 
        
        :param_names: List of param_names to check against the file 
        :extension: Leave it alone, thank you. 
        
        :return: True if param_names match, else False
        """
        
        extension=self.pn_ext(extension)
        
        params_recorded = read_header_as_list(self.chains_dir + self.info_root + extension)
        
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
            print("\nmatch_param_names: Error: existing file found at \n" + self.chains_dir + self.info_root + extension 
                 + "\nbut parameters do not match expected.")
            raise FileExistsError
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

        extension=self.pn_ext(extension)

        if param_names==None:
            param_names=self.param_order

        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) 
        if lkl_prof_table.size==0:
            print("match_param_line: File empty ")
            return False
        else: 
            try:
                lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
                if False in [lkl_prof_table[loc, param_names.index(param)] == MLs[param] for param in param_names]:
                    return False
                else:
                    return True 
            except IndexError:
                print("match_param_line: Only one entry in file, checking that entry ")
                if False in [lkl_prof_table[param_names.index(param)] == MLs[param] for param in param_names]:
                    return False 
                else:
                    return True               


    def increment_update_yaml(self, MLs, lkl_pro_yaml, yaml_ext = '_lkl_prof'):
        """
        Update yaml info to next increment 
        
        
        :MLs: dictionary of {'param_name': ML_value }
        :lkl_pro_yaml: dictionary of yaml info for running lkl profile incremental minimizer 
        :yaml_ext: Leave it alone. 
        
        :return: dictionary for yaml info of profile lkl parameter, including incremented value and latex info
        """
        yaml_ext=self.pn_ext(yaml_ext)
        
        # update profile lkl param 
        latex_info = lkl_pro_yaml['params'][self.prof_param]['latex']
        lkl_pro_yaml['params'][self.prof_param] = {'value': MLs[self.prof_param]+self.prof_incr, 'latex': latex_info}
        lkl_pro_yaml['output'] = self.info_root + yaml_ext
        # update all other independent parameters 
        for param in lkl_pro_yaml['params']:
            if 'prior' in lkl_pro_yaml['params'][param]:
                lkl_pro_yaml['params'][param]['ref'] = MLs[param]
        # dump yaml to file for running 
        with open(self.chains_dir+self.info_root+yaml_ext+'.minimize.input.yaml', 'w') as yaml_file:
            dump(lkl_pro_yaml, yaml_file, default_flow_style=False)    
        return lkl_pro_yaml['params'][self.prof_param]
    
    def run_minimizer(self, min_folder='', prev_bf=None):
        """
        Run minimizer as described in 2107.10291, by incrementally running a finer MCMC 
        with more discrening lklfactor that increases preference for moving towards higher likelihoods 
        and taking smaller jumps between points so we sample a finer grid in parameter space. 
        This can be modified as wanted, changing step sizes per rung, lkl factor and jumping factors. 
        Default:
        N_steps = 30000, 10000, 10000
        lklfactor = 10, 200, 1000
        jumping factor (-f) = 0.5, 0.1, 0.05

        Requires the folder min_folder to already be popualted with a log.param file 

        :min_folder: Folder to run minimizer in 
        :prev_bf: Starting best-fit file. 
                This is the MAP of the MCMC chains for the global bf run, 
                and the previous point for the lkl prof. 

        :return: True
        """
        
        print('!!!!!!!!!!!!!!!!!')
        print("reduced steps per rung for testing. Fix before doing actual runs ")
        print('!!!!!!!!!!!!!!!!!')
        

        # Prep the command 
        if not min_folder:
            min_folder = '.'
        elif min_folder[-1] == '/':
            min_folder = min_folder[:-1]

        if not prev_bf:
            prev_bf = self.info_root
        elif '.bestfit' in prev_bf:
            prev_bf = prev_bf[:-8]

        ## First rung 

        # MCMC
        run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=100, 
            f = 0.5, 
            lkl = 10
        )
        run(run_command, shell=True)
        # analyse 
        run_command = "mpirun -np 1 MontePython.py info {folder} --keep-non-markovian --noplot".format(
            folder=self.chains_dir+min_folder+'/'
        )
        run(run_command, shell=True)
        # print output 
        if min_folder=='.':
            prev_bf = [x for x in str(os.getcwd()).split('/') if x][-1]
            # switch to current directory as bf root, ensures that we're using the most recent file 
        else:
            prev_bf = min_folder+'/'+min_folder 
            # switch to most recently produced bf file in the minimizer directory as bf root 
        new_min_point = get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
        print("-----> After first minimizer rung, -logL minimized to  {logL}".format(logL=new_min_point['-logLike']))

        ## Second rung 

        # MCMC
        run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=100, 
            f = 0.1, 
            lkl = 200
        )
        run(run_command, shell=True)
        # analyse 
        run_command = "mpirun -np 1 MontePython.py info {folder} --keep-non-markovian --noplot".format(
            folder=self.chains_dir+min_folder+'/'
        )
        run(run_command, shell=True)
        # print output 
        new_min_point = get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
        print("-----> After second minimizer rung, -logL minimized to {logL}".format(logL=new_min_point['-logLike']))

        ## Third rung 

        # MCMC
        run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=100, 
            f = 0.05, 
            lkl = 1000
        )
        run(run_command, shell=True)
        # analyse 
        run_command = "mpirun -np 1 MontePython.py info {folder} --keep-non-markovian --noplot".format(
            folder=self.chains_dir+min_folder+'/'
        )
        run(run_command, shell=True)
        # print output 
        new_min_point = get_MP_bf_dict(self.chains_dir+prev_bf+'.bestfit')
        print("-----> After third minimizer rung, -logL minimized to  {logL}".format(logL=new_min_point['-logLike']))

        return True
    
    
    def init_lkl_prof(self):
        """
        Initialise profile lkl yaml:
        1) copy the global log.param into a profile lkl folder if not already done
        2) read the last line of the lkl output file and set that as the current MLs dictionary, as self.MLs. 
        this updates the location in prof_param that we're at for running prof lkls 
        3) check that this last point is the same as the prev_bf file 
        4) check that the lkl prof log.param matches this point???
        
        :return: the current lkl prof_param value 
        """
        
        global_lp = self.chains_dir+'log.param'
        lkl_lp = "lkl_prof_"+self.prof_param
        lkl_lp = self.pn_ext(lkl_lp)
        run("mkdir "+lkl_lp, shell=True)
        
        copy_log_param = "cp {global_lp} {lkl_lp}/".format(global_lp=global_lp, lkl_lp=lkl_lp)
        run(copy_log_param, shell=True)

        ############
        
        extension = '_lkl_prof'
        try:
            lkl_pro_yaml = yaml_load_file(self.chains_dir+self.info_root+self.pn_ext(extension)+'.minimize.input.yaml')
        except FileNotFoundError:
            run("cp "+self.chains_dir+self.info_root+'.minimize.updated.yaml'+" "
                    +self.chains_dir+self.info_root+self.pn_ext(extension)+'.minimize.input.yaml', shell=True)
            lkl_pro_yaml = yaml_load_file(self.chains_dir+self.info_root+self.pn_ext(extension)+'.minimize.input.yaml')
        lkl_pro_yaml['sampler'] = self.minimizer_settings
        lkl_pro_yaml['sampler']['minimize']['covmat'] = self.covmat_file

        param_ML = self.read_lkl_output()
        self.MLs = dict(zip(self.param_names, param_ML))
        
        self.lkl_pro_yaml = deepcopy(lkl_pro_yaml)
        
        return self.lkl_pro_yaml
        
    def run_lkl_prof(self, time_mins=False):
        """
        Run the likelihood profile loop. 
        Initialise time-keeping file if wanted. 
        
        While we are within the bounds of the profile param we want to explore: 
        1) check if the point we are currently at i.e. param_ML and MLs, matches the last entry in the lkl_prof table.
            - if it does, the last minimum was run and saved successfully.
            - if not, check if a minimum file exists. 
                - if it does, read it in and save it in the lkl prof txt. minimum run successfully. 
                - if not, this happens when we have updated the yaml but the minimizer didn't finish. 
                  Run the yaml again without updating. 
        2) check if minimum was run and saved. 
            - if yes, update the yaml and increment the prof lkl param, 
              update all other params to new values from current ML. 
              Assign the MLs values for the independent params in the yaml as new reference starting points. 
        3) run the minimizer 
        4) save minimizer output 

        :time_mins: boolean for whether you want to time each minimiser increment or not 
        
        :return: the value of the profile lkl parameter at the end of this loop 
        """
        if time_mins == True:
            time_extension = '_time_stamps.txt'
            time_extension = self.pn_ext(time_extension)
            with open(self.chains_dir + self.info_root + time_extension, 'w') as lkl_txt:
                lkl_txt.write("#")
                lkl_txt.write(" %s \t step_size \t minimizer_time " % self.prof_param)
                lkl_txt.write("\n")

        extension = '_lkl_prof'
        extension = self.pn_ext(extension)

        MLs = deepcopy(self.MLs)
        lkl_pro_yaml = deepcopy(self.lkl_pro_yaml)

        while ((MLs[self.prof_param] < self.prof_max) and (MLs[self.prof_param] > self.prof_min)):
            last_entry_matches_current_params = self.match_param_line(MLs)
            if last_entry_matches_current_params:
                run('rm '+self.chains_dir + self.info_root + extension + '.minimum*', shell=True)
                minimum_successfully_run_and_saved = True
            else:
                try:
                    param_names, param_ML, MLs = self.read_minimum()
                    self.write_MLs(MLs)
                    run('rm '+self.chains_dir + self.info_root + extension + '.minimum*', shell=True)
                    minimum_successfully_run_and_saved = True 
                    print("-----> Minimizer run successfully for "+self.prof_param+" = "+str(MLs[self.prof_param]))
                except OSError:
                    minimum_successfully_run_and_saved = False
                    print("-----> Minimizer not run for "+self.prof_param+" = "+str(MLs[self.prof_param]))
                    print("       Rerunning this point")

            if minimum_successfully_run_and_saved:
                self.increment_update_yaml(MLs, lkl_pro_yaml)
                run('rm '+self.chains_dir + self.info_root + extension + '.minimize.updated.yaml', shell=True)

            time_start = time()

            self.run_minimizer()

            time_end = time()
            time_taken = time_end - time_start

            if time_mins == True:
                with open(self.chains_dir + self.info_root + self.pn_ext(time_extension), 'a') as lkl_txt:
                    lkl_txt.write("{:.4g} \t {:.2g} \t {:.2f} \n".format(lkl_pro_yaml['params'][self.prof_param]['value'], 
                                                                         self.prof_incr, time_taken))
                print("       Time taken for minimizer = {:.2f}".format(time_taken))

            param_names, param_ML, MLs = self.read_minimum()

    #         prof_incr *= 2. # Readjust prof lkl increment if wanted by copying this function and adding such a line 

        # outside loop now 
        last_entry_matches_current_params = self.match_param_line(MLs)
        if not last_entry_matches_current_params:
            param_names, param_ML, MLs = self.read_minimum()
            self.write_MLs(MLs)
            print("-----> Minimizer run successfully for "+self.prof_param+" = "+str(MLs[self.prof_param]))
        
        return MLs[self.prof_param]
    
    
    def full_lkl_prof_array(self):
        """
        Combine positive and negative increment files into one array 
        
        :return: full likelihood profile array 
        """
        try:
            all_MLs_p = np.loadtxt(self.chains_dir+self.info_root+'_p_lkl_profile.txt')
            pos_file = True
        except OSError:
            pos_file = False
        try:
            all_MLs_n = np.loadtxt(self.chains_dir+self.info_root+'_n_lkl_profile.txt')
            if pos_file==True:
                all_MLs = np.concatenate( (np.flip(all_MLs_n, 0),all_MLs_p) )
            else:
                all_MLs = np.flip(all_MLs_n, 0)
        except OSError:
            all_MLs = all_MLs_p

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

        for param_num in range(len(self.param_order)):
            full_prof_dict[self.param_order[param_num]] = full_lkl_prof_array[:,param_num]

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
            
