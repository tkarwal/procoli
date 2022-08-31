from getdist import mcsamples, plots, chains
from getdist.mcsamples import MCSamplesError
import numpy as np
from subprocess import run
import os
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
        self.mcmc_chains = None
        
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
            
        :read_all_chains: boolean for whether to read all the chains in the chains directory 
        
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
            np.loadtxt(self.chains_dir+self.info_root+'.bestfit')
            print("check_global_min: Found minimum with file name "+self.info_root)
            
            new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
            if self.info_root != new_info_root:
                reset_info_root = 'cp '+self.info_root+'.bestfit '+new_info_root+'.bestfit '
                run(reset_info_root, shell=True)
                reset_info_root = 'cp '+self.info_root+'.log '+new_info_root+'.log '
                run(reset_info_root, shell=True)
                self.info_root = new_info_root
                
            return True
        except OSError:
            try:
                new_info_root = [x for x in self.chains_dir.split('/') if x][-1]
                run("mpirun -np 1 MontePython.py info "+self.chains_dir+" --keep-non-markovian --noplot --want-covmat", shell=True)
                np.loadtxt(self.chains_dir+new_info_root+'.bestfit')
                self.info_root = new_info_root
                print("check_global_min: Found minimum with file name "+self.info_root)
                return True 
            except OSError:
                print("check_global_min: Cannot run MP info for global minimum. Something went wrong. ")
                return False 
        
    def global_min(self, run_glob_min=False, N_min_steps=3000):
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
            run("mkdir global_min", shell=True)
            run("cp log.param global_min/log.param", shell=True)
            
            self.run_minimizer(min_folder='global_min', N_steps=N_min_steps)

            run("cp global_min/global_min.bestfit "+self.info_root+".bestfit", shell=True)
            run("cp global_min/global_min.log "+self.info_root+".log", shell=True)

        param_names, param_ML, MLs = self.read_minimum(extension='')
        self.global_ML = deepcopy(MLs)
        self.param_order = param_names.tolist()

        extension = '_lkl_profile.txt' 
        extension = self.pn_ext(extension)
        
        try:
            self.match_param_names(self.param_order)
        except FileNotFoundError:
            print("global_min: File not found. Starting a new file now: " + self.chains_dir + self.info_root + extension + '\n') 
            with open(self.chains_dir + self.info_root + extension, 'w') as lkl_txt: 
                lkl_txt.write("#")
                for param_recorded in self.param_order:
                    lkl_txt.write("\t %s" % param_recorded)
                lkl_txt.write("\n")

        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) 

        if lkl_prof_table.shape!=(0,):
            if not self.match_param_line(self.global_ML, loc=0):
                print("global_min: Something went wrong. The first line of the lkl_profile.txt file which should be global ML does not match the global ML in file \n"
                     +self.chains_dir + self.info_root + '.bestfit') 
                raise FileExistsError
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
        
        :return: Dict of parameters
        """

        extension=self.pn_ext(extension)

        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) 
        try:
            lkl_prof_table.shape[1] # check that lkl_prof_table has multiple rows
            lkl_prof_table = lkl_prof_table[loc, :]
        except IndexError:
            pass
        
        self.param_names = read_header_as_list(self.chains_dir + self.info_root + extension)
        
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
        Matches combination, not permutation. 
        
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

#         print("match_param_line: checking file {file}".format(file=self.chains_dir + self.info_root + extension) )
        lkl_prof_table = np.loadtxt(self.chains_dir + self.info_root + extension) 
        
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
    
    
    def run_minimizer(self, min_folder="lkl_prof", prev_bf=None, N_steps=5000):
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

        ##### First rung #####

        # MCMC
        run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=N_steps, 
            f = 0.2, 
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
        print("\n\n------------------> After first minimizer rung, -logL minimized to  {logL} \n\n".format(
            logL=new_min_point['-logLike']))

        ##### Second rung #####

        # MCMC
        run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=N_steps, 
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
        print("\n\n------------------> After second minimizer rung, -logL minimized to {logL} \n\n".format(
            logL=new_min_point['-logLike']))

        ##### Third rung #####

        # MCMC
        run_command = "mpirun -np {procs} MontePython.py run -p {param} -o {output} -b {bf} -c {covmat} -N {steps} -f {f} --lklfactor {lkl}".format(
            procs=self.processes,
            param=self.chains_dir+min_folder+'/log.param', 
            output=self.chains_dir+min_folder+'/',
            bf=self.chains_dir+prev_bf+'.bestfit', 
            covmat=self.chains_dir+self.info_root+'.covmat',
            steps=N_steps, 
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
        print("\n\n ------------------> After third minimizer rung, -logL minimized to  {logL} \n\n".format(
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
        
        global_lp = self.chains_dir+'log.param'
        lkl_dir += self.pn_ext('/')
        run("mkdir "+lkl_dir, shell=True)

        copy_log_param = "cp {global_lp} {lkl_dir}".format(global_lp=global_lp, lkl_dir=lkl_dir)
        run(copy_log_param, shell=True)
                
        try: 
            self.read_minimum()
        except OSError:
            # the lkl prof bf and lof files don't exist
            # copy global bf 
            copy_global_bf_to_lkl_prof_bf = "cp "+self.info_root+".bestfit "+self.info_root+self.pn_ext("_lkl_prof")+".bestfit"
            run(copy_global_bf_to_lkl_prof_bf, shell=True)
            # copy global log 
            copy_global_log_to_lkl_prof_log = "cp "+self.info_root+".log "+self.info_root+self.pn_ext("_lkl_prof")+".log"
            run(copy_global_log_to_lkl_prof_log, shell=True)
            # now this should work 
            self.read_minimum()
        
        # # /!\ Used to be initialised to last entry of lkl prof txt file 
        # self.MLs = self.read_lkl_output()
        # # Copy last lkl profile txt point into the bestfit file:
        # lkl_prof_header = read_header_as_list(self.info_root+self.pn_ext('_lkl_profile.txt'))
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
        lkl_dir += self.pn_ext('/')
        lkl_lp = lkl_dir+"log.param"

        with open(lkl_lp, 'r') as f:
            lkl_lp_lines = f.readlines()

        line_modified = False
        lp_prof_param_string = "'"+self.prof_param+"'"
        with open(lkl_lp, 'w') as f:
            for line in lkl_lp_lines:
                if lp_prof_param_string in line:
                    # print("Original: \t"+line)
                    prof_param_lp_line = line.split("=")
                    prof_param_lp_data = prof_param_lp_line[1].split(",")
                    
                    updated_prof_param = self.MLs[self.prof_param]+self.prof_incr
                    prof_param_lp_data[0] = "[" + str(updated_prof_param)
                    prof_param_lp_data[3] = "0." 
                
                    prof_param_lp_line = prof_param_lp_line[0] + " = " + ",".join(prof_param_lp_data)
                    line = prof_param_lp_line
                    # print("Modified: \t"+line)
                    f.write(line)
                    line_modified = True
                else:
                    f.write(line)

        if line_modified == False:
            print("Error: increment_update_logparam: could not find line with profile lkl parameter {prof_param} in log.param at {lp_file}".format(
                prof_param=self.prof_param, lp_file=lkl_lp))
            raise KeyError

        return updated_prof_param, prof_param_lp_line
    
    
    def get_prof_param_value_from_lp(self, lp_dir = "lkl_prof"):
        """
        Get current value of the prof lkl parameter from the lop param file 
        
        :lp_dir: directory of the log.param file to read. Default set for using function internally. 
        
        :return: 'mean' of prof lkl parameter in the log.param as float 
        """
        if lp_dir:
            lp_dir += self.pn_ext('/')
        lp_file = lp_dir+"log.param"

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
        extension=self.pn_ext(extension)

        min_output_bf = self.chains_dir + "lkl_prof" + self.pn_ext('/') + "lkl_prof" + self.pn_ext('/')[:-1] + ".bestfit"

        with open(min_output_bf, 'r') as f:
            bf_lines = f.readlines()

        bf_lines[0] = bf_lines[0][:-1]+',        '+self.prof_param+'\n'
        bf_lines[1] = bf_lines[1][:-1]+" "+str(self.current_prof_param)+'\n'

        save_output_bf = self.chains_dir + self.info_root + extension + '.bestfit'

        with open(save_output_bf, 'w') as f:
            for line in bf_lines:
                f.write(line)
                
        copy_log_to_main_folder = "cp lkl_prof" + self.pn_ext('/') + "lkl_prof" + self.pn_ext('/')[:-1] + ".log "+self.info_root+extension+".log"
        run(copy_log_to_main_folder, shell=True)

        return self.current_prof_param
        
    def run_lkl_prof(self, time_mins=False, N_min_steps=5000):
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
            time_extension = '_time_stamps.txt'
            time_extension = self.pn_ext(time_extension)
            with open(self.chains_dir + self.info_root + time_extension, 'a') as lkl_txt:
                lkl_txt.write("#")
                lkl_txt.write(" %s \t step_size \t minimizer_time " % self.prof_param)
                lkl_txt.write("\n")

        extension = '_lkl_prof'
        extension = self.pn_ext(extension)

        while ((self.MLs[self.prof_param] < self.prof_max) and (self.MLs[self.prof_param] > self.prof_min)):
            print("run_lkl_prof: -----> Running point {param} = {value}".format(param=self.prof_param, value=self.MLs[self.prof_param]))
            last_entry_matches_current_params = self.match_param_line(self.MLs)
            if last_entry_matches_current_params:
                minimum_successfully_run_and_saved = True
            else:
                param_names, param_ML, self.MLs = self.read_minimum()
                # read_min updates self.MLs 
                self.write_MLs(self.MLs)
                minimum_successfully_run_and_saved = True 
                print("run_lkl_prof: -----> Minimizer run successfully for "+self.prof_param+" = "+str(self.MLs[self.prof_param]))

            if minimum_successfully_run_and_saved:
                self.current_prof_param, prof_param_string_in_logparam = self.increment_update_logparam()
                run('rm '+self.chains_dir + "lkl_prof" + self.pn_ext('/') + '20*', shell=True)
                run('rm '+self.chains_dir + "lkl_prof" + self.pn_ext('/') + "lkl_prof" + self.pn_ext('/')[:-1] + "*", 
                        shell=True)    
            else:
                self.current_prof_param = self.get_prof_param_value_from_lp()
                run('rm '+self.chains_dir + "lkl_prof" + self.pn_ext('/') + '20*', shell=True)
                run('rm '+self.chains_dir + "lkl_prof" + self.pn_ext('/') + "lkl_prof" + self.pn_ext('/')[:-1] + "*", 
                        shell=True)    

            time_start = time()

            self.run_minimizer(prev_bf=self.info_root+self.pn_ext("_lkl_prof"), 
                               min_folder="lkl_prof" + self.pn_ext('/')[:-1],
                               N_steps=N_min_steps)
            self.update_and_save_min_output() 

            time_end = time()
            time_taken = time_end - time_start
            
            if time_mins == True:
                with open(self.chains_dir + self.info_root + time_extension, 'a') as lkl_txt:
                    lkl_txt.write("{:.4g} \t {:.2g} \t {:.2f} \n".format(self.current_prof_param, 
                                                                         self.prof_incr, time_taken))
                print("run_lkl_prof:        Time taken for minimizer = {:.2f}".format(time_taken))

            param_names, param_ML, self.MLs = self.read_minimum()
            minimum_successfully_run_and_saved = False


            # prof_incr *= 2. # Readjust prof lkl increment if wanted by copying this function and adding such a line 

        # outside loop now 
        last_entry_matches_current_params = self.match_param_line(self.MLs)
        if not last_entry_matches_current_params:
            param_names, param_ML, self.MLs = self.read_minimum()
            self.write_MLs(self.MLs)
            print("run_lkl_prof: -----> Minimizer run successfully for "+self.prof_param+" = "+str(self.MLs[self.prof_param]))
        
        return self.MLs[self.prof_param]
    
    
    def full_lkl_prof_array(self):
        """
        Combine positive and negative increment files into one array 
        But first check that they have the same param order. 

        :return: full likelihood profile array 
        """
        pos_filename = self.chains_dir+self.info_root+'_+'+self.prof_param+'_lkl_profile.txt'
        neg_filename = self.chains_dir+self.info_root+'_-'+self.prof_param+'_lkl_profile.txt'

        
            
        try:
            pos_header = read_header_as_list(pos_filename)
            all_MLs_p = np.loadtxt(pos_filename)
            pos_file = True
        except FileNotFoundError:
            pos_file = False
        try:
            neg_header = read_header_as_list(neg_filename)
            all_MLs_n = np.loadtxt(neg_filename)
            if pos_file==True:
                if pos_header==neg_header:
                    all_MLs = np.concatenate( (np.flip(all_MLs_n, 0),all_MLs_p) )
                else:
                    print("full_lkl_prof_array: the positive and negative files either have different parameters \
                            or have them in different orders. \
                            \nEither way, this function cannot correctly combine them. ")
                    return 0
            else:
                all_MLs = np.flip(all_MLs_n, 0)
        except FileNotFoundError:
            if pos_file == True:
                all_MLs = all_MLs_p
            else:
                print("full_lkl_prof_array: could not find files \n{pos} \n{neg} ".format(pos=pos_filename, neg=neg_filename))
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
            
