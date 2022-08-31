from MP_classes_lkl_prof_functions import *

lcdm = lkl_prof('/home2/karwal/lkl_prof/lcdm_mp_mcmc/', 'lcdm_mp_mcmc', 'H0')
# if you don't have MCMC chains, ensure that the info_root (the second argument) is the same as the folder name, as shown in this example. 
# also ensure that the .bestfit and .covmat files you have are named <info_root>.bestfit and <info_root>.covmat 

lcdm.prof_incr = -0.1 # run two separate jobs with both a + increment and a - increment
lcdm.prof_max = 72
lcdm.prof_min = 65.
lcdm.processes = 6

lcdm.check_mcmc_chains(read_all_chains=True) # remove if no chains exist 
lcdm.check_mcmc_convergence() # remove if no chains exist 

lcdm.global_min(run_glob_min=True, N_min_steps=100) 
# it is important to change this to False for restarted runs that already have a global min. 
# Otherwise the old global min may get overwritten 

print("Global minimum:")
print(lcdm.global_ML)

lcdm.init_lkl_prof()

lcdm.run_lkl_prof(time_mins=True,N_min_steps=3000)