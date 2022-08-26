from MP_classes_lkl_prof_functions import *

lcdm = lkl_prof('/home2/karwal/lkl_prof/lcdm_mp_mcmc/', 'lcdm_mp_mcmc', 'H0')

lcdm.prof_incr = -0.1 # run two separate jobs with both a + increment and a - increment
lcdm.prof_max = 72
lcdm.prof_min = 65.
lcdm.processes = 6

lcdm.check_mcmc_chains(read_all_chains=True)

lcdm.check_mcmc_convergence()

lcdm.global_min(run_glob_min=True)

print(lcdm.global_ML)

lcdm.init_lkl_prof()

lcdm.run_lkl_prof(time_mins=True,N_min_steps=3000)