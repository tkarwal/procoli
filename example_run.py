from procoli import lkl_prof

profile = lkl_prof(
  chains_dir='directory/with/chains/or/bestfit/and/covmat/files/', 
  prof_param='H0', # any parameter varied in the log.param 
                   # that MontePython recognises 
  # info_root= '', # bf_and_covmat_file_names_if_different_from_directory_name, 
                   # and if no chains are provided 
                   # ensure that the .bestfit and .covmat files you have 
                   # are named <info_root>.bestfit and <info_root>.covmat 
)

# Set the other profile parameters either through the class above 
# or as shown below 
profile.prof_max = 72
profile.prof_min = 65.
profile.processes = 6

profile.prof_incr = 0.1 # run two separate jobs with 
                        # both a + increment and a - increment

# Settings for global best fit search 
# The below are the defaults 
profile.set_global_jump_fac([1, 0.8, 0.5, 0.2, 0.1, 0.05])
profile.set_global_temp([0.3333, 0.25, 0.2, 0.1, 0.005, 0.001])

# Check the global best fit, run if necessary, 
# record this point in the likelihood profile txt file 
profile.global_min(
  # run_glob_min=True, # if you don't pass this parameter, the code will 
                       # automatically decide whether to run the global minimizer. 
                       # It will run it if no global_min/global_min.bestfit 
                       # file exists or if the corresponding .log file has a worse
                       # -logLike than the info_root.log file 
  N_min_steps=4000 # default number of steps taken by optimizer 
                   # for each step in the simulated annealing ladder 
) 

# Print as a check 
print("Global minimum: ")
print(profile.global_ML)

# Initialise the likelihood profile 
profile.init_lkl_prof()

# Settings for profile likelihood optimizations 
# The below are defaults 
profile.set_jump_fac([0.15, 0.1, 0.05])
profile.set_temp([0.1, 0.005, 0.001])

# Run the profile likelihood
profile.run_lkl_prof(
  time_mins=True, # time each optimization 
  N_min_steps=3000 # default number of steps taken by optimizer 
                   # for each step in the simulated annealing ladder 
)

# Run the same steps above, but with a negative increment for the profile parameter 
profile.prof_incr = -0.1

profile.global_min(
  N_min_steps=4000 
) 

profile.init_lkl_prof()

profile.run_lkl_prof(
  time_mins=True, 
  N_min_steps=3000
)
