from MP_classes_lkl_prof_functions import *

ede = lkl_prof('/home2/karwal/lkl_prof/VP_rec_multimodal_Feb28/', 'planck_fede', 'fraction_axion_ac')
# if you don't have MCMC chains, ensure that the info_root (the second argument) is the same as the folder name, as shown in this example.
# also ensure that the .bestfit and .covmat files you have are named <info_root>.bestfit and <info_root>.covmat

ede.prof_incr = 0.01 # run two separate jobs with both a + increment and a - increment
ede.prof_max = 0.14
ede.prof_min = 0.01
ede.processes = 2


jump_fac_lists = [
                    [1, 0.5, 0.2, 0.1, 0.05],
                    [1.5, 0.5, 0.2, 0.1, 0.05],
                    [1, 0.5, 0.2, 0.1, 0.05],
                    [1.5, 0.5, 0.2, 0.1, 0.05],
]

lkl_fac_lists = [
                    [2, 3, 10, 200, 1000],
                    [2, 3, 10, 200, 1000],
                    [3, 5, 10, 200, 1000],
                    [3, 5, 10, 200, 1000],
]

for i in range(len(jump_fac_lists)):
    ede.jump_fac = jump_fac_lists[i]
    ede.lkl_fac = lkl_fac_lists[i]

    ede.info_root = 'planck_fede'
    ede.check_global_min()

    # Run and store this bf
    ede.global_min(run_glob_min=True,N_min_steps=3000)

    # Copy output to identifying files
    info_string = "j_{:}_lkl_{:}".format(ede.jump_fac[0], ede.lkl_fac[0])
    copy_bf = "cp "+ede.chains_dir+"global_min/global_min.bestfit "+ede.chains_dir+"min_"+info_string+".bestfit"
    run(copy_bf, shell=True)
    copy_bf = "cp "+ede.chains_dir+"global_min/global_min.log "+ede.chains_dir+"min_"+info_string+".log"
    run(copy_bf, shell=True)

    # Scrub files to restart runs
    remove_files = "rm VP_rec_multimodal_Feb28* global_min/*"
    run(remove_files, shell=True)
