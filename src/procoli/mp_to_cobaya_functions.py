import numpy as np

from mp_procoli_functions import *
# TODO fix imports, needs yaml_load_file from cobaya
# TODO only import read_header and get_mp_bf_dict from mp_procoli_functions

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

    :return: dictionary of keys = MP param names and values = best fit point
    """
    MP_param_values = np.loadtxt(MP_bf_file)
    
    MP_param_names = read_header_as_list(MP_bf_file)
    
    MP_bf = dict(zip(MP_param_names, MP_param_values))
    
    return MP_bf


MP_to_Cobaya_names = {
    # # Only contains params that are different between the two. 
    'ln10^{10}A_s' : 'logA',
    'scf_parameters__1': 'scf_param_1',
    '100*theta_s': 'theta_s_1e2',
}

Cobaya_to_MP_names = {y: x for x, y in MP_to_Cobaya_names.items()}

def Cob_bf_from_MP(MP_bf, MP_to_Cobaya_names=MP_to_Cobaya_names):
    """
    Convert MP best fit dictionary to Cobaya bf dictionary. 
    This just changes the param names from one convention to the other. 
    For param name changes, edit the global dictionary MP_to_Cobaya_names.
    The reverse dictionary is based on that, so just edit one. 

    :MP_bf: dictionary of MP best fit. Maybe produced from the previous function. 

    :return: dictionary of keys = Cobaya param names and values = best fit point 
    """
    cob_bf = {}
    for key in MP_bf:
        if key in MP_to_Cobaya_names:
            cob_bf[MP_to_Cobaya_names[key]] = MP_bf[key]
        else:
            cob_bf[key] = MP_bf[key]
    return cob_bf

def MP_bf_from_Cob(cob_bf, Cobaya_to_MP_names=Cobaya_to_MP_names):
    """
    Convert Cobaya best fit dictionary to MP bf dictionary. 
    This just changes the param names from one convention to the other. 
    For param name changes, edit the global dictionary MP_to_Cobaya_names.
    The reverse dictionary is based on that, so just edit one. 

    :cob_bf: dictionary of Cobaya best fit 

    :return: dictionary of keys = MP param names and values = best fit point 
    """
    MP_bf = {}

    for key in cob_bf:
        if key in Cobaya_to_MP_names:
            MP_bf[Cobaya_to_MP_names[key]] = cob_bf[key]
        else:
            MP_bf[key] = cob_bf[key]

    return MP_bf


def cob_to_MP(param_names, Cobaya_to_MP_names=Cobaya_to_MP_names):
    """
    Convert Cobaya param names list to MP format. 
    This just changes the param names from one convention to the other. 
    For param name changes, edit the global dictionary MP_to_Cobaya_names.
    The reverse dictionary is based on that, so just edit one. 

    :param_names: list of Cobaya convention param names 

    :return: list of MP convention param names 
    """
    new_names = []
    for name in param_names:
        if name in Cobaya_to_MP_names:
            new_names.append(Cobaya_to_MP_names[name])
        else:
            new_names.append(name)
    return new_names

def MP_to_cob(param_names, MP_to_Cobaya_names=MP_to_Cobaya_names):
    """
    Convert MP param names list to Cobaya format. 
    This just changes the param names from one convention to the other. 
    For param name changes, edit the global dictionary MP_to_Cobaya_names.
    The reverse dictionary is based on that, so just edit one. 

    :param_names: list of MP convention param names 

    :return: list of Cobaya convention param names 
    """
    new_names = []
    for name in param_names:
        if name in MP_to_Cobaya_names:
            new_names.append(MP_to_Cobaya_names[name])
        else:
            new_names.append(name)
    return new_names


def MP_covmat_to_cob(MP_covmat_file, cob_yaml_file, output_covmat_file):
    """
    Convert MP covmat to a Cobaya covmat. 
    Specifically, take the MP covmat, and convert it to one useful for Cobaya 
    by retaining only the rows and columns that correspond 
    to current MCMC independent params in the Cobaya reference yaml. 
    This removes derived params and any other params that don't match 
    what is being varied in the MCMC 
    (params that have a 'prior' argument in the yaml). 
    
    :MP_covmat_file: full path to the MP covmat you want to convert 
    :cob_yaml_file: full path to the Cobaya yaml you want to reference. 
                    This should be the xxx.updated.yaml so it includes the nuisance params!! 
    :output_covmat_file: full path to the output file you want to write. 
                            This file should end in 'xxx.covmat'!!
    
    :return: header of the new covmat file written 
    """
    # First get the covmat file and the header 
    covmat = np.loadtxt(MP_covmat_file)
    MP_covmat_params = read_header_as_list(MP_covmat_file)
    
    # Also get cobaya yaml params we want covamt for 
    cob_yaml = yaml_load_file(cob_yaml_file)
    indep_params = [key for key in cob_yaml['params'] if 'prior' in cob_yaml['params'][key]]
    cob_indep_params_MP_names = cob_to_MP(indep_params)
    
    # Using these, which are the indices in the covmat that we want to remove 
    index_to_remove = []
    for param in MP_covmat_params:
        if param not in cob_indep_params_MP_names:
            index_to_remove.append(MP_covmat_params.index(param) )

    # Reverse remove these indices so we don't try to remove the end indices after shortening the array 
    for index in index_to_remove[::-1]:
        covmat = np.delete(covmat, index, 0)
        covmat = np.delete(covmat, index, 1)
        del MP_covmat_params[index]
    
    # Make a Cobaya format header for this new file 
    covmat_header = ' '.join(MP_to_cob(MP_covmat_params) )
    
    # Output the covmat 
    np.savetxt(output_covmat_file, covmat, fmt='%.18e', delimiter=' ', newline='\n', header=covmat_header)
    
    return covmat_header


def Cob_covmat_to_MP(cob_covmat_file, output_covmat_file):
    """
    Convert Cobaya covmat to an MP covmat. 
    
    !!!!! need to check to make sure that MP only reads the rows and columns relevant for its parameters 

    :cob_covmat_file: full path to the MP covmat you want to convert 
    :output_covmat_file: full path to the output file you want to write. 
                            This file should end in 'xxx.covmat'!!
    
    :return: header of the new covmat file written 
    """
    # First get the covmat file and the header 
    covmat = np.loadtxt(cob_covmat_file)
    cob_covmat_params = read_header_as_list(cob_covmat_file)

    # Make a Cobaya format header for this new file 
    covmat_header = ',        '.join(cob_to_MP(cob_covmat_params) )

    # Output the covmat 
    np.savetxt(output_covmat_file, covmat, fmt='%.18e', delimiter='    ', newline='\n', header=covmat_header)
    
    return covmat_header

# - Function below should also be added to classes notebook making test --> self. Or rewrite such that test --> lkl_prof instance. It lets you use an MP covmat for a Cobaya run. Any sampler. 
# - Check if directory/filename.covmat already exists. 
# - If it does, Cobaya will default to that file, so this function should return an error. 
# - Otherwise, grab Cobaya updated yaml file so nuisance parameters are listed
# - Set covmat that yaml points to to be the output of this function
# - For the Cobaya updated yaml file passed, grab the rows and columns of the independent parameters from the MP covmat, converting to the naming and header conventions of Cobaya

# def use_MP_covmat(MP_covmat_file):
#     try:
#         np.loadtxt(test.chains_dir+test.chain_file+'.covmat')
#         print('This function will not work as the file '+test.chains_dir+test.chain_file+'.covmat'+\
#                 ' already exists at this location. \nCobaya defaults to this file instead of any file'+\
#                  ' passed through this function. ')
#         return FileExistsError
#     except OSError:
#         cob_yaml_file = test.chains_dir + test.chain_file + '.updated.yaml'
#         save_covmat_file = test.chains_dir +  'MP_to_cob.covmat'
#         test.covmat_file = save_covmat_file
#         return MP_covmat_to_cob(MP_covmat_file, cob_yaml_file, save_covmat_file)


# #### Longer list of MP names to Cobaya 

# MP_to_Cobaya_names = {
#     # # Commenting out params that are the same between the two 
#     # 'omega_b': 'omega_b',
#     # 'omega_cdm': 'omega_cdm',
#     # 'H0': 'H0',
#     # 'n_s': 'n_s',
#     # 'A_s': 'A_s',
#     'ln10^{10}A_s' : 'logA',
#     # 'tau_reio': 'tau_reio',
#     # 'f_axion_ac': 'f_axion_ac',
#     'scf_parameters__1': 'scf_param_1',
#     # 'log10_axion_ac': 'log10_axion_ac',
    
#     # # Nuisance params shouldn't matter. Should be identical between the two 
#     # 'A_cib_217': 'A_cib_217',
#     # 'xi_sz_cib': 'xi_sz_cib',
#     # 'A_sz': 'A_sz',
#     # 'ps_A_100_100': 'ps_A_100_100',
#     # 'ps_A_143_143': 'ps_A_143_143',
#     # 'ps_A_143_217': 'ps_A_143_217',
#     # 'ps_A_217_217': 'ps_A_217_217',
#     # 'ksz_norm': 'ksz_norm',
#     # 'gal545_A_100': 'gal545_A_100',
#     # 'gal545_A_143': 'gal545_A_143',
#     # 'gal545_A_143_217': 'gal545_A_143_217',
#     # 'gal545_A_217': 'gal545_A_217',
#     # 'galf_TE_A_100': 'galf_TE_A_100',
#     # 'galf_TE_A_100_143': 'galf_TE_A_100_143',
#     # 'galf_TE_A_100_217': 'galf_TE_A_100_217',
#     # 'galf_TE_A_143': 'galf_TE_A_143',
#     # 'galf_TE_A_143_217': 'galf_TE_A_143_217',
#     # 'galf_TE_A_217': 'galf_TE_A_217',
#     # 'calib_100T': 'calib_100T',
#     # 'calib_217T': 'calib_217T',
#     # 'A_planck': 'A_planck',
#     # # 'M': '', # Unclear??? 
    
#     # # Derived params. Not needed for providing bf point, nor for covmats 
#     # 'age': 'age',
#     # 'rs_rec': 'rs_rec',
#     # # '100*theta_s': '', # terrible python name. Unclear if it works. 
#     #                     # If it does, 'theta_s_1e2' could be the Cobaya read in. 
#     #                     # We'll usually be using H0 as input though. 
#     # 'sigma8': 'sigma8',
#     # 'Omega_m': 'Omega_m',
#     # 'Omega_Lambda': 'Omega_Lambda',
#     # 'log10_f_axion': 'log10_f_axion',
#     # 'log10_m_axion': 'log10_m_axion',
#     # 'f_ede': 'f_ede',
#     # 'log10_z_c': 'log10_z_c',
# }
