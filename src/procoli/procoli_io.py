from os import makedirs, remove
from glob import glob
from shutil import copy as file_copy

import numpy as np

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

    MP_bf = dict(zip(MP_param_names, MP_param_values))

    try:
        with open(MP_bf_file[:-8]+'.log') as log_file:
            last_line = log_file.readlines()[-1]
            neg_logLike = float(last_line.split(":")[-1])
            MP_bf['-logLike'] = neg_logLike
    except FileNotFoundError:
        pass

    return MP_bf

def rm_files_wildcared(path):
    files = glob(path)

    for file in files:
        try:
            remove(file)
        except OSError:
            print(f"Error while deleting {file}")

    return True

def load_mp_info_files(path):
    return np.loadtxt(path)