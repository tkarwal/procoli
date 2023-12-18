from os import remove, makedirs
from glob import glob
from shutil import copy

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
        with open(f'{MP_bf_file[:-8]}.log') as log_file:
            last_line = log_file.readlines()[-1]
            neg_logLike = float(last_line.split(':')[-1])
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
            print(f'Error while deleting {file}')

    return True

def load_mp_info_files(path):
    """
    Wrapper to load files with np.loadtxt
    
    :path: Path of the file to load

    :return: The loaded numpy array
    """
     
    return np.loadtxt(path)

def save_mp_info_files(path, array, fmt='%.18e', delimiter=' '):
    """
    Wrapper to save numpy arrays
    
    :path: Path to save the file to
    :array: Numpy array to save
    :fmt: Formatting as a string or list of strings to save the data
    :delimiter: Delimiter to use between data in the file

    :return: The result of 
    """

    return np.savetxt(path, array, fmt=fmt, delimiter=delimiter)

def make_path(path, exist_ok=True):
    """
    Wrapper around the os.makedirs function to make a directory
    and the path to it
    
    :path: Path for the directory to create
    :exist_ok: If true then don't throw an error if the directory exists

    :return: Nothing
    """
        
    return makedirs(path, exist_ok=exist_ok)

def file_copy(target, destination):
    """
    Wrapper around the shutil.copy function to copy files
    
    :target: Path of the target file
    :destination: Path to where the file should be copied

    :return: Nothing
    """
        
    return copy(target, destination)