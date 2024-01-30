from os import remove, makedirs
from glob import glob
from shutil import copy
from importlib_resources import files

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
    """
    Wrapper to remove files
    
    :path: Path of the directory to remove files from

    :return: The True
    """

    files = glob(path)

    for file in files:
        try:
            remove(file)
        except OSError:
            print(f'Error while deleting {file}')

    return True

def read_file(path):
    """
    Wrapper to load files as a single string
    
    :path: Path of the file to load

    :return: The files text
    """

    with open(path, 'r') as f:
            file_txt = f.read()

    return file_txt

def readlines_file(path):
    """
    Wrapper to load files as a list by line
    
    :path: Path of the file to load

    :return: The files text
    """

    with open(path, 'r') as f:
            file_txt = f.readlines()

    return file_txt

def save_file(path, lines):
    """
    Wrapper to write files as a list by line
    
    :path: Path of the file to save to
    :lines: List of strings to save to the file

    :return: Nothing
    """

    with open(path, 'w') as f:
            for line in lines:
                f.write(line)

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

def write_bf_dict_to_file(bf_dict, bf_file):
    """
    Write a best-fit dictionary to a file in the style of MontePython .bestfit files.

    Args:
    best_fit_dict (dict): A dictionary containing parameter names as keys 
      and their corresponding best-fit values.
    bf_file (str): The path to the file where the best-fit dictionary 
      will be written.

    Returns:
    str: The path to the file where the best-fit dictionary has been written.

    Example:
    >>> best_fit_params = {'param1': 0.123, 'param2': 1.456, 'param3': 2.789}
    >>> bf_file_path = '/path/to/best_fit_results.bestfit'
    >>> write_bf_dict_to_file(best_fit_params, bf_file_path)
    '/path/to/best_fit_results.bestfit'
    """
    params_line = '#      '+',      '.join([key for key in bf_dict])+'\n'
    values_line = '      '.join([str(bf_dict[key]) for key in bf_dict])

    with open(bf_file, 'w') as f:
        f.write(params_line)
        f.write(values_line)
        
    return bf_file

def get_experiment_crosslist():
    """
    Load and return a dictionary mapping log.param names to corresponding output names.
    Some MontePython likelihoods are confusing to Procoli, because they are called by certain 
    names in the log.param, but they are output by different names. This function is required 
    to relate the two, particularly for getting the chi2 per experiment output. 

    This function reads the MP_experiment_crosslist.csv file, which contains pairs of log.param
    names and their corresponding output names, for likelihoods that have this confusing 
    behaviour. It then creates a dictionary with log.param names as keys and output names 
    as values.

    Returns:
    dict: A dictionary mapping log.param names to corresponding output names.

    Example:
    >>> crosslist_dict = get_experiment_crosslist()
    >>> print(crosslist_dict)
    {'Pantheon_Plus': 'Pantheon_Plus_test', 'Pantheon_Plus_SH0ES': 'Pantheon_Plus_test', ...}
    """
    exp_crosslist_file = files('procoli.data').joinpath('MP_experiment_crosslist.csv')
    log_exp, output_exp = np.loadtxt(exp_crosslist_file, dtype=str, delimiter=',', unpack=True)
    experiment_crosslist = dict(zip(log_exp, output_exp))
    
    return experiment_crosslist