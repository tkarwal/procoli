import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Functions to plot profile likelihoods and a quadratic cur, 
#   plus the difference of the profile from the curve 
# Doing this only makes sense for Gaussian parameters
# More generally, profiles need not be Gaussian 

def parabola(x, a, b, c):
    """
    Computes the value of a parabolic function at a given point.
    The parabolic function is defined as f(x) = ax^2 + bx + c.

    Parameters:
    - x (float or array-like): The input value(s) at which to evaluate the parabola.
    - a (float): Coefficient of the quadratic term.
    - b (float): Coefficient of the linear term.
    - c (float): Constant term.

    Returns:
    float or array-like: The value(s) of the parabolic function at the given input(s).

    Example:
    result = parabola(3, 1, -2, 1)
    # Output: 10 (since f(x) = x^2 - 2x + 1 evaluates to 10 when x = 3)
    """
    return a*x**2 + b*x + c

def plot_profile_and_parabola_diff_list( list_of_lkl_prof_dict, x_param, 
                                        legend_list=None, y_chi2='-logLike', 
                                        x_label=None, y_label=r'$\chi^2$',colours=None, 
                                        norm_to_min=True):
    """
    Plots a list of likelihood profiles along with the corresponding 
      best-fit parabolas and their differences.

    Parameters:
    - list_of_lkl_prof_dict (list): A list of dictionaries, where each dictionary 
      represents a set of likelihood profiles for a particular parameter. 
      Each dictionary should contain keys 'x_param' and 'y_chi2'
      corresponding to the x-axis parameter and the y-axis values, respectively.
    - x_param (str): The key in each dictionary representing the parameter 
      to be plotted on the x-axis.
    - legend_list (list, optional): A list of legend labels for each set of profiles. 
      Default is None.
    - y_chi2 (str, optional): The key in each dictionary representing the y-axis 
      values to be plotted. Default is '-logLike'.
    - x_label (str, optional): The label for the x-axis. Default is None, and the 
      x-axis label is set to 'x_param'.
    - y_label (str, optional): The label for the y-axis. Default is r'$\chi^2$'.
    - colours (list, optional): A list of colors for each set of profiles. 
      Default is None, and matplotlib's default color cycle is used.
    - norm_to_min (bool, optional): If True, normalizes the y-axis values by 
      subtracting the minimum y-axis value for each set of profiles. Default is True.

    Returns:
    tuple: A tuple containing the matplotlib Figure and Axes objects.

    Example:
    profiles = [{'x_param': x1, 'y_chi2': chi2_1}, {'x_param': x2, 'y_chi2': chi2_2}]
    fig, ax = plot_profile_and_parabola_diff_list(profiles, 'temperature', 
                        legend_list=['Profile 1', 'Profile 2'], norm_to_min=True)
    plt.show()
    """
    
    fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    max_y = -1e5
    min_x = 1e5
    max_x = -1e5
    
    if colours is None:
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col_num = 0
    
    if legend_list is None:
        legend_list = [str(i) for i in range( len(list_of_lkl_prof_dict) )]
    
    for full_prof_dict in list_of_lkl_prof_dict:
        xparam = full_prof_dict[x_param]
        y_passed = full_prof_dict[y_chi2]
        if norm_to_min:
            ychi2 = 2*(y_passed - min(y_passed) )
        else: 
            ychi2 = 2*(y_passed )
        
        fit_params, pcov = curve_fit(parabola, xparam, ychi2)
        
        ax[0].scatter(xparam, ychi2, c=colours[col_num], alpha=0.6, 
                      label=legend_list[col_num], s=5.)
        ax[0].plot(xparam, parabola(xparam, *fit_params), c=colours[col_num], alpha=0.6)
    
        ax[1].scatter(xparam, ychi2 - parabola(xparam, *fit_params), 
                      c=colours[col_num], alpha=0.6, s=5.)
        
        max_y = max(max_y, max(ychi2))
        min_x = min(min_x, min(xparam))
        max_x = max(max_x, max(xparam))
        
        col_num += 1
        
    if norm_to_min:
        ax[0].set_ylim((-0.01,max_y))
        ax[0].set_xlim((min_x,max_x))
    ax[0].set_ylabel(y_label, fontsize=14)
    ax[0].legend(frameon=False, fontsize=14)

    if x_label is None: 
        x_label = x_param
    ax[1].set_xlabel(x_label, fontsize=12)
    if norm_to_min:
        ax[1].set_ylabel(r'$\Delta$'+y_label, fontsize=14)
    else:
        ax[1].set_ylabel(y_label, fontsize=14)
    

    plt.subplots_adjust(hspace=0.05)
    
    return fig, ax


def plot_profile_list( list_of_lkl_prof_dict, x_param, y_chi2='-logLike', x_label=None, 
                      y_label=r'$\chi^2$', legend_list=None, colours=None, 
                      norm_to_min=True):
    """
    Plots a list of likelihood profiles as a function of a given parameter.

    Parameters:
    - list_of_lkl_prof_dict (list): A list of dictionaries, where each dictionary 
      represents a set of likelihood profiles for a particular parameter. 
      Each dictionary should contain keys 'x_param' and '-logLike' corresponding to 
      the x-axis or the profile parameter, and the negative logLikelihood, respectively.
      Other parameters can also be plot in place of the profile parameter and/or 
      the negative logLikelihood. 
    - x_param (str): The key in each dictionary representing the parameter to be 
      plotted on the x-axis.
    - y_chi2 (str, optional): The key in each dictionary representing the chi-squared 
      values to be plotted on the y-axis.
      Default is '-logLike', which is half the chi-squared.
    - x_label (str, optional): The label for the x-axis. Default is None, 
      and the x-axis label is set to the string 'x_param'.
    - y_label (str, optional): The label for the y-axis. Default is None, and is set 
      based on the value of 'y_chi2'.
    - legend_list (list, optional): A list of legend labels for each set of profiles. 
      Default is None, and profiles are labelled numerically. 
    - colours (list, optional): A list of colors for each set of profiles. 
      Default is None, and matplotlib's default color cycle is used.
    - norm_to_min (bool, optional): If True, and if y_chi2= -logLike , normalizes 
      the chi-squared values by subtracting the minimum individual chi-squared value 
      for each profile. Default is True. 

    Returns:
    tuple: A tuple containing the matplotlib Figure and Axes objects.

    Example:
    profiles = [{'x_param': x1, 'y_chi2': chi2_1, ...}, {'x_param': x2, 'y_chi2': chi2_2, ...}]
    fig, ax = plot_profile_list(profiles, 'H0', legend_list=['Profile 1', 'Profile 2'], norm_to_min=True)
    plt.show()
    """
    fig, ax = plt.subplots()

    max_y = -1e5
    min_x = 1e5
    max_x = -1e5
    
    if colours is None:
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col_num = 0
    
    if legend_list is None:
        legend_list = [str(i) for i in range( len(list_of_lkl_prof_dict) )]
    
    for full_prof_dict in list_of_lkl_prof_dict:
        xparam = full_prof_dict[x_param]
        y_passed = full_prof_dict[y_chi2]
        if y_chi2=='-logLike':
            if norm_to_min:
                ychi2 = 2*(y_passed - min(y_passed) )
                y_passed = ychi2
            else: 
                ychi2 = 2*(y_passed )
                y_passed = ychi2
        
        ax.plot(xparam, y_passed, c=colours[col_num], alpha=0.6, 
                label=legend_list[col_num], )
        
        max_y = max(max_y, max(y_passed))
        min_x = min(min_x, min(xparam))
        max_x = max(max_x, max(xparam))
        
        col_num += 1
        
    if norm_to_min:
        ax.set_ylim((-0.01,max_y))
        y_label = r'$\Delta$ '+y_label
    ax.set_ylabel(y_label, fontsize=14)

    ax.set_xlim((min_x,max_x))
    if x_label is None: 
        x_label = x_param
    ax.set_xlabel(x_label, fontsize=12)
    
    ax.legend(frameon=False, fontsize=14)

    plt.subplots_adjust(hspace=0.05)
    
    return fig, ax


# Functions to plot the progression of a simulated-annealing optimizer 
# This gets the files, puts them in chronoligical order which takes 
#   into account which SA step was first 
# Splits them into batches based on how many processes or chains were run per step
# Then plots 

def segement_list(list_of_files, size_each_list): 
    """
    Segments a list of files into sublists of a specified size.
    Usually, the list of files are chains from a simulated-annealing 
    ladder optimiser and the size of segments is the number of 
    processes or chains run per rung of the ladder. 

    Parameters:
    - list_of_files (list): The list of files to be segmented, or chain files.
    - size_each_list (int): The size of each sublist, or processes 

    Returns:
    list: A list of sublists, where each sublist contains 'size_each_list' elements
          from the original 'list_of_files'. The last sublist may have fewer elements
          if the length of 'list_of_files' is not evenly divisible by 'size_each_list'.
    """
    seg_list = []
    # looping till length l 
    for i in range(0, len(list_of_files), size_each_list):  
        seg_list.append(list_of_files[i:i + size_each_list])
    return seg_list

def get_SA_step_chains_chi2(dir_chains,chains_per_step=5, chain_len=3000):
    """
    Reads and organizes Markov chain data from simulated-annealing steps for 
    chi-squared values.

    Parameters:
    - dir_chains (str): The directory path where the chain files are located.
    - chains_per_step (int, optional): The number of chains per simulated 
      annealing step. Default is 5.
    - chain_len (int, optional): The length of each chain. Default is 3000.

    Returns:
    dict: A dictionary where keys represent simulated annealing steps, and values are 
          lists of numpy arrays containing the chi-squared values for each chain 
          in that step.

    Example:
    chains_data = get_SA_step_chains_chi2('/path/to/chains/', chains_per_step=4, chain_len=5000)
    # Output: {0: [array([chi2_0_chain_0, chi2_1_chain_0, ...]), array([chi2_0_chain_1, chi2_1_chain_1, ...]), ...],
    #          1: [array([chi2_0_chain_0, chi2_1_chain_0, ...]), array([chi2_0_chain_1, chi2_1_chain_1, ...]), ...],
    #          ...}
    """
    all_chain_files = glob(dir_chains+'*'+str(chain_len)+'*.txt')
    all_chain_files.sort(key=os.path.getmtime)
    files_SA_step = segement_list(all_chain_files, chains_per_step)
    
    chains_per_SA_step = {}
    for SA_step in range(len(files_SA_step)):
        chains_per_SA_step[SA_step] = []
        for chain in files_SA_step[SA_step]:
            chains_per_SA_step[SA_step].append(2*np.loadtxt(chain, usecols=(1,)))
    return chains_per_SA_step


def get_SA_step_chains(dir_chains,chains_per_step=5, chain_len=3000):
    """
    Extracts and organizes chains from simulated annealing steps.

    Parameters:
    - dir_chains (str): Directory path where the chain files are located.
    - chains_per_step (int, optional): Number of chains per simulated annealing step. 
      Defaults to 5.
    - chain_len (int, optional): Length of the chains. Defaults to 3000.

    Returns:
    Tuple: A tuple containing parameter names and chains organized by simulated 
      annealing steps.
    
    - param_names (list): List of parameter names, assuming 'weight', '-logLike', are 
      the first two paraemters, 
      followed by cosmological and nauisance parameters.
    - chains_per_SA_step (dict): Dictionary where keys are simulated annealing steps, 
      and values are lists of chains. Each chain is represented as a NumPy array.

    Example:
    ```
    param_names, chains_per_SA_step = get_SA_step_chains('/path/to/chain/files/', 
                                                    chains_per_step=5, chain_len=3000)
    ```
    """
    param_file = f'{dir_chains}*{str(chain_len)}*.paramnames'
    base_param_names = list(np.loadtxt(glob(param_file)[0], usecols=(0,),dtype=str))
    param_names = ['weight', '-logLike'] + base_param_names

    all_chain_files = glob(f'{dir_chains}*{str(chain_len)}*.txt')
    all_chain_files.sort(key=os.path.getmtime)
    files_SA_step = segement_list(all_chain_files, chains_per_step)

    chains_per_SA_step = {}
    for SA_step in range(len(files_SA_step)):
        chains_per_SA_step[SA_step] = []
        for chain in files_SA_step[SA_step]:
            chains_per_SA_step[SA_step].append(np.loadtxt(chain, ) )
            
    return param_names, chains_per_SA_step


def plot_SA_min_chains(SA_chains_per_step, alpha=0.2,colour='cornflowerblue', 
                       readj_chi2=0.0001):
    """
    Plots Markov chain data for simulated annealing steps, 
    focusing on the minimum chi-squared values.

    Parameters:
    - SA_chains_per_step (dict): A dictionary where keys represent simulated annealing 
      steps, and values are lists of numpy arrays containing chi-squared values for 
      each chain in that step.
    - alpha (float, optional): The transparency of the plotted lines. Default is 0.2.
    - colour (str, optional): The color of the plotted lines. 
      Default is 'cornflowerblue'.
    - readj_chi2 (float, optional): A value to adjust the minimum chi-squared for 
      better visualization. Default is 0.001.

    Returns:
    None

    Example:
    plot_SA_min_chains(chains_data, alpha=0.3, colour='darkorange', readj_chi2=0.002)
    # Displays a plot of Markov chains with minimum chi-squared values adjusted for 
      better visibility.
    """
    min_chi2 = min( 
        [min( 
            [ min(chain) for chain in SA_chains_per_step[step] ] 
        ) for step in SA_chains_per_step] 
    )

    min_chi2 -= readj_chi2
    
    for step in SA_chains_per_step:
        x_start = sum( [max([len(i) for i in SA_chains_per_step[step]]) 
                        for step in range(0,step) ] )
        for chain in SA_chains_per_step[step]:
            plt.plot( np.linspace(0, len(chain), len(chain)+1)[:-1]+x_start, 
                     chain-min_chi2, alpha=alpha,c=colour)
    plt.ylabel(r'$\Delta \chi^2$')
    plt.xlabel('MCMC step')        
    
def plot_SA_chains(SA_chains_per_step, param_list, param_to_plot, alpha=0.2, 
                   colour='cornflowerblue', ):
    """
    Plots the evolution of a specific parameter over MCMC steps for 
    simulated annealing (SA) chains.

    Parameters:
    - SA_chains_per_step (dict): Dictionary where keys are simulated annealing steps, 
      and values are lists of SA chains. 
      Each chain is represented as a NumPy array.
    - params_list (list): list of parameter names in the same order as in the rows 
      of chains. Should be  produced using a .paramnames file, 
      or from the `get_SA_step_chains' function.
    - param_to_plot (str): The name of the parameter to plot, matching names in 
      the .paramnames file
    - alpha (float, optional): The transparency of the plotted lines. Defaults to 0.2.
    - colour (str, optional): Color of the plotted lines. Defaults to 'cornflowerblue'.

    Returns:
    None

    Example:
    ```
    plot_SA_min_chains(SA_chains_per_step, 'some_parameter', alpha=0.5, colour='red')
    ```
    The plot shows the evolution of the specified parameter over MCMC steps for each 
    SA chain, with transparency and color options. The x-axis represents the 
    cumulative MCMC steps, and the y-axis represents the change in param_to_plot values.

    """
    param_index = param_list.index(param_to_plot)

    for step in SA_chains_per_step:
        x_start = sum( [max([len(i) for i in SA_chains_per_step[step]]) 
                        for step in range(0,step) ] )
        for chain in SA_chains_per_step[step]:
            plt.plot( np.linspace(0, len(chain), len(chain)+1)[:-1]+x_start, 
                     chain[:,param_index], alpha=alpha,c=colour)
    plt.ylabel(param_to_plot)
    plt.xlabel('MCMC step') 