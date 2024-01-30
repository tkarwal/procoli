# Procoli


| Procoli        | Profiles of cosmological likelihoods                 |
|----------------|------------------------------------------------------|
| Authors        | Tanvi Karwal and Daniel Pfeffer                      |
| Installation   | `pip install procoli`                                |
| Reference      | [arXiv:2401.14225](https://arxiv.org/abs/2401.14225) |


## Description 

Procoli is a python package for extracting profile likelihoods in cosmology. It wraps MontePython, a fast sampler written specifically for the CLASS Boltzmann code. All likelihoods available for use with MontePython are hence immediately available to all Procoli users. 

It is based on a simulated-annealing optimizer to find the global maximum likelihoods value as well as the maximum likelihood points along the profile of any use input parameter. 

## Installation 

Prerequisites are [MontePython](https://github.com/brinckmann/montepython_public) and [GetDist](https://getdist.readthedocs.io/en/latest/) and everything that those entail. Please refer to their individual documentations to see specific installation instructions.  
- GetDist can be pip-installed and Procoli will attempt to install it by itself. 
- MontePython must be installed by the user and must be on your PATH, such that it is callable from any directory, as described [here](https://github.com/brinckmann/montepython_public/blob/3.6/README.rst#the-montepython-part). 
- Install [CLASS](https://github.com/lesgourg/class_public) and any likelihoods that you may want to use, eg. the [Planck Likelihood Code](https://pla.esac.esa.int/pla/#home).

To install Procoli, simply run  
`pip install procoli --upgrade`

Alternatively you can clone the GitHub and install by doing:
```
git clone git@github.com:tkarwal/procoli.git
cd procoli
pip install -e .
```
where the `-e` can be omitted if you do not want an editable version of the code. 

## Running Procoli 

Please see the notebooks, python scripts and bash scripts for example runs. 
The recommended running strategy is to run Procoli via a python script submitted to a cluster, ideally running on several processors to speed up the minimization, i.e., `cpus-per-task` in your job script should be ~4. 
The code should also be run using several parallel chains for robustness, i.e. input `processes` >= 4, say.
For more details, see the [example_run.py](https://github.com/tkarwal/procoli/blob/main/example_run.py) and [example_bash_script.sh](https://github.com/tkarwal/procoli/blob/main/example_bash_script.sh) files. 

The code outputs the file `<name>_<+/-><prof_param>_lkl_profile.txt` that contains the values of all parameters at the minimized points for each iteration of the profile parameter, plus derived params, the $-\log \mathcal{L}$ as well as the individual $\chi^2$ per experiment. 
The `<+/->` in the filename indicates whether the positive or negative tail of the profile was explored, starting from the best fit. 
As shown in the example files, both directions can be explored and these files can then be plotted, with some quick basic built-in functions demonstrated in [lcdm_example.ipynb](https://github.com/tkarwal/procoli/blob/main/lcdm_example.ipynb). 

## Citing us

Please cite the release paper [arXiv:2401.14225](https://arxiv.org/abs/2401.14225), along with MontePython, GetDist and CLASS. 

```
% Procoli:
@article{karwal2024procoli,
      author = "Karwal, Tanvi and Patel, Yashvi and Bartlett, Alexa and Poulin, Vivian and Smith, Tristan L. and Pfeffer, Daniel N.",
      title = "{Procoli: Profiles of cosmological likelihoods}", 
      eprint = "2401.14225",
      archivePrefix = "arXiv",
      primaryClass = "astro-ph.CO",
      year = "2024"
}

% CLASS:
@article{Blas:2011rf,
    author = "Blas, Diego and Lesgourgues, Julien and Tram, Thomas",
    title = "{The Cosmic Linear Anisotropy Solving System (CLASS) II: Approximation schemes}",
    eprint = "1104.2933",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "CERN-PH-TH-2011-082, LAPTH-010-11",
    doi = "10.1088/1475-7516/2011/07/034",
    journal = "JCAP",
    volume = "07",
    pages = "034",
    year = "2011"
}
% MontePython:
@article{Brinckmann:2018cvx,
      author         = "Brinckmann, Thejs and Lesgourgues, Julien",
      title          = "{MontePython 3: boosted MCMC sampler and other features}",
      year           = "2018",
      eprint         = "1804.07261",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      SLACcitation   = "%%CITATION = ARXIV:1804.07261;%%"
}
@article{Audren:2012wb,
      author         = "Audren, Benjamin and Lesgourgues, Julien and Benabed,
                        Karim and Prunet, Simon",
      title          = "{Conservative Constraints on Early Cosmology: an
                        illustration of the Monte Python cosmological parameter
                        inference code}",
      journal        = "JCAP",
      volume         = "1302",
      pages          = "001",
      doi            = "10.1088/1475-7516/2013/02/001",
      year           = "2013",
      eprint         = "1210.7183",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      reportNumber   = "CERN-PH-TH-2012-290, LAPTH-048-12",
      SLACcitation   = "%%CITATION = ARXIV:1210.7183;%%",
}
% GetDist
@article{Lewis:2019xzd,
 author         = "Lewis, Antony",
 title          = "{GetDist: a Python package for analysing Monte Carlo
                   samples}",
 year           = "2019",
 eprint         = "1910.13970",
 archivePrefix  = "arXiv",
 primaryClass   = "astro-ph.IM",
 SLACcitation   = "%%CITATION = ARXIV:1910.13970;%%",
 url            = "https://getdist.readthedocs.io"
}
```

