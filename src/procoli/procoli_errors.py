class ParamDifferenceError(Exception):
    """
    A custom exception for Procoli
    Is raised when there is a mismatch of parameters
    """

    def __init__(self, path, error_code=1):
        """
        Constructs the exception using the base class

        :path: Path to file with mismatched parameter names

        :return: Nothing
        """

        super().__init__(self.message(path))
        self.error_code = error_code

    def message(self, path):
        """
        Generate the message associated with the exception

        :path: Path to file with mismatched parameter names

        :return: The error message to give with the exception
        """
        return ('\nmatch_param_names: Error: existing file found at ' 
                    f'\n{path} ' 
                    '\nbut parameters do not match expected.')
    
class GlobalMLDifferenceError(Exception):
    """
    A custom exception for Procoli
    Is raised when previous minimum likelihood does not match
        what is currently being used
    """

    def __init__(self, path, error_code=1):
        """
        Constructs the exception using the base class

        :path: Path to the bestfit file that doesn't match the 
            lkl_profile.txt file

        :return: Nothing
        """
        
        super().__init__(self.message(path))
        self.error_code = error_code

    def message(self, path):
        """
        Generate the message associated with the exception

        :path: Path to the bestfit file that doesn't match the 
            lkl_profile.txt file

        :return: The error message to give with the exception
        """

        return ('global_min: Something went wrong. The first line of the '\
                'lkl_profile.txt file which should be global ML does not match '\
                'the global ML in file \n' \
                f'{path}.bestfit')
    
class LogParamUpdateError(Exception):
    """
    A custom exception for Procoli
    Is raised when the parameter the profile is being generated for
        can not be find in the parameter log
    """

    def __init__(self, prof_param, lkl_lp, error_code=1):
        """
        Constructs the exception using the base class

        :prof_param: Parameter of interest for likelihood profiling
        :lkl_lp: Path to log.param file

        :return: Nothing
        """

        super().__init__(self.message(prof_param, lkl_lp))
        self.error_code = error_code

    def message(self, prof_param, lkl_lp):
        """
        Generate the message associated with the exception

        :prof_param: Parameter of interest for likelihood profiling
        :lkl_lp: Path to log.param file

        :return: The error message to give with the exception
        """

        return ('Error: increment_update_logparam: could not find line with '\
                f'profile lkl parameter {prof_param} in log.param at {lkl_lp}')