class ParamDifferenceError(Exception):
    def __init__(self, path, error_code=1):
        super().__init__(self.message(path))
        self.error_code = error_code

    def message(self, path):
        return ('\nmatch_param_names: Error: existing file found at ' 
                    f'\n{path} ' 
                    '\nbut parameters do not match expected.')
    
class GlobalMLDifferenceError(Exception):
    def __init__(self, path, error_code=1):
        super().__init__(self.message(path))
        self.error_code = error_code

    def message(self, path):
        return ('global_min: Something went wrong. The first line of the '\
                'lkl_profile.txt file which should be global ML does not match '\
                'the global ML in file \n' \
                f'{path}.bestfit')
    
class LogParamUpdateError(Exception):
    def __init__(self, prof_param, lkl_lp, error_code=1):
        super().__init__(self.message(prof_param, lkl_lp))
        self.error_code = error_code

    def message(self, prof_param, lkl_lp):
        return ('Error: increment_update_logparam: could not find line with '\
                f'profile lkl parameter {prof_param} in log.param at {lkl_lp}')