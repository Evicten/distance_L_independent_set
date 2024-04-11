import SP_auxiliary_functions as aux

class my_SP:
    def __init__(self, rule, survey_init=None, tol=1e-7, max_iter=500, damping_parameter=0.8):
        self.rule=rule
        self.tol=tol
        self.max_iter=max_iter
        self.damping_parameter=damping_parameter

        if survey_init is not None:    
            self.survey_values=survey_init.copy()
        else:
            self.survey_values=None
        
        self.complexity=None
        
    def __repr__(self):
        description="Instance of class \'SP\'\nRule : "+str(self.rule)

        return description
    
    def run(self, initialization_method="frozen_random_dontcare", verbose=False):
        """
        methods accepted:
            - "random": all random 
            - "random_smart": all random, except first warning which is set to 0
            - "random_smarter": all random, except first and last warnings which are set to 0
            - "frozen_random": only warnings with one + are nonzero - random values 
            - "frozen_equal": only warnings with one + are nonzero - same values 
            - "frozen_random_dontcare": same as frozen_random, but with a bit of noise on dont care warning
            - "frozen_equal_dontcare": same as frozen_equal, but with a bit of noise on dont care warning
        """
        if self.survey_values is None:
            self.survey_values = aux.initialize_survey(initialization_method)
        self.survey_values = aux.survey_propagation_eff(self.rule, self.survey_values, self.max_iter, self.damping_parameter, self.tol)
        self.complexity = aux.complexity(self.rule, self.survey_values)
        if verbose:
            print("Complexity : "+str(self.complexity))
            print("Survey values : "+str(self.survey_values))

    
    