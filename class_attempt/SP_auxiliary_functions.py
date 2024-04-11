import numpy as np
from tqdm import tqdm


def initialize_survey(method='frozen_random'):
    """
    Initialize a survey message.
    warning_keys = ['----', 
                    '---+', 
                    '--+-', 
                    '--++', 
                    '-+--', 
                    '-+-+', 
                    '-++-', 
                    '-+++', 
                    '+---', 
                    '+--+', 
                    '+-+-', 
                    '+-++', 
                    '++--', 
                    '++-+', 
                    '+++-', 
                    '++++'] # 16 possible warnings corresponding to the survey_values array indices

    methods accepted:
    - "random": all random 
    - "random_smart": all random, except first warning which is set to 0
    - "random_smarter": all random, except first and last warnings which are set to 0
    - "frozen_random": only warnings with one + are nonzero - random values 
    - "frozen_equal": only warnings with one + are nonzero - same values 
    - "frozen_random_dontcare": same as frozen_random, but with a bit of noise on dont care warning
    - "frozen_equal_dontcare": same as frozen_equal, but with a bit of noise on dont care warning
    """
    survey_values = np.zeros((16,1))
    if method == 'random':
        for i in range(16):
            survey_values[i] = np.random.uniform(0,1)
    elif method == 'random_smart':
        for i in range(1, 16):
            survey_values[i] = np.random.uniform(0,1)
    elif method == 'random_smarter':
        for i in range(1, 15):
            survey_values[i] = np.random.uniform(0,1)
    elif method == 'frozen_random':
        survey_values[1] = np.random.uniform(0,1)
        survey_values[2] = np.random.uniform(0,1)
        survey_values[4] = np.random.uniform(0,1)
        survey_values[8] = np.random.uniform(0,1)
    elif method == 'frozen_equal':
        survey_values[1] = 0.25
        survey_values[2] = 0.25
        survey_values[4] = 0.25
        survey_values[8] = 0.25
    elif method == 'frozen_random_dontcare':
        survey_values[1] = np.random.uniform(0,1)
        survey_values[2] = np.random.uniform(0,1)
        survey_values[4] = np.random.uniform(0,1)
        survey_values[8] = np.random.uniform(0,1)
        survey_values[0] = 0.05
    elif method == 'frozen_equal_dontcare':
        survey_values[1] = 0.25
        survey_values[2] = 0.25
        survey_values[4] = 0.25
        survey_values[8] = 0.25
        survey_values[0] = 0.05
    else:
        raise ValueError('Invalid method')
    
    survey_values = survey_values/np.sum(survey_values)
    return survey_values

def generate_all_possible_warnings():
    """
    Generate all possible warnings.
    """
    warnings = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    warnings.append(np.array([[i, j], [k, l]]))
    return np.array(warnings)

def get_warning_index(warning):
    """
    Get the index of a warning in the list of all possible warnings.
    """
    warnings = generate_all_possible_warnings()
    for i in range(len(warnings)):
        if np.array_equal(warning, warnings[i]):
            return i
    return -1

def gen_configurations(d):
    """
    returns all possible configurations of d-1 neighbours
    """
    return np.array(np.meshgrid(*[[0, 1]] * (d-1))).T.reshape(-1, d-1)

def respect_rule(rule, i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if rule[outer_density]=='0':
            return True if i==0 else False
        elif rule[outer_density]=='1':
            return True if i==1 else False
        elif rule[outer_density]=='+':
            return True
        elif rule[outer_density]=='-':
            return False
        
def neighbouring_warnings_allow(sigma_i, configuration, neighbouring_warnings):
    """
    returns True if the neighbouring warnings allow the configuration, False otherwise
    """
    
    for k, warning in enumerate(neighbouring_warnings):
        sigma_k = configuration[k]
        if warning[sigma_k, sigma_i] == 0:
            return False

    return True


def warning_config_is_fixed_point(warning_index, config, rule):
    """
    Check if a warning configuration is a fixed point of a rule.
    """
    warning_list = generate_all_possible_warnings()
    warning = warning_list[warning_index]
    new_warning = np.zeros((2, 2))
    for sigma_i in range(2):
        for sigma_j in range(2):
            configurations = gen_configurations(len(config)+1)
            for configuration in configurations:
                if respect_rule(rule, sigma_i, sigma_j, configuration) and neighbouring_warnings_allow(sigma_i, configuration, config):
                    new_warning[sigma_i, sigma_j] = 1
                    break
    
    if np.array_equal(new_warning, warning):
        return True
    else:
        return False
    
def fixed_point_update(config, rule):
    """
    Check if a warning configuration is a fixed point of a rule.
    """
    new_warning = np.zeros((2, 2))
    for sigma_i in range(2):
        for sigma_j in range(2):
            configurations = gen_configurations(len(config)+1)
            for configuration in configurations:
                if respect_rule(rule, sigma_i, sigma_j, configuration) and neighbouring_warnings_allow(sigma_i, configuration, config):
                    new_warning[sigma_i, sigma_j] = 1
                    break
    
    return new_warning

def gen_configurations(d):
    """
    returns all possible configurations of d-1 neighbours
    """
    return np.array(np.meshgrid(*[[0, 1]] * (d-1))).T.reshape(-1, d-1)

def respect_rule(rule, i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if rule[outer_density]=='0':
            return True if i==0 else False
        elif rule[outer_density]=='1':
            return True if i==1 else False
        elif rule[outer_density]=='+':
            return True
        elif rule[outer_density]=='-':
            return False
        
def neighbouring_warnings_allow(sigma_i, configuration, neighbouring_warnings):
    """
    returns True if the neighbouring warnings allow the configuration, False otherwise
    """
    
    for k, warning in enumerate(neighbouring_warnings):
        sigma_k = configuration[k]
        if warning[sigma_k, sigma_i] == 0:
            return False

    return True


def warning_config_is_fixed_point(warning_index, config, rule):
    """
    Check if a warning configuration is a fixed point of a rule.
    """
    warning_list = generate_all_possible_warnings()
    warning = warning_list[warning_index]
    new_warning = np.zeros((2, 2))
    for sigma_i in range(2):
        for sigma_j in range(2):
            configurations = gen_configurations(len(config)+1)
            for configuration in configurations:
                if respect_rule(rule, sigma_i, sigma_j, configuration) and neighbouring_warnings_allow(sigma_i, configuration, config):
                    new_warning[sigma_i, sigma_j] = 1
                    break
    
    if np.array_equal(new_warning, warning):
        return True
    else:
        return False
    
def fixed_point_update(config, rule):
    """
    Check if a warning configuration is a fixed point of a rule.
    """
    new_warning = np.zeros((2, 2))
    for sigma_i in range(2):
        for sigma_j in range(2):
            configurations = gen_configurations(len(config)+1)
            for configuration in configurations:
                if respect_rule(rule, sigma_i, sigma_j, configuration) and neighbouring_warnings_allow(sigma_i, configuration, config):
                    new_warning[sigma_i, sigma_j] = 1
                    break
    
    return new_warning

def generate_combinations(matrix_set, d):
    def generate_combinations_recursive(matrix_set, d, current_combination):
        if d == 0:
            print("ola!")
            return [current_combination.copy()]
        
        combinations = []
        for i in range(len(matrix_set)):
            current_combination.append(matrix_set[i])
            combinations += generate_combinations_recursive(matrix_set, d - 1, current_combination)
            current_combination.pop()

        return combinations

    return generate_combinations_recursive(matrix_set, d, [])

def survey_propagation_eff(rule, initial_surveys, num_iters=500, dampening=0.8, tol=1e-7):
    survey_values = initial_surveys.copy()
    d = len(rule)-1
    warning_list = generate_all_possible_warnings()
    warning_configs = generate_combinations(warning_list, d-1)
    for _ in tqdm(range(num_iters)):
        survey_values_old = survey_values.copy()
        update_sum = np.zeros((16,1))
        for config in warning_configs:
            new_warning = fixed_point_update(config, rule)
            if np.array_equal(new_warning, np.zeros((2,2))):
                continue
            for warning_idx in range(16):
                if np.array_equal(new_warning, warning_list[warning_idx]):
                    update_prod = 1
                    for k in range(d-1):
                        update_prod *= survey_values[get_warning_index(config[k])]
                    update_sum[warning_idx] += update_prod
        survey_values = update_sum/np.sum(update_sum)
        survey_values = (1-dampening)*survey_values + dampening*survey_values_old

        if np.linalg.norm(survey_values - survey_values_old) < tol:
            break
    return survey_values

def warning_config_is_fixed_point_complexity(config, rule):
    d = len(config)
    configurations = gen_configurations(d+1)
    for sigma_i in range(2):
        for configuration in configurations:
            if respect_rule(rule, sigma_i, configuration[0], configuration[1:]) and neighbouring_warnings_allow(sigma_i, configuration, config):
                return True
    return False


def edge_complexity_condition(warning1, warning2):
    configurations = gen_configurations(3)
    for configuration in configurations:
        if warning1[configuration[0], configuration[1]] == 1 and warning2[configuration[1], configuration[0]] == 1:
            return True
    return False
    

def complexity(survey_values, rule):
    """
    Calculate the complexity of a survey fixed point.
    """
    d = len(rule)-1
    warning_list = generate_all_possible_warnings()
    warning_configs_node_complexity = generate_combinations(warning_list, d)
    zi = 0
    for config in warning_configs_node_complexity:
        if warning_config_is_fixed_point_complexity(config, rule):
            update_prod = 1
            for k in range(d):
                update_prod *= survey_values[get_warning_index(config[k])]
            zi += update_prod
    
    warning_configs_edge_complexity = generate_combinations(warning_list, 2) 
    zia = 0
    for config in warning_configs_edge_complexity:
        if edge_complexity_condition(config[0], config[1]):
            zia += survey_values[get_warning_index(config[0])] * survey_values[get_warning_index(config[1])]
    
    complexity = np.log(zi) - d/2 * np.log(zia)
    return complexity