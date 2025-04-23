from itertools import product


def generate_parameter_combinations(param_grid):
    parameter_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    return [dict(zip(param_keys, combination)) for combination in parameter_combinations]
