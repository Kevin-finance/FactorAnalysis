from itertools import product

def get_fda_parameter_grid():
    grid = {
        "starting_day": [2],
        "stop_loss_pct": [0.05, 0.1],
        "holding_days" : [10,15,20],
        "max_positions" : [2,3,4],
    }
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in product(*values)]



