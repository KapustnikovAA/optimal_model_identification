import numpy as np 
from typing import Any, List, Generator, Dict, Tuple

from utils.data_saver import HDF5
from utils.yaml_loader import load_yaml

from scripts.complexity_calculation import main as complex_calculation

def model_identification (config_path: str,
                          threshold: float | None = None,
                          exp_range: Any | None = None) -> List[Tuple[str, int]]:
    
    config = load_yaml(config_path = config_path)

    if threshold is None:

        try: 
            threshold = config["global_settings"]["threshold"]
        except:
            threshold = False

    result_list = []
    for data in get_relative_error_to_min_ratio(exp_range = exp_range):
        
        if threshold:

            index_min_ratio = int((np.where(data["min_ratio"] < threshold))[0][0])
            
            complexity = data["complexity"][index_min_ratio]
            polynome = data["arg_P_min"][index_min_ratio] + 1

            result_list.append((complexity, polynome))
            print(f"Experiment_{data['exp_number']}:\nSelected model -> {complexity}\nPolynome degree is {polynome}")
        
        else:
            
            print("models | minimal errors")
            for model, error in zip(data["complexity"], data["all_min_errors"]):
                print(f"{model}\t{error}") 

            print(f"Experiment_{data['exp_number']}: Threshold is not set")
    
    return result_list

def get_relative_error_to_min_ratio (exp_range: Any | None = None) -> Generator[Dict[int, Any], None, None]:
    """
    min_ratio = Θ_{mod_i} / Θ_min
    """

    if isinstance(exp_range, int):
        exp_range = range(exp_range, exp_range + 1)
    elif isinstance(exp_range, list):
        exp_range = range(exp_range[0], exp_range[-1] + 1)
    else:
        hdf5_data = HDF5(path = "none", exp_num = -1)
        exp_range = hdf5_data.get_num_experiments(path = "data")

    complexity_dict = complex_calculation()
    complexity = sorted(complexity_dict.keys(), key = lambda item: complexity_dict[item])

    for exp_number in exp_range:
        try:

            hdf5_data = HDF5(path = "data",
                                exp_num = exp_number)
            errors_data = hdf5_data.load_group(group_name = "errors")

            model_pool = hdf5_data.load_group(group_name = "phi_functions")["model_pool"]
            model_pool = dict((int(item[0]), item[-1]) for item in model_pool.items())
            reversed_model_pool_dict = {value: key for key, value in model_pool.items()}

            complexity_index = [reversed_model_pool_dict[item] for item in complexity]
            
            absolute_min_error = errors_data["min_error"]
            all_P_errors = errors_data["all_P_errors"]
            arg_P_min = errors_data["arg_P_min"]

            all_min_errors = all_P_errors[tuple(range(all_P_errors.shape[0])), arg_P_min][complexity_index]
            arg_P_min =  arg_P_min[complexity_index]
            min_ratio = all_min_errors / absolute_min_error

            yield {"exp_number": exp_number,
                   "min_ratio": min_ratio,
                   "complexity": complexity,
                   "arg_P_min": arg_P_min,
                   "all_min_errors": all_min_errors}
        
        except Exception as e:
            
           print(f"Error with experiment file number {exp_number}:\n{e}") 