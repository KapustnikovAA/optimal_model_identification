import numpy as np
from time import time
from multiprocessing import Pool
from tempfile import mkstemp
from os import close, getpid
from typing import Tuple, List,Any
from functools import partial
import logging

from utils.data_saver import HDF5
from utils.yaml_loader import load_yaml
from utils.temporary_cache import create_cache_folder, remove_cache_folder
from utils.auto_import_models import import_models
from utils.logger import setup_logger

def testing_all_models_PHI_function(polynome: int,
                                    cache_dir_name: str,
                                    data_args: List[np.ndarray],
                                    all_model_classes: Tuple) -> Tuple[List, str]:
    
    du_series, t_series, u_series = data_args
    model_class = all_model_classes[0]

    Phi_numerical, S_error = model_class.phi_numerical_fast(t_series, 
                                                            u_series, 
                                                            du_series, 
                                                            P = polynome)

    ERRORS = []                                                            
    Phi_ = [Phi_numerical]														
    for model in all_model_classes:
        
        calc_result = model.errors_calculation(t_series,
                                               u_series,
                                               du_series,
                                               Phi_numerical,
                                               S_error)
        
        ERRORS.append(calc_result[0])
        Phi_.append(calc_result[-1])

    #1. Creating one secure temporary file for this task
    fd, temp_path = mkstemp(suffix = ".npy", 
                            prefix = f"polynome_{polynome}_",
                            dir = cache_dir_name)
    
    # 2. Closing the system descriptor to avoid file freezes
    close(fd)

    # 3. Saving all three heavy arrays to this file
    np.save(temp_path, np.array(Phi_))

    # 4. Forcibly clearing the memory of the current process from heavy data
    del Phi_

    return ERRORS, temp_path

def main(config_path: str,
         exp_range: Any | None = None,
         cache_folder_name: str = "scripts/script_cache") -> None:

    pid = getpid()
    time_0 = time()
    
    TAYLOR_POLY_DEGREE_MAX = 9

    create_cache_folder(folder_name = cache_folder_name,
                        pid = pid)

    logger = setup_logger(__name__, 
                          log_dir = "logs",
                          log_file = f"PHI_script_{pid}.log", 
                          level = logging.INFO)
    
    try:

        logger.info(f"Code run started")

        models_dict = import_models()

        config = load_yaml(config_path = config_path)

        prll_mode = config["global_settings"]["parallel_calculations"]
        prll_processes = config["global_settings"]["parallel_processes"]
        prll = Pool(processes = prll_processes if prll_processes <= TAYLOR_POLY_DEGREE_MAX else TAYLOR_POLY_DEGREE_MAX)
        
        if isinstance(exp_range, int):
             exp_range = range(exp_range, exp_range + 1)
        elif isinstance(exp_range, list):
            exp_range = range(exp_range[0], exp_range[-1] + 1)
        else:
            hdf5_data = HDF5(path = "none", exp_num = -1)
            exp_range = hdf5_data.get_num_experiments(path = "data")
        
        for exp_number in exp_range:
            logger.info(f"experiment_nimber: {exp_number}\nconfig:{config}")
            hdf5_data = HDF5(path = "data",
                             exp_num = exp_number)
            _, *derivatives_data = hdf5_data.load_group(group_name = "derivatives").values()
            '''
            u_min = min(derivatives_data[-1])
            try:
                if -u_min < 0:
                    for model_name in models_dict:
                        models_dict[model_name].initial_guesses = -u_min - 0.1
                else:
                    for model_name in models_dict:
                        models_dict[model_name].initial_guesses = -u_min + 0.1
            except: pass
            '''
            u_min = min(derivatives_data[-1])
            u_max = max(derivatives_data[-1])
            try:
                if (-u_min < 0) and (-u_max < 0):
                    for model_name in models_dict:
                        models_dict[model_name].initial_guesses = -u_max - 0.1
                elif (-u_min > 0) and (-u_max < 0):
                    for model_name in models_dict:
                        models_dict[model_name].initial_guesses = -u_max - 0.1
                elif (-u_min > 0) and (-u_max > 0):
                    for model_name in models_dict:
                        models_dict[model_name].initial_guesses = -u_min + 0.1
                
                logger.info(f"u_max: {u_max}\nu_min:{u_min}\nig: {models_dict[model_name].initial_guesses}")

            except: pass
            
            calc_all_PHI_function = partial(testing_all_models_PHI_function,
                                            cache_dir_name = f"{cache_folder_name}_{pid}",
                                            data_args = derivatives_data,
                                            all_model_classes = tuple(models_dict.values()))
            
            if prll_mode:
                result = tuple(prll.map(calc_all_PHI_function, 
                                        range(1, TAYLOR_POLY_DEGREE_MAX + 1)))
                exp_errors = tuple(items[0] for items in result)
                temp_file_paths = tuple(items[-1] for items in result)
            else:
                result = tuple(map(calc_all_PHI_function,
                                   range(1, TAYLOR_POLY_DEGREE_MAX + 1)))
                exp_errors = tuple(items[0] for items in result)
                temp_file_paths = tuple(items[-1] for items in result)
            
            all_P_errors = np.column_stack(exp_errors)
            min_error = np.min(all_P_errors)
            arg_P_min = np.argmin(all_P_errors, axis = 1)

            polynome_phi = []

            for temp_path in temp_file_paths:

                data = np.load(temp_path)

                phi_num = data[0, :]
                phi_models = data[1:, :]
                
                polynome_phi.append(phi_models)
                
            hdf5_data.create_group(group_name = "phi_functions",
                                   data = (phi_num, *polynome_phi),
                                   attr = dict((key, item) for key, item in enumerate(models_dict.keys())))
            
            hdf5_data.create_group(group_name = "errors",
                                   data = (arg_P_min,
                                           all_P_errors),
                                   attr = min_error)
    
    except Exception:
    
        logger.exception(f"Experiment_{exp_number} could not be processed.")
    
    else:

        logger.info(f"Experiment_{exp_number} completed successfully.")

    finally: 
        
        remove_cache_folder(folder_name = cache_folder_name,
                            pid = pid)
        if prll is not None:
            prll.close()
            prll.join()

    print(f"running time: {time() - time_0} s\nphi_function_approximation.py completed successfully")