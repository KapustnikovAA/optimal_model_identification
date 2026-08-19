import numpy as np
from time import time
from multiprocessing import Pool
from typing import Tuple, Any
from functools import partial
from tempfile import mkstemp
from os import close, getpid

from models.basic_algorithms_class import General_Solver

from utils.data_saver import HDF5
from utils.yaml_loader import load_yaml
from utils.temporary_cache import create_cache_folder, remove_cache_folder

def derivative_calc (metadata_param: Tuple[int, int, np.ndarray, np.ndarray],
                     cache_dir_name: str,
                     pid: int,
                     julia_path: str |  bool,
                     julia_mode: bool) -> Tuple[int, int, str]:

      if julia_mode and julia_path:
            abstract_model = General_Solver(julia_path = julia_path,
                                            julia_mode = julia_mode)
      else: abstract_model = General_Solver()

      exp_number, abstract_model.window, t_series, per_series = metadata_param
      abstract_model.dt = t_series[1] - t_series[0]
      abstract_model.sl =  (abstract_model.window - 1) // 2

      d_per_series = abstract_model.polynomdif(per_series,
                                  dt = abstract_model.dt,
                                  P = abstract_model.poly_degree,
                                  m = abstract_model.window)
      
      per_series = d_per_series[:, 0]
      d_per_series = d_per_series[:, 1]
      t_series = t_series[abstract_model.sl:-abstract_model.sl]

      # 1. Creating one secure temporary file for this task
      fd, temp_path = mkstemp(suffix = ".npz", 
                              prefix = f"exp_{exp_number}_",
                              dir =  f"{cache_dir_name}_{pid}")
      # 2. Closing the system descriptor to avoid file freezes
      close(fd)

      # 3. Saving all three heavy arrays to this file
      np.savez(temp_path, 
               t_series = t_series, 
               per_series = per_series, 
               d_per_series = d_per_series)

      # 4. Forcibly clearing the memory of the current process from heavy data
      del t_series, per_series, d_per_series
      # ----------------------------------------

      # We return metadata and the TEXT PATH to the cache instead of the arrays themselves.
      return exp_number, abstract_model.window, temp_path

def window_size_choose (file_class: type,
                        window_size_dict: dict) -> int:

      model_name, *_, noise_sigma = file_class.load_group(group_name = "metadata")["integation_parameters"].values()

      return window_size_dict[model_name][f"noise_sigma_{noise_sigma}"]

def main (config_path: str,
          exp_range: Any | None = None,
          cache_folder_name: str = "scripts/script_cache") -> None:
      
      pid = getpid()
      time_0 = time()

      create_cache_folder(folder_name = cache_folder_name,
                          pid = pid)

      try:

            config = load_yaml(config_path = config_path)

            prll_mode = config["global_settings"]["parallel_calculations"]
            prll_processes = config["global_settings"]["parallel_processes"]
            prll = Pool(processes = prll_processes)

            window_size_dict = config["window_size"]

            julia_path = config["global_settings"]["julia_path"]
            julia_mode = config["global_settings"]["julia_mode"]

            diff_calc = partial(derivative_calc,
                              cache_dir_name = cache_folder_name,
                              pid = pid,
                              julia_path = julia_path, 
                              julia_mode = julia_mode)

            if isinstance(exp_range, int):
                  exp_range = range(exp_range, exp_range + 1)
            elif isinstance(exp_range, list):
                  exp_range = range(exp_range[0], exp_range[-1] + 1)
            else:
                  hdf5_data = HDF5(path = "none", exp_num = -1)
                  exp_range = hdf5_data.get_num_experiments(path = "data")
            
            t_series = []
            u_series = []
            window_size = []

            for exp_numb in exp_range:
                  hdf5_data = HDF5(path = "data", 
                                   exp_num = exp_numb)
                  raw_data = hdf5_data.load_group(group_name = "raw")

                  t_series.append(raw_data["t_series"])
                  u_series.append(raw_data["u_series"])
                  window_size.append(window_size_choose(file_class = hdf5_data,
                                                        window_size_dict = window_size_dict))

            arguments = zip(exp_range,
                            window_size,
                            t_series,
                            u_series)

            if prll_mode:
                  result = tuple(prll.map(diff_calc, arguments))
            else:
                  result = tuple(map(diff_calc, arguments))
                 
            for exp_number, window_size, temp_path in result:
                  
                  with np.load(temp_path) as loaded_file:
                        t_series_slice = loaded_file["t_series"]
                        per_series_slice = loaded_file["per_series"]
                        d_per_series = loaded_file["d_per_series"]

                  hdf5_data = HDF5(path = "data", 
                                   exp_num = exp_number)      
                  hdf5_data.create_group(group_name = "derivatives",
                                         data = (d_per_series,
                                                 t_series_slice,
                                                 per_series_slice),
                                         attr = window_size)
      
      finally: remove_cache_folder(folder_name = cache_folder_name,
                                   pid = pid)

      print(f"running time: {time() - time_0} s\nderivative_calculations.py completed successfully")
