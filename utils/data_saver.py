import os
import json
import h5py 
import numpy as np

from typing import Tuple, Dict, Union

class HDF5:

    def __init__(self,
                 path: str,
                 exp_num: int) -> None:
        """
        ```Full structure of file
        Experiment_number.h5
        │
        ├── metadata
        │   └── attrs
        │       ├── integation_parameters --> type: Json_dict
        │       └── model_parameters --> type: Json_dict
        │
        ├── raw
        │   ├── t_series --> shape: (number_time_points,), type: np.ndarray
        │   ├── u_series --> shape: (number_time_points,), type: np.ndarray
        │   └── v_series --> shape: (number_time_points,), type: np.ndarray
        │
        ├── derivatives
        │   ├── attrs
        │   │   └── window_size --> type: int
        │   │
        │   ├── du_series --> shape: (number_time_points,), type: np.ndarray
        │   ├── t_slice_series --> shape: (number_time_points,), type: np.ndarray
        │   └── u_slice_series --> shape: (number_time_points,), type: np.ndarray
        │
        ├── phi_functions
        ├── attrs
        │   │   └── model_pool --> type: Json_dict
        │   │
        │   ├── phi_num --> shape: (number_time_points,), type: np.ndarray 
        │   ├── phi_p1 --> shape: (model, number_time_points), type: np.ndarray
        │   ├── phi_p2 --> shape: (model, number_time_points), type: np.ndarray
        │   ├── ...
        │   └── phi_p9 --> shape: (model, number_time_points), type: np.ndarray
        │
        └── errors
            ├── attrs
            │   └── min_error --> type: float
            │
            ├── arg_P_min --> shape: (model,), type: np.ndarray
            └── all_P_errors --> shape: (model, poly_degree), type: np.ndarray
        ```
        """
        
        self.path = path
        self.exp_num = exp_num
        self.file_name = f"Experiment_{self.exp_num:04d}"
        self.path_to_file = os.path.join(self.path, self.file_name + ".h5")

        os.makedirs(self.path, exist_ok = True)

        self.raw_structure = ("t_series",
                              "u_series",
                              "v_series")
        self.derivatives = ("du_series",
                            "t_slice_series",
                            "u_slice_series")
        self.phi_functions = ("phi_num", *(f"phi_p{item}" for item in range(1, 10)))
        self.errors = ("arg_P_min",
                       "all_P_errors")

        self.possible_attrs = {"derivatives": "window_size",
                               "phi_functions": "model_pool",
                               "errors": "min_error"}
        self.possible_groups = {"derivatives": self.derivatives,
                               "phi_functions": self.phi_functions,
                               "errors": self.errors}

    @staticmethod
    def get_num_experiments(path: str) -> Tuple[int, ...]:
        
        if not os.path.isdir(path):
            return ()
        
        prefix = "Experiment_"
        suffix = ".h5"

        experiments = []
        for fname in os.listdir(path):
            
            if fname.startswith(prefix) and fname.endswith(suffix):
                num_str = fname[len(prefix):-len(suffix)]
                
                if num_str.isdigit():
                    experiments.append(int(num_str))
        
        return tuple(sorted(experiments))

    def create_experiment (self,
                           integ_param: dict,
                           model_param: dict,
                           data: Tuple[np.ndarray, 
                                       np.ndarray, 
                                       np.ndarray]) -> None:
        
        if os.path.exists(self.path_to_file):
            
            existing = HDF5.get_num_experiments(self.path)
            last = max(existing) if existing else 0

            raise FileExistsError(
                f"Experiment {self.exp_num:04d} already exists. "
                f"Last available: {last:04d}"
            )

        model_param["x0"] = tuple(model_param["x0"])

        with h5py.File(self.path_to_file, "w")as f:

            metadata = f.require_group("metadata")
            metadata.attrs["integation_parameters"] = json.dumps(integ_param)
            metadata.attrs["model_parameters"] = json.dumps(model_param)
            
            raw = f.require_group("raw")
            for index, data_item in enumerate(data):
                raw.create_dataset(self.raw_structure[index], data = data_item)

    def create_group (self,
                      group_name: str,
                      data: Tuple,
                      attr: Union[int, float, Dict]) -> None:
        
        if not os.path.exists(self.path_to_file):
            raise FileNotFoundError(
                f"Experiment file {self.exp_num:04d} not found.\nFirst, call create_experiment."
            )

        if group_name not in self.possible_groups:
            raise ValueError(f"Unknown group '{group_name}'")
            
        if len(data) != len(self.possible_groups[group_name]):
            raise ValueError(f"{group_name}: expected {len(self.possible_groups[group_name])} datasets, got {len(data)}")
            
        with h5py.File(self.path_to_file, "a") as f:

            group = f.require_group(group_name)

            for index, selection in enumerate(self.possible_groups[group_name]):
                group.create_dataset(selection, data = data[index])

            if isinstance(attr, dict):
                group.attrs[self.possible_attrs[group_name]] = json.dumps(attr)
            else:
                group.attrs[self.possible_attrs[group_name]] = attr
    
    def load_group(self,
               group_name: str) -> Dict:

        if not os.path.exists(self.path_to_file):
            raise FileNotFoundError(
                f"Experiment file {self.exp_num:04d} not found.\nFirst, call create_experiment."
            )

        with h5py.File(self.path_to_file, "r") as f:

            if group_name not in f:
                raise ValueError(f"Unknown group '{group_name}'")

            group = f[group_name]

            result = {}

            # Attr reader
            for key, value in group.attrs.items():

                if group_name == "metadata":
                    result[key] = json.loads(value)
                elif group_name == "phi_functions":
                    result[key] = json.loads(value)
                else:
                    result[key] = value

            # Group reader
            for key in group.keys():
                result[key] = group[key][:]

            return result