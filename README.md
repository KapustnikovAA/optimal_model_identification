# optimal_model_identification
A method for identifying systems of two ODEs from time series using only one measurable variable is proposed. Identification reduces to recovering a scalar function and selecting the best model. Applicable to models such as Lotka–Volterra, p53, FitzHugh–Nagumo, Morris-Lecar etc. 

## General structure of HDF5-files

```Full structure of file
        Experiment_number.h5
        │
        ├── **metadata**
        │   └── **attrs**
        │       ├── **integation_parameters** --> type: Json_dict
        │       │   └── {**model**: str, 
        │       │        solver: str,
        │       │        **N**: int,
        │       │        Ntrans: int,
        │       │        **dt**: float,
        │       │        max_step: float,
        │       │        **noise_sigma**: float|str}
        │       └── model_parameters --> type: Json_dict
        │
        ├── **raw**
        │   ├── **t_series** --> shape: (number_time_points,), type: np.ndarray
        │   ├── **u_series** --> shape: (number_time_points,), type: np.ndarray
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
