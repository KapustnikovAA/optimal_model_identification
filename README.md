# optimal_model_identification
A method for identifying systems of two ODEs from time series using only one measurable variable is proposed. Identification reduces to recovering a scalar function and selecting the best model. Applicable to models such as Lotka–Volterra, p53, FitzHugh–Nagumo, Morris-Lecar etc. 

## General structure of HDF5-files

Each experiment file must have a name in the format:

`Experiment_XXXX.h5`

where `XXXX` is a four‑digit experiment number, for example `Experiment_0000.h5`.

To run the identification pipeline from scratch, the file must contain the minimum required structure, highlighted in **bold** in the diagram below.

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

### `raw`

Contains the original time series data. Depending on the experiment, it may contain time, the main time series, and the time series of the second variable.

### `derivatives`

Contains the results of calculating numerical derivatives.

Since smoothing is applied before calculating the derivatives, the group contains the `window_size` parameter, which defines the size of the smoothing window.

Due to smoothing, several points at the edges of the original time series become unavailable for further calculation. Therefore, in `derivatives` the already truncated versions of time and time series corresponding to the region for which the derivatives were calculated are also saved.

### `phi_functions`

Contains numerical and model phi-functions for all candidate models and for each of the nine polynomials under consideration.

The group also contains the `model_pool` attribute. It is used for automatically loading available models when calculating phi-functions.

Models are automatically imported from the `models` folder if they correctly inherit from the base class `GeneralSolver`. Therefore, to add a new model, you do not need to manually modify the model loading code.

After calculating the phi-functions, the approximation errors are also saved, including the minimum error among the considered polynomials.
