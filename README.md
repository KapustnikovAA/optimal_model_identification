# optimal_model_identification
A method for identifying systems of two ODEs from time series using only one measurable variable is proposed. Identification reduces to recovering a scalar function and selecting the best model. Applicable to models such as Lotka–Volterra, p53, FitzHugh–Nagumo, Morris-Lecar etc. 

## Pipeline

The identification pipeline takes an **HDF5 (`.h5`) experiment file** as input. The file must already contain the required time series and metadata needed for the identification procedure.

The pipeline processes the experiment file through three sequential stages:

```text
HDF5 experiment file
        ↓
Read time series and metadata
        ↓
Numerical derivative calculation
        ↓
phi-function approximation
        ↓
Model selection
        ↓
Identified model + polynomial degree
```
### 1. Numerical Derivative Calculation

The input HDF5 file is opened, and the required time series and metadata are read. Numerical derivatives are then calculated from the input time series.

The results of this stage are **appended to the existing HDF5 file** without overwriting the original data.

### 2. phi-function Approximation

The calculated numerical derivatives are used to compute and approximate the phi-functions for the candidate models.

The results of this stage are also **appended to the existing HDF5 file**.

### 3. Model Selection

The calculated phi-functions and approximation errors are used to compare the candidate models and select the most appropriate one.

The final result is the **identified model** and the **polynomial degree** corresponding to the selected approximation.

## Running the Project

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/KapustnikovAA/optimal_model_identification
cd optimal_model_identification
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

---

### Full End-to-End Pipeline

Run the complete identification pipeline:

```bash
python main.py --all
```

This runs all identification stages sequentially, from loading the input HDF5 file to the final model selection. The output result is an identified model for each processed time series.

---

### Run a Specific Stage

If some stages have already been completed, a specific stage can be run separately:

```bash
python main.py --stage [STAGE]
```
**Available stages:**
- deriv_calculation
- phi_func_approx
- optimal_model_identification
- emperical_threshold

For example:

```bash
python main.py --stage deriv_calculation
```

This allows the pipeline to continue from a specific stage without repeating calculations that have already been completed.

### Note

Works with `Python 3.12` and above. `Julia 1.11.4` or later is also required for the Julia-based calculations.

## Project Structure

### `analysis/`

Contains auxiliary scripts that are **not required for the end-to-end identification pipeline**.

These scripts are used for additional analysis, empirical calculations, visualization, validation, and generation of additional time series.

---

### `configs/`

Contains YAML configuration files for the project.

* `pipeline.yaml` — the main configuration file used to control the identification pipeline.
* Other configuration files define parameters for individual pipeline stages.

---

### `data/`

Contains experiment data, primarily HDF5 (`.h5`) files.

Experiment files must follow the naming convention:

```text
Experiment_0000.h5
Experiment_0001.h5
Experiment_0002.h5
...
```

See **General structure of HDF5-files** for details.

---

### `images/`

Contains generated plots and other visualizations.

---

### `logs/`

Contains execution logs generated during the calculations.

---

### `models/`

Contains the dynamical system models, base classes, and Julia code used for numerical calculations.

Each specific model inherits from the corresponding base class and defines the equations and parameters required for its numerical integration.

The directory also contains the Julia implementation used for optional Julia-based computations.

---

### `scripts/`

Contains the main stages of the identification pipeline:

* `derivative_calculations.py` — numerical derivative calculation.
* `phi_function_approximation.py` — phi-function approximation.
* `model_selection.py` — candidate model selection.
* `complexity_calculation.py` — сalculates the index of structural complexity of models.

---

### `utils/`

Contains auxiliary components used by the pipeline:

* YAML configuration loading;
* HDF5 file management;
* automatic model importing;
* logging;
* temporary disk-based caching.

The temporary cache is used to store intermediate results on disk and reduce RAM usage when processing large time series. It is removed after the corresponding computation is completed.

## General structure of HDF5-files

Each experiment file must have a name in the format:

`Experiment_XXXX.h5`

where `XXXX` is a four‑digit experiment number, for example `Experiment_0000.h5`.

To run the identification pipeline from scratch, the file must contain the minimum required structure, highlighted in ** ** in the diagram below.

```Full structure of file
        Experiment_number.h5
        │
        ├── **metadata**
        │   └── **attrs**
        │       ├── **integation_parameters** --> type: Json_dict
        │       │   └── {**model**: str = "unknown", 
        │       │        solver: str,
        │       │        **N**: int,
        │       │        Ntrans: int,
        │       │        **dt**: float,
        │       │        max_step: float,
        │       │        **noise_sigma**: float|str = "unknown"}
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
```
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

### `errors`
After calculating the phi-functions, the approximation errors are also saved. Where `all_p_errors` stores the approximation error for each candidate model and each polynomial degree, `min_error` stores the minimum error across all models and polynomial degrees, while `arg_p_min` stores the polynomial degree at which each candidate model achieved its minimum error.

### Experimental Data Requirements

If you are using experimental time series, the structure of the HDF5 file must strictly meet the following requirements:

* **Metadata Nesting:** Inside the file, the `"metadata"` group (within its `attrs` attributes) must contain the `"integation_parameters"` dictionary.
* Within this dictionary, the value for the `model` key must be set to the string value `"unknown"`.
* Within this dictionary, the value for the `noise_sigma` key must be set to the string value `"unknown"`.
* **Parameter Path Example:**
 - `file["metadata"].attrs["integation_parameters"]["model"] = "unknown"`
   
    and
   
 - `file["metadata"].attrs["integation_parameters"]["noise_sigma"] = "unknown"`

## Parallel Computing

### Numerical Derivative Calculation

The numerical derivative calculation stage (`deriv_calculation`) supports parallel execution.

Parallelization can be enabled or disabled in the configuration file `configs/config_derivatives.yaml` using:

```yaml
parallel_calculations: true
```

The number of parallel processes can be configured using:

```yaml
parallel_processes: 9
```

Any suitable number of processes can be specified depending on the available hardware.

#### Julia Support

The numerical derivative calculation can also be performed using Julia.

Julia execution is controlled in the configuration file `configs/config_derivatives.yaml` by:

```yaml
julia_mode: false
julia_path: false
```

By default, `julia_mode` and `julia_path` are set to `false`, and the calculations are performed using Python.
To enable Julia-based calculations, set `julia_mode` to `true` and specify the path to the Julia executable (for example `"/usr/local/bin/julia"`).
The exact path depends on the Julia installation.

---

### phi-function Approximation

The phi-function approximation stage (`phi_func_approx`) also supports parallel execution.

Parallelization can be enabled or disabled in the configuration file `configs/config_approximation.yaml` using:

```yaml
parallel_calculations: true
```

The number of parallel processes can be configured using:

```yaml
parallel_processes: 9
```

The number of processes is limited to **9**, since the approximation is performed for nine polynomial degrees. If a value greater than 9 is specified, the number of processes is limited internally to 9.

## Adding a New Model

The project allows new mathematical models to be added without modifying the main identification pipeline. 
**Note:** The only mandatory manual modification required is in the `scripts/complexity_calculation.py` file. You will need to explicitly update it with the specific formula `(16)` from the preprint (https://www.researchgate.net/publication/400471213_Identification_of_models_described_by_two_differential_equations_from_one_scalar_time_series) to correctly calculate and output the structural complexity index for your new model.

The general structure is:

```text
basic_algorithms_class / General_Solver
            🠉
        New model
```

To add a new model:

1. Create a model class.
2. Inherit it from the required base class `General_Solver` from `models.basic_algorithms_class`.
3. Define the system of differential equations.
4. Define the required model parameters.
5. Define the required reconstruct_coefficients function.
6. Define the required phi_analytical function.
7. Add or update the corresponding configuration if necessary.
8. Verify that the model can be integrated numerically and processed by the identification pipeline.

Models placed in the `models/` directory are automatically imported if they correctly implement the required interface.

### Example Model

```text
class New_model(General_Solver):
    """
    Default attributes:\n
        self.method = "RK45"
        self.max_step  = 2**-5\n
    """
    def __init__(self, 
                 N: int = 10000, 
                 Ntrans: int = 5000, 
                 dt: float = 2 ** -10, 
                 window: int = 5, 
                 poly_degree: float = 4) -> None:
         super().__init__(N, Ntrans, dt, window, poly_degree)
         self.model_name = "new_model"

    def ODE_equations (self, 
                       t: np.ndarray, 
                       initial_vectors: np.ndarray, 
                       param_1: float, 
                       param_2: float, 
                       ...,
                       param_n: float) -> np.ndarray:
        """
        Name of parameters: \n
        \tparam_1 = float; param_2 = float; ...; param_n = float
        """
        u, v = initial_vectors[::2], initial_vectors[1::2]

        dl = np.empty(2)

        dl[0] = ...
        dl[1] = ...
        
        return dl

    def __call__(self, **kwargs):
        return self.ODE_solver({**kwargs, "order": ("param_1", "param_2", ..., "param_n")})

    def reconstruct_coefficients (self, 
                                  Series: np.ndarray, 
                                  Vector: np.ndarray) -> np.ndarray:
        """Result --- approximation coefficients\n 
        for Φ_analytical(u) by least square method"""

        coef = ...
        
        return coef
    
    def phi_analytical (self, 
                        c: np.ndarray, 
                        Series: np.ndarray) -> np.ndarray:
        """
        Φ(u) --- analytical function for FitzHugh-Nagumo model\n
        c --- coefficients of model\n
        Series --- variable of model 
        """
        return ...

    def help(self) -> None:
        print("New_model(param_1 = 0.1, param_2 = 12, ..., param_n= -0.87, x0 = np.array([1.0, 1.0]))")
        print("N = 400000, dt = 2**-10, Ntrans = 5000")
```
