import sys
from utils.yaml_loader import load_yaml

from analysis.time_series_generation import main as ts_generation
from analysis.load_test import main as data_load
from analysis.emperical_threshold import accuracy_curve_calculation as emperical_threshold

from scripts.derivative_calculations import main as deriv_calculation
from scripts.phi_function_approximation import main as phi_func_approx
from scripts.model_selection import model_identification as optimal_model_identification

def main (config_path: str = "configs/pipeline.yaml") -> None:
    
    config = load_yaml(config_path = config_path)
    exp_range = config["experiment_number"]

    args = sys.argv[1:]

    if args[0] == "--all":

        for function_name in config["stages"]:
            function_to_call = globals().get(function_name)
            config_path = config[function_name]["config_path"]

            function_to_call(config_path = config_path,
                             exp_range = exp_range)

    elif args[0] == "--stage":

        function_name = args[-1]
        function_to_call = globals().get(function_name)
        config_path = config[function_name]["config_path"]

        function_to_call(config_path = config_path,
                         exp_range = exp_range)

if __name__ == "__main__":

    """
    #For generation example hdf5 file with time series
    ts_generation(config_path = "configs/config_model.yaml",
                  EXPERIMENT_NUMBER = -1)
    """
    
    main(config_path = "configs/pipeline.yaml")