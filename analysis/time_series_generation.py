import numpy as np
from time import time

from models.FitzHugh_Nagumo_model import FitzHugh_Nagumo_model
from models.Lotka_Volterra_model import Lotka_Volterra_model
from models.p53_dynamics_model import p53_dynamics_model
from models.Wilson_Cowan_model import Wilson_Cowan_model
from models.Morris_lecar_model import Morris_lecar_model

from utils.data_saver import HDF5
from utils.yaml_loader import load_yaml

def main (config_path: str, 
          EXPERIMENT_NUMBER: int = 0) -> None:
    
    time_0 = time()

    config = load_yaml(config_path = config_path)

    model_name = config["global_settings"]["model"]
    vary_parameter = config["global_settings"]["vary_parameter"]
    noise_addition = config["global_settings"]["noise"]

    models_dict = {"FHN": FitzHugh_Nagumo_model(N = 400000, dt = 2**-10, Ntrans = 5000),
                   "LV": Lotka_Volterra_model(N = 2000000, Ntrans = 5000, dt = 2**-10),
                   "p53": p53_dynamics_model(N = 400000, Ntrans = 5000, dt = 2**-10),
                   "WC": Wilson_Cowan_model(N = 100000, Ntrans = 5000, dt = 2**-10),
                   "ML": Morris_lecar_model(N = 4000000, Ntrans= 100000, dt = 2**-10)
    }

    parameters_dict = config["models"][model_name]
    parameters_dict["x0"] = np.array(parameters_dict["x0"])

    bifurcation_parameters_dict = {"FHN":
                                   ("Iext", np.round(np.linspace(0.05, 1.7, 10), 2)),
                                   "LV":
                                   ("alpha", np.round(np.linspace(0.05, 1.7, 10), 2)),
                                   "p53":
                                   ("m", np.round(np.linspace(1.275, 2.1, 10), 3)),
                                   "WC":
                                   ("I", np.round(np.linspace(0.05, 1.7, 10), 2)),
                                   "ML":
                                   ("Iext", np.round(np.linspace(0.05, 1.7, 10), 2))
    }

    if vary_parameter:
        model = models_dict[model_name]
        model_param = parameters_dict.copy()

        model_metadata_dict = {"model": model.model_name,
                               "solver": model.method,
                               "N": model.N,
                               "Ntrans": model.Ntrans,
                               "dt": model.dt,
                               "max_step": model.max_step,
                               "noise_sigma": None
        }
        
        bif_param, range_param = bifurcation_parameters_dict[model_name]

        for param in range_param:
            model_param[bif_param] = float(param)

            result_model = model(**model_param)

            if noise_addition:

                EXPERIMENT_NUMBER += 1
                model_metadata_dict["noise_sigma"] = 0

                model_file = HDF5(path = "data",
                                  exp_num = EXPERIMENT_NUMBER)
                model_file.create_experiment(integ_param = model_metadata_dict,
                                             model_param = model_param,
                                             data = (result_model.t, 
                                                     result_model.y[0], 
                                                     result_model.y[1]))

                for noise_sigma in (1e-3, 1e-2, 1e-1):
                    
                    EXPERIMENT_NUMBER += 1
                    model_metadata_dict["noise_sigma"] = noise_sigma

                    u_noise = model.add_noise(Series = result_model.y[0], 
                                              k = noise_sigma)
                    v_noise = model.add_noise(Series = result_model.y[1], 
                                              k = noise_sigma)
                    
                    model_file = HDF5(path = "data",
                                  exp_num = EXPERIMENT_NUMBER)
                    model_file.create_experiment(integ_param = model_metadata_dict,
                                                 model_param = model_param,
                                                 data = (result_model.t,
                                                         u_noise,
                                                         v_noise))

            else:

                EXPERIMENT_NUMBER += 1
                model_metadata_dict["noise_sigma"] = 0

                model_file = HDF5(path = "data",
                                  exp_num = EXPERIMENT_NUMBER)
                model_file.create_experiment(integ_param = model_metadata_dict,
                                             model_param = model_param,
                                             data = (result_model.t, 
                                                     result_model.y[0], 
                                                     result_model.y[1]))
    
    else:

        model = models_dict[model_name]
        model_param = parameters_dict.copy()

        model_metadata_dict = {"model": model_name,
                               "solver": model.method,
                               "N": model.N,
                               "Ntrans": model.Ntrans,
                               "dt": model.dt,
                               "max_step": model.max_step,
                               "noise_sigma": None
        }

        result_model = model(**model_param)

        if noise_addition:

            EXPERIMENT_NUMBER += 1
            model_metadata_dict["noise_sigma"] = 0

            model_file = HDF5(path = "data",
                                  exp_num = EXPERIMENT_NUMBER)
            model_file.create_experiment(integ_param = model_metadata_dict,
                                         model_param = model_param,
                                         data = (result_model.t,
                                                 result_model.y[0],
                                                 result_model.y[1]))

            for noise_sigma in (1e-3, 1e-2, 1e-1):

                EXPERIMENT_NUMBER += 1
                model_metadata_dict["noise_sigma"] = noise_sigma

                u_noise = model.add_noise(Series = result_model.y[0], 
                                              k = noise_sigma)
                v_noise = model.add_noise(Series = result_model.y[1], 
                                            k = noise_sigma)

                model_file = HDF5(path = "data",
                                  exp_num = EXPERIMENT_NUMBER)
                model_file.create_experiment(integ_param = model_metadata_dict,
                                             model_param = model_param,
                                             data = (result_model.t,
                                                     u_noise,
                                                     v_noise))

        else:

            EXPERIMENT_NUMBER += 1
            model_metadata_dict["noise_sigma"] = 0

            model_file = HDF5(path = "data",
                                  exp_num = EXPERIMENT_NUMBER)
            model_file.create_experiment(integ_param = model_metadata_dict,
                                         model_param = model_param,
                                         data = (result_model.t,
                                                 result_model.y[0],
                                                 result_model.y[1]))
    
    print(f"running time: {time() - time_0} s\ntime_series_generation.py completed successfully")