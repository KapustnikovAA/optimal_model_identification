#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any

from scripts.model_selection import get_relative_error_to_min_ratio, model_identification

from utils.data_saver import HDF5

def real_model_error_ratio_calculation (exp_range: Any | None = None) -> List:

    real_model_error_ratio = []
    true_model_list = []
    for data in get_relative_error_to_min_ratio(exp_range = exp_range):

        exp_number = data["exp_number"]
        hdf5_data = HDF5(path = "data",
                         exp_num = exp_number)
        true_model = hdf5_data.load_group(group_name = "metadata")["integation_parameters"]["model"]
        true_model_index = data["complexity"].index(true_model)
        
        true_model_error = data["min_ratio"][true_model_index]
        absolute_min_error = data["min_ratio"].min()

        real_model_error_ratio.append(float(true_model_error / absolute_min_error))
        true_model_list.append(true_model)

    np.savetxt('data/distribution_ratio.txt', real_model_error_ratio)

    return real_model_error_ratio, true_model_list

def accuracy_curve_calculation (config_path: str,
                                exp_range: Any | None = None) -> float:

    real_model_error_ratio, true_models_list = real_model_error_ratio_calculation(exp_range = exp_range)
    real_model_error_ratio = sorted(real_model_error_ratio)
    quantile_95_np = np.quantile(real_model_error_ratio, 0.95)

    accuracy_list = []
    delta = 0.01
    thresholds = np.arange(1 + delta, quantile_95_np + delta, delta) 

    for threshold in thresholds:
        
        selection_result = model_identification(config_path = config_path,
                                                threshold = threshold,
                                                exp_range = exp_range)
        
        accuracy_tmp_list = []
        for items, true_model in zip(selection_result, true_models_list):
            selected_model = items[0]
            if selected_model == true_model: accuracy_tmp_list.append(1)

        accuracy_list.append(len(accuracy_tmp_list) / 100)
    
    accuracy = np.array(accuracy_list)
    max_acc = accuracy.max()

    max_indices = np.where(accuracy == max_acc)[0]

    optimal_index = None
    if len(max_indices) >= 3:
        split_indices = np.where(np.diff(max_indices) != 1)[0] + 1
        groups = np.split(max_indices, split_indices)
        
        for g in groups:
            if len(g) >= 3:
                optimal_index = g[len(g) // 2]  # Берем середину этого плато
                break

    if optimal_index is None:
        optimal_index = np.argmax(accuracy)

    optimal_threshold = thresholds[optimal_index]

    print("max: ", max_acc)
    print(f"Optimal threshold: {optimal_threshold}")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(thresholds, accuracy, color='green', linewidth=2, zorder=1)
    ax.scatter(optimal_threshold, max_acc, color='red', marker='*', s=250, zorder=2)

    fig.canvas.draw()
    ymin, ymax = ax.get_ylim()
    y_offset = (ymax - ymin) * 0.025

    ax.text(optimal_threshold, max_acc + y_offset, f"({optimal_threshold:.2f}, {max_acc:.2f})", 
            color='red', fontsize=12, weight='bold', va='bottom', ha='center')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    ax.set_xlabel("Thresholds", fontsize = 22)
    ax.set_ylabel("Accuracy", fontsize = 22)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-0.001, max_acc + 5 * y_offset)
    ax.set_xlim(0.999, thresholds.max() + 5 * delta)

    plt.savefig(f"images/Accuracy(threshold = {round(optimal_threshold, 2)}).pdf")
    plt.close()

    print(f"Accuracy(threshold) figure is save")

    return optimal_threshold