"""import matplotlib
matplotlib.use('TkAgg')
"""
import matplotlib.pyplot as plt
import numpy as np

from utils.data_saver import HDF5

def main (data_path: str, file_name: str) -> None:
    
    for exp_numb in range(301, 302):

        hdf5 = HDF5(path = "data",
                        exp_num = exp_numb)
    
        data_phi = hdf5.load_group(group_name = "phi_functions")
    
        data_u = hdf5.load_group(group_name = "derivatives")["u_slice_series"]
        #data_u  = hdf5.load_group(group_name = "raw")["u_series"]

        plt.plot(data_u)
        plt.title(f"exp_{exp_numb}")

        plt.show()

        print(data_phi["model_pool"])

        model_phi = 2
        # Создаем сетку 3x3
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

        # Превращаем двумерную матрицу графиков в плоский список для удобства итерации
        axes_flat = axes.flatten()

        # Проходим циклом по всем 9 предсказаниям
        for i in range(1, 10):
            ax = axes_flat[i - 1]
            p_key = f'phi_p{i}'
            
            # Отображаем истинные значения (phi_num) синими точками
            # Векторы огромные (404996 элементов), поэтому берем шаг [::100], 
            # чтобы график не завис и точки не перекрывали друг друга.
            ax.scatter(data_u[::100], data_phi['phi_num'][::100], 
                    color='blue', s=1, alpha=0.5, label='phi_num')
            
            # Отображаем предсказания (phi_p1...9) оранжевыми точками (берем 0-ю строку)
            ax.scatter(data_u[::100], data_phi[p_key][model_phi, ::100], 
                    color='orange', s=1, alpha=0.5, label=p_key)
            
            # Настраиваем заголовки и сетку
            ax.set_title(f'Сравнение с {p_key}')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Добавляем легенду только на первый график, чтобы не перегружать картинку
            if i == 1:
                ax.legend(loc='upper right', markerscale=10)

        # Добавляем общие подписи осей для всей фигуры
        fig.text(0.5, 0.04, 'Индекс элемента', ha='center', fontsize=12)
        fig.text(0.04, 0.5, 'Значение phi', va='center', rotation='vertical', fontsize=12)

        plt.tight_layout()
        plt.show()