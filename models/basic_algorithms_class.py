import numpy as np
from numpy.linalg import lstsq as MNK
from scipy.linalg import lstsq as MNK_polydiff

from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares as NMNK

from math import factorial
from os import path 

class General_Solver:
    """
    Default attributes:

        self.method = "RK45"
        self.max_step  = 2**-5
    """
    def __init__(self, 
                 N: int = 10000, 
                 Ntrans: int = 5000, 
                 dt: float = 2**-10, 
                 window: int = 5, 
                 poly_degree: int = 4,
                 julia_path: str | bool = False,
                 julia_mode: bool = False) -> None:
        self.N = N
        self.Ntrans = Ntrans #5000
        self.dt = dt #2**(-10)#-5
        self.window = window #3 #5
        self.poly_degree = poly_degree #2 #4
        self.sl = (self.window - 1) // 2
        self.t = np.linspace(-self.Ntrans * self.dt, self.N * self.dt, self.N + self.Ntrans, False)
        self.method = "RK45"
        self.max_step  = 2**-5

        if julia_mode:

            current_dir = path.dirname(path.abspath(__file__))
            julia_file_path = path.join(current_dir, "General_Solver_Julia_Methods.jl")

            from julia import Julia
            Julia(runtime = julia_path, compiled_modules = True)
            from julia import Main
            self.julia_Main = Main 
            self.julia_Main.include(julia_file_path) # "General_Solver_Julia_Methods.jl"

            self.polynomdif = self.polynomdif_julia
        else: self.polynomdif = self.polynomdif_python
    
    def ODE_solver (self, kwargs):
        """
        Name of parameters of model: \n
        \ta = float; b = float; Iext = float\n
        Name of parameters of ODE solver:\n
        \tx0 = ndarray; method = str; max_step = float
        """

        if ("method" in kwargs.keys()) and ("max_step" in kwargs.keys()):

            self.method = kwargs.pop("method")
            self.max_step = kwargs.pop("max_step")

        elif ("method" in kwargs.keys()): self.method = kwargs.pop("method")
        elif ("max_step" in kwargs.keys()): self.max_step = kwargs.pop("max_step")

        sorted_args = tuple(kwargs[key] for key in kwargs["order"]) 

        solve = solve_ivp(self.ODE_equations,
                          [-self.Ntrans * self.dt, self.N * self.dt],
                            y0 = kwargs.pop("x0"),
                            t_eval = self.t,
                            method = self.method,
                            max_step = self.max_step,
                            dense_output = True, 
                            rtol = 1e-5, 
                            atol = 1e-10,
                            args = sorted_args)
        
        del(kwargs)
        del(sorted_args)

        return solve

    def Taylor_U_function_integral_new (self, 
                                        u: np.ndarray, 
                                        time: np.ndarray, 
                                        i: np.ndarray, 
                                        initial_point: float, 
                                        u0: float) -> np.ndarray:
        return integrate.cumulative_simpson((u - u0)**i, x = time, initial = initial_point, axis = 0)

    def phi_numerical_fast (self, 
                            time: np.ndarray, 
                            u: np.ndarray, 
                            du: np.ndarray, 
                            P: int, 
                            initial_point: float = 0, 
                            u0: float = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Result:
            Phi_numerical --- values (1D-array) of Φ_numerical(u)
            S_error --- error of approximation of Φ_numerical(u)
        """
        N = len(time) - 1

        X = np.empty((N, 2 * (P + 1)))
        Y = np.empty((N, 1))

        indexes_sort = np.argsort(u)
        n, pn = indexes_sort[:-1], indexes_sort[1:]

        u_matrix = np.broadcast_to(u[:, np.newaxis], (len(u), P + 1)) # u для всех P 
        i = np.broadcast_to(np.arange(u_matrix.shape[1]), (len(u), P + 1)) # все i
        U_i_for_all_P = self.Taylor_U_function_integral_new(u_matrix, time, i, initial_point, u0) # как U_all
          
        X[:, :P + 1] = u[pn, np.newaxis] * U_i_for_all_P[pn, :] - u[n, np.newaxis] * U_i_for_all_P[n, :] #X1
        X[:, P + 1:] = U_i_for_all_P[pn, :] - U_i_for_all_P[n, :] # X2
        
        Y[:, 0] = du[pn] - du[n]

        beta = MNK(X, Y, rcond = None)[0]

        S_error = np.abs((du[n] - du[pn]) + (beta[:, 0][:P + 1] * X[:, :P + 1]).sum(axis = 1) + (beta[:, 0][P + 1:] * X[:, P + 1:]).sum(axis = 1))
        Phi_numerical = du - (beta[:, 0][:P + 1] * u[:, np.newaxis] * U_i_for_all_P).sum(axis = 1) - (beta[:, 0][P + 1:] * U_i_for_all_P).sum(axis = 1)
    
        return Phi_numerical, S_error
        
    def polynomdif_python (self,
                    x: np.ndarray, 
                    dt: float, 
                    P: int = 1, 
                    m: int = 3) -> np.ndarray:
        ''' 
        Проводит дифференцирование со сглаживанием полиномом\n
        Аргуметны:
            P --- степень полинома
            m --- количество точек в окне
        Выход:
            dx --- прямоугольная матрица
                столбцы --- n-а производная
                строки --- k-й элемент массива
        '''
        m2 = (m - 1) // 2

        # Формируем безразмерное время:
        tn = np.linspace(-float(m2), float(m2), m)

        # Формируем базисные функции:
        Phi = np.ones((m, P + 1))
        for k in range(1, P + 1):
            Phi[:, k] = Phi[:, k - 1] * tn
            
        # Матрица производных:
        dx = np.empty((len(x) - 2 * m2, P + 1), order = 'F')

        # По всем точкам:
        for n in range(len(x) - 2 * m2):
            dx[n, :] = MNK_polydiff(Phi, x[n: n + m], lapack_driver = 'gelsy')[0]

        # Масштабируем:
        for k in range(1, P + 1):
            dx[:, k] *= factorial(k) / (dt**k)

        return dx
    
    def polynomdif_julia (self, 
                          x: np.ndarray, 
                          dt: float, 
                          P: int = 1, 
                          m: int = 3) -> np.ndarray:
        ''' 
        Проводит дифференцирование со сглаживанием полиномом\n
        Аргуметны:
            P --- степень полинома
            m --- количество точек в окне
        Выход:
            dx --- прямоугольная матрица
                столбцы --- n-а производная
                строки --- k-й элемент массива
        '''
        return np.array(self.julia_Main.fast_polynomdif.polynomdif(x, dt, P, m))
    
    def errors_calculation (self, 
                            time: np.ndarray, 
                            u: np.ndarray, 
                            du: np.ndarray, 
                            Phi_numerical: np.ndarray, 
                            S_error: np.ndarray) -> tuple[float, float, np.ndarray]:
        """
        Result:
            Error_rec --- error in parameter reconstruction between Φ_analytical(u) and Φ_numerical(u)
            Error_approx --- error of approximation of Φ_numerical(u)
            rec_Phi_analytical --- reconstructed Φ_analytical(u) 
        """
        coef = self.reconstruct_coefficients(u, Phi_numerical)
        rec_Phi_analytical = self.phi_analytical(coef, u)
        
        Error_rec = ((Phi_numerical - rec_Phi_analytical) @ (Phi_numerical - rec_Phi_analytical)) / (np.var(Phi_numerical) * len(u))
        Error_approx = S_error @ S_error
        
        return Error_rec, Error_approx, rec_Phi_analytical

    def add_noise (self, 
                   Series: np.ndarray,
                   k: float =  1e-1) -> np.ndarray:
        """Added noise to Series array and create Series_noise array for save values"""
        Series_noise = Series + np.random.normal(0, np.std(Series) * k, 
                                                 size = (len(Series)))
        return Series_noise
