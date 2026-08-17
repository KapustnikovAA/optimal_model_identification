import numpy as np
from numpy.linalg import lstsq as MNK
from scipy.optimize import least_squares as NMNK

from .basic_algorithms_class import General_Solver

class Lotka_Volterra_model(General_Solver):
    """
    Default attributes:\n
        self.method = "RK45"
        self.max_step  = 2**-5\n
    Name of parameters of model: \n    
        alpha = float; beta = float;
        delta = float; gamma = float\n
    Name of parameters of ODE solver:\n
        x0 = ndarray; method = str; max_step = float
    """
    def __init__(self, 
                 N: int = 10000, 
                 Ntrans: int = 5000, 
                 dt: float = 2 ** -10, 
                 window: int = 5, 
                 poly_degree: int = 4) -> None:
        super().__init__(N, Ntrans, dt, window, poly_degree)
        self.model_name = "LV"
    
    def ODE_equations (self, 
                       t: np.ndarray, 
                       initial_vectors: np.ndarray, 
                       alpha: float, 
                       beta: float, 
                       delta: float, 
                       gamma: float) -> np.ndarray:
        """
        Name of parameters: \n
        \talpha = float; beta = float; 
        \tdelta = float; gamma = float
        """
        u, v = initial_vectors[::2], initial_vectors[1::2]

        dl = np.empty(2)
        
        dl[0] = alpha * u - beta * u * v
        dl[1] = delta * u * v - gamma * v
        
        return dl
    
    def __call__(self, **kwargs):
        return self.ODE_solver({**kwargs, "order": ("alpha", "beta", "delta", "gamma")})
    
    def reconstruct_coefficients (self, 
                                  Series: np.ndarray, 
                                  Vector: np.ndarray) -> np.ndarray:
        """Result --- approximation coefficients\n 
        for Φ_analytical(u) by least square method"""

        A = np.empty((len(Vector), 3))
        A[:, 0] = Series
        A[:, 1] = Series**2
        A[:, 2] = Series * np.log(np.abs(Series))

        B = np.empty((len(Vector), 1))
        B[:, 0] = Vector

        coef = MNK(A, B, rcond = None)[0]
        
        return coef.T[0]

    def phi_analytical (self, 
                        c: np.ndarray, 
                        Series: np.ndarray) -> np.ndarray:
        """
        Φ(u) --- analytical function for Lotka-Volterra model\n
        c --- coefficients of model\n
        Series --- variable of model 
        """
        return c[0] * Series + c[1] * Series * Series + c[2] * Series * np.log(np.abs(Series)) 
    
    def help(self) -> None:
        print("Lotka_Volterra_model(alpha = 0.1, beta = 0.02 , delta = 0.02,")
        print("\tgamma = 0.4, x0 = np.array([10.0, 100.0]))")
        print("N = 400000, Ntrans = 5000, dt = 2**-10")
        print("Попробовать более длинный ряд N = 2000000")
