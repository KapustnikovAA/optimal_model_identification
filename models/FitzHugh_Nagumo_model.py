import numpy as np
from numpy.linalg import lstsq as MNK
from scipy.optimize import least_squares as NMNK

from .basic_algorithms_class import General_Solver

class FitzHugh_Nagumo_model(General_Solver):
    """
    Default attributes:\n
        self.method = "RK45"
        self.max_step  = 2**-5\n
    Name of parameters of model: \n    
        a = float; b = float; Iext = float\n
    Name of parameters of ODE solver:\n
        x0 = ndarray; method = str; max_step = float
    """
    def __init__(self, 
                 N: int = 10000, 
                 Ntrans: int = 5000, 
                 dt: float = 2 ** -10, 
                 window: int = 5, 
                 poly_degree: float = 4) -> None:
         super().__init__(N, Ntrans, dt, window, poly_degree)
         self.model_name = "FHN"

    def ODE_equations (self, 
                       t: np.ndarray, 
                       initial_vectors: np.ndarray, 
                       a: float, 
                       b: float, 
                       Iext: float) -> np.ndarray:
        """
        Name of parameters: \n
        \ta = float; b = float; Iext = float
        """
        u, v = initial_vectors[::2], initial_vectors[1::2]

        dl = np.empty(2)

        dl[0] = u - ((u**3.0) / 3.0) - v + Iext
        dl[1] = u + a - b * v
        
        return dl

    def __call__(self, **kwargs):
        return self.ODE_solver({**kwargs, "order": ("a", "b", "Iext")})

    def reconstruct_coefficients (self, 
                                  Series: np.ndarray, 
                                  Vector: np.ndarray) -> np.ndarray:
        """Result --- approximation coefficients\n 
        for Φ_analytical(u) by least square method"""
        
        A = np.empty((len(Vector), 3))
        A[:, 0] = Series**0
        A[:, 1] = Series**1
        A[:, 2] = Series**3

        B = np.empty((len(Vector), 1))
        B[:, 0] = Vector

        coef = MNK(A, B, rcond = None)[0]
        
        return coef.T[0]
    
    def phi_analytical (self, 
                        c: np.ndarray, 
                        Series: np.ndarray) -> np.ndarray:
        """
        Φ(u) --- analytical function for FitzHugh-Nagumo model\n
        c --- coefficients of model\n
        Series --- variable of model 
        """
        return c[0] + c[1] * Series + c[2] * Series**3
    
    def help(self) -> None:
        print("FitzHugh_Nagumo_model(a = 0.7, b = 0.1, Iext = 0.5, x0 = np.array([1.0, 1.0]))")
        print("N = 400000, dt = 2**-10, Ntrans = 5000")
