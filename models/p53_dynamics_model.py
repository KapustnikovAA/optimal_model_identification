import numpy as np
from numpy.linalg import lstsq as MNK
from scipy.optimize import least_squares as NMNK

from .basic_algorithms_class import General_Solver

class p53_dynamics_model(General_Solver):
    """
    Default attributes:\n
        self.method = "RK45"
        self.max_step  = 2**-5\n
    Name of parameters of model: \n    
        a = float; b = float; c = float;
        d = float; n = float; z = float;
        tau1 = float; tau2 = float;
        m = float; r = float
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
        self.model_name = "p53"
    
    def ODE_equations (self, 
                       t: np.ndarray, 
                       initial_vectors: np.ndarray, 
                       a: float, 
                       b: float, 
                       c: float, 
                       d: float, 
                       n: float, 
                       z: float, 
                       tau1: float, 
                       tau2: float, 
                       m: float, 
                       r: float) -> np.ndarray:
        """
        Name of parameters: \n
        \ta = float; b = float; c = float;
        \td = float; n = float; z = float;
        \ttau1 = float; tau2 = float;
        \tm = float; r = float
        """
        u, v = initial_vectors[::2], initial_vectors[1::2]
    
        dl = np.empty(2)
        
        dl[0] = (-u * (r * (u * u - (a + b) * u) + a * b - r * d + c * v)) / tau1
        dl[1] = (z + m * u - n * v) / tau2
        
        return dl
    
    def __call__(self, **kwargs):
        return self.ODE_solver({**kwargs, "order": ("a", 
                                                    "b", 
                                                    "c", 
                                                    "d", 
                                                    "n", 
                                                    "z", 
                                                    "tau1", 
                                                    "tau2", 
                                                    "m", 
                                                    "r")})

    def reconstruct_coefficients (self, 
                                  Series: np.ndarray, 
                                  Vector: np.ndarray) -> np.ndarray:
        """Result --- approximation coefficients\n
          for Φ_analytical(u) by least square method"""

        A = np.empty((len(Vector), 4))
        A[:, 0] = Series**3
        A[:, 1] = Series**2
        A[:, 2] = Series
        A[:, 3] = Series * np.log(np.abs(Series))

        B = np.empty((len(Vector), 1))
        B[:, 0] = Vector 

        coef = MNK(A, B, rcond = None)[0]
        
        return coef.T[0]

    def phi_analytical (self, 
                        c: np.ndarray, 
                        Series: np.ndarray) -> np.ndarray:
        """
        Φ(u) --- analytical function for p53 dynamics model\n
        c --- coefficients of model\n
        Series --- variable of model 
        """
        return c[0] * (Series**3) + c[1] * (Series**2) + c[2] * Series + c[3] * Series * np.log(np.abs(Series)) 

    def help(self) -> None:
        print("p53_dynamics_model(a = 5, b = 10, c = 15, d = 70, n = 2.4, z = 2.1,")
        print("\ttau1 = 1, tau2 = 1, m = 1.5, r = 1, x0 = np.array([5.0, 2.0]))")
        print("N = 400000, Ntrans = 5000, dt = 2**-10")
