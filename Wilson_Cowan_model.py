import numpy as np
from numpy.linalg import lstsq as MNK
from scipy.optimize import least_squares as NMNK
from functools import partial

from .basic_algorithms_class import General_Solver

class Wilson_Cowan_model(General_Solver):
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
                 poly_degree: int = 4, 
                 initial_guesses: float = 0) -> None:
        super().__init__(N, Ntrans, dt, window, poly_degree)
        self.model_name = "WC"
        self.initial_guesses = initial_guesses
    
    def ODE_equations (self, 
                       t: np.ndarray, 
                       initial_vectors: np.ndarray, 
                       C: float, 
                       tau: float, 
                       I: float, 
                       k1: float, 
                       k2: float, 
                       k3: float, 
                       k4: float, 
                       k5: float, 
                       k6: float, 
                       k7: float, 
                       k8: float) -> np.ndarray:
        """
        Name of parameters: \n
        \tC = float; tau = float; I = float;
        \tk1 = float; k2 = float; k3 = float;
        \tk4 = float; k5 = float; k6 = float;
        \tk7 = float; k8 = float
        """
        u, v = initial_vectors[::2], initial_vectors[1::2]
    
        dl = np.empty(2)
	
        dl[0] = ((I - k2 * k4) - (k1 * k4 + k2) * u - (k3 * k4 + k1) * (u**2) - k3 * (u**3) - (k5 * k6 + k5 * u) * v) / C
        dl[1] = (k7 * u + k8 - v) / tau
        
        return dl

    def __call__(self, **kwargs):
        return self.ODE_solver({**kwargs, "order": ("C", 
                                                    "tau", 
                                                    "I", 
                                                    "k1", 
                                                    "k2", 
                                                    "k3", 
                                                    "k4", 
                                                    "k5", 
                                                    "k6", 
                                                    "k7", 
                                                    "k8")})
    
    def difference_error(self, 
                         c: np.ndarray, 
                         Series: np.ndarray, 
                         Vector: np.ndarray) -> np.ndarray:
        return self.phi_analytical(c, Series) - Vector

    def reconstruct_coefficients (self, 
                                  Series: np.ndarray, 
                                  Vector: np.ndarray) -> np.ndarray:
        """Result --- approximation coefficients\n
          for Φ_analytical(u) by least square method"""

        A = np.empty((len(Vector), 4))
        A[:, 0] = Series**0
        A[:, 1] = Series**1
        A[:, 2] = Series**2
        A[:, 3] = Series**3

        B = np.empty((len(Vector), 1))
        B[:, 0] = Vector

        coef = MNK(A, B, rcond = None)[0]

        coef_0 = self.initial_guesses * np.ones(7)
        coef_0[:len(coef.T[0])] = coef.T[0]

        optimize_roots = partial(self.difference_error, Series = Series, Vector = Vector)
        jac_WC = partial(self.jac_phi_analytical, Series = Series, Series2 = A[:, 2], Series3 = A[:, 3],Vector = Vector)
        roots = NMNK(optimize_roots, coef_0, method = 'lm', jac = jac_WC)

        return roots.x

    def phi_analytical (self, 
                        c: np.ndarray, 
                        Series: np.ndarray) -> np.ndarray:
        """
        Φ(u) --- analytical function for Wilson-Cowan model\n
        c --- coefficients of model\n
        Series --- variable of model 
        """
        return c[0] + c[1] * Series + c[2] * Series * Series + c[3] * Series * Series * Series + c[4] * np.log(np.abs(c[5] + Series)) + c[6] * Series * np.log(np.abs(c[5] + Series))

    def jac_phi_analytical (self, 
                            c: np.ndarray, 
                            Series: np.ndarray, 
                            Vector: np.ndarray, 
                            Series2: np.ndarray, 
                            Series3: np.ndarray,)-> np.ndarray:
        """
        Jacobian for Wilson-Cowan model
        """
        J = np.empty((Series.shape[0], c.shape[0]))
        J[:, 0] = 1
        J[:, 1] = Series
        J[:, 2] = Series2
        J[:, 3] = Series3
        J[:, 4] = np.log(np.abs(c[5] + Series))
        J[:, 5] = (c[4] + c[6] * Series) / (c[5] + Series)
        J[:, 6] = Series * np.log(np.abs(c[5] + Series))

        return J

    def help(self) -> None:
        print("Wilson_Cowan_mdoel(C = 0.8, tau = 1.9, I = 1, k1 = 47.71, k2 = 17.81, k3 = 32.63,")
        print("\tk4 = -0.55, k5 = 26, k6 = 0.92, k7 = 1.35, k8 = 1.03, x0 = np.array([0.0, 0.0]))")
        print("N = 100000, Ntrans = 5000, dt = 2**-10")
