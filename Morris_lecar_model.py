import numpy as np
from numpy.linalg import lstsq as MNK
from scipy.optimize import least_squares as NMNK
from functools import partial

from .basic_algorithms_class import General_Solver

class Morris_lecar_model(General_Solver):
    """
    Default attributes:\n
        self.method = "RK45"
        self.max_step  = 2**-5\n
    Name of parameters of model: \n    
        C = float; Iext = float; Isyn = float;
        gL = float; gCa = float; gK = float;
        uL = float; uCa = float; uK = float;
        v1 = float; v2 = float; v3 = float;
        v4 = float; phi = float
    Name of parameters of ODE solver:\n
        x0 = ndarray; method = str; max_step = float
    """
    def __init__(self, 
                 N: int = 10000, 
                 Ntrans: int = 100000, 
                 dt: float = 2 ** -10, 
                 window: int = 5, 
                 poly_degree: int = 4, 
                 initial_guesses: float = 1) -> None:
        super().__init__(N, Ntrans, dt, window, poly_degree)
        self.model_name = "ML"
        self.initial_guesses = initial_guesses
    
    def ODE_equations (self, 
                       t: np.ndarray, 
                       initial_vectors: np.ndarray, 
                       C: float, 
                       Iext: float, 
                       Isyn: float, 
                       gL: float, 
                       gCa: float, 
                       gK: float, 
                       uL: float, 
                       uCa: float, 
                       uK: float, 
                       v1: float, 
                       v2: float, 
                       v3: float, 
                       v4: float, 
                       phi: float) -> np.ndarray:
        """
        Name of parameters: \n
        \tC = float; Iext = float; Isyn = float;
        \tgL = float; gCa = float; gK = float;
        \tuL = float; uCa = float; uK = float;
        \tv1 = float; v2 = float; v3 = float;
        \tv4 = float; phi = float
        """
        u, v = initial_vectors[::2], initial_vectors[1::2]
    
        dl = np.empty(2)
	
        M_inf = 0.5 * (1 + np.tanh((u - v1) / v2))
        n_inf = 0.5 * (1 + np.tanh((u - v3) / v4))
        tau_n = 1 / (phi * np.cosh((u - v3) / (2 * v4)))
	
        dl[0] = (Iext - Isyn - gL * (u - uL) - gCa * (M_inf) * (u - uCa) - gK * v * (u - uK)) / C
        dl[1] = (n_inf - v) / tau_n
        
        return dl

    def __call__(self, **kwargs):
        return self.ODE_solver({**kwargs, "order": ("C", 
                                                    "Iext", 
                                                    "Isyn", 
                                                    "gL", 
                                                    "gCa", 
                                                    "gK", 
                                                    "uL", 
                                                    "uCa", 
                                                    "uK", 
                                                    "v1", 
                                                    "v2", 
                                                    "v3", 
                                                    "v4", 
                                                    "phi")})

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
        
        A = np.empty((len(Vector), 2))
        A[:, 0] = Series**0
        A[:, 1] = Series**1

        B = np.empty((len(Vector), 1))
        B[:, 0] = Vector

        coef_0 = MNK(A, B, rcond = None)[0]

        coef = self.initial_guesses * np.ones(12)
        coef[:coef_0.shape[0]] = coef_0.T[0]

        optimize_roots = partial(self.difference_error, Series = Series, Vector = Vector)
        jac_ML = partial(self.jac_phi_analytical, Series = Series, Vector = Vector)
        roots = NMNK(optimize_roots, coef, method = 'lm', jac = jac_ML)

        return roots.x

    def phi_analytical (self, 
                        c: np.ndarray, 
                        Series: np.ndarray) -> np.ndarray:
        """
        Φ(u) --- analytical function for Morris-lecar model\n
        c --- coefficients of model\n
        Series --- variable of model 
        """
        return c[0] + c[1] * Series + c[2] * np.tanh((Series + c[3]) / c[4]) + c[5] * Series * np.tanh((Series + c[3]) / c[4]) + (c[6] * np.sinh((Series + c[7]) / c[8]) * (c[9] - ((Series + c[10]) / c[8])**2) / (Series + c[10])**2) + (c[6] * c[8] * np.cosh((Series + c[7]) / c[8]) * (c[11] - ((Series + c[10]) / c[8])**2) / (Series + c[10])**3)

    def jac_phi_analytical (self, 
                            c: np.ndarray, 
                            Series: np.ndarray, 
                            Vector: np.ndarray)-> np.ndarray:
        """
        Jacobian for Morris-Lecar model
        """
        J = np.empty((Series.shape[0], c.shape[0]))
        
        # Повторяющиеся подвыражения
        tanh_arg = (c[3] + Series) / c[4]
        tanh_val = np.tanh(tanh_arg)
        tanh_deriv = 1 - tanh_val**2

        cosh_arg = (c[7] + Series) / c[8]
        cosh_val = np.cosh(cosh_arg)
        sinh_val = np.sinh(cosh_arg)

        c10s = c[10] + Series
        c8_2 = c[8] ** 2
        c8_3 = c[8] ** 3
        c10s_2 = c10s ** 2
        c10s_3 = c10s ** 3
        c10s_4 = c10s ** 4

        expr1 = (c[11] - c10s_2 / c8_2)
        expr2 = (c[9] - c10s_2 / c8_2)

        # Заполнение матрицы Якоби
        J[:, 0] = 1
        J[:, 1] = Series
        J[:, 2] = tanh_val
        J[:, 3] = (c[2] + c[5]*Series) * tanh_deriv / c[4]
        J[:, 4] = -(c[2] + c[5]*Series) * tanh_deriv * (c[3] + Series) / c[4]**2
        J[:, 5] = Series * tanh_val
        J[:, 6] = (
            c[8]*expr1*cosh_val/c10s_3 +
            expr2*sinh_val/c10s_2
        )
        J[:, 7] = (
            c[6]*expr1*sinh_val/c10s_3 +
            c[6]*expr2*cosh_val/(c[8]*c10s_2)
        )
        J[:, 8] = (
            c[6]*expr1*cosh_val/c10s_3 -
            c[6]*expr1*(c[7] + Series)*sinh_val/(c[8]*c10s_3) +
            2*c[6]*cosh_val/(c8_2*c10s) -
            c[6]*(c[7] + Series)*expr2*cosh_val/(c8_2*c10s_2) +
            2*c[6]*sinh_val/c8_3
        )
        J[:, 9] = c[6]*sinh_val/c10s_2
        J[:, 10] = (
            -3*c[6]*c[8]*expr1*cosh_val/c10s_4 -
            2*c[6]*expr2*sinh_val/c10s_3 -
            c[6]*2*c10s*cosh_val/(c[8]*c10s_3) -
            c[6]*2*c10s*sinh_val/(c8_2*c10s_2)
        )
        J[:, 11] = c[6]*c[8]*cosh_val/c10s_3

        return J

    def help(self) -> None:
        print("Morris_lecar_model(C = 10, Iext = 0.1, Isyn = 0, gL = 0.5, gCa = 1.33, gK = 2.0,")
        print("\tuL = -0.5,uCa = 1, uK = -1.5, v1 = -0.01, v2 = 0.15, v3 = 0.1,")
        print("\tv4 = 0.145, phi = 0.1, x0 = np.array([0.0, 0.0]))")
        print("N = 4000000, Ntrans= 100000, dt = 2**-10")
