import numpy as np
import tensorly as tl
from dataclasses import dataclass


@dataclass
class Winograd2x3:
    """Transformation matrices for Winograd MFA for F(2, 3) case"""
    # _m – matrix
    B_m = np.array([[1, 0, 0, 0],
                    [0, 1, -1, 1],
                    [-1, 1, 1, 0],
                    [0, 0, 0, -1]])
    G_m = np.array([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])
    A_m = np.array([[1, 0], [1, 1], [1, -1], [0, -1]])

    # _t - tensor
    B_t = tl.tensor(B_m.T)
    G_t = tl.tensor(G_m)
    A_t = tl.tensor(A_m.T)
