import numpy as np

import numpy as np

def MMV_Psi(x, lam):
    """
    Maps x in [0,1] to distorted probability measure under lambda.
    """
    return 1 - (1 - x ** (1 / (lam + 1))) ** (1 + lam)

def MMV_Phi(a, lam):
    """
    Computes Phi(a) = 1 - a / (1 + a^{1/lam})^{lam+1}
    """
    return 1 - a / (1 + a ** (1 / lam)) ** (lam + 1)

