import numpy as np

from GLM import GLM

def newton_iteration(coef_old, gradient, hessian_inv):
    coef_new = coef_old - hessian_inv @ gradient
    return coef_new
