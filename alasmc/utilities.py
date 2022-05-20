import numpy as np


def get_model_id(model: np.ndarray):
    """
    Returns the binary number associated to a model
    """
    return int('0b' + ''.join((model * 1).astype(str)), 2)


def model_id_to_vector(model_id: int, p: int):
    return np.array(list(np.binary_repr(model_id).zfill(p))).astype(np.int8)


def unzip(l: list):
    return tuple(zip(*l))


def create_model_matrix(X: np.ndarray):
    p = X.shape[1]
    model_matrix = np.ndarray((2**p, p))
    for model_id in range(2**p):
        model_str = bin(model_id)[2:]
        model_str = '0'*(p - len(model_str)) + model_str
        model_matrix[model_id, ] = np.array(list(model_str), dtype=int)
        model_matrix = model_matrix == 1
    return model_matrix


def create_data(n: int, n_covariates: int, n_active: int, rho: float, model: object):
    beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.ones(n_active)])
    n = n
    rho = rho
    sigma_x = np.diag([1.0] * n_covariates)
    sigma_x[np.triu_indices(n_covariates, 1)] = rho
    sigma_x[np.tril_indices(n_covariates, -1)] = rho
    X = np.random.multivariate_normal(np.zeros(n_covariates), sigma_x, n)
    y = model.sample(X, beta_true)
    return X, y, beta_true
