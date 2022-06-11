import numpy as np


def get_model_id(model: np.ndarray, force_intercept: bool = False) -> str:
    """
    Returns the string of zeros and ones (without left zeros) associated to a model, i.e.:
    array([False, True, True]) -> '11'.

    Note that model without covariates corresponds to an empty string, that is: array([False, False, False]) -> ''.
    """
    start = 1 if force_intercept else 0
    return ''.join((model[start:] * 1).astype(str)).lstrip('0')


# def get_model_id(model: np.ndarray):
#     """
#     Returns the binary number associated to a model
#     """
#     return int('0b' + ''.join((model * 1).astype(str)), 2)

def model_id_to_vector(model_id: str, p: int, force_intercept: bool = False) -> np.ndarray:
    if force_intercept:
        return np.append(True, np.array(list(map(int, model_id.rjust(p - 1, '0'))), dtype=bool))
    else:
        return np.array(list(map(int, model_id.rjust(p, '0'))), dtype=bool)

# def model_id_to_vector(model_id: int, p: int):
#     return np.array(list(np.binary_repr(model_id).zfill(p))).astype(np.int8)


def unzip(li: list) -> tuple[np.ndarray]:
    return tuple([np.array(elem, dtype=object) for elem in zip(*li)])


def create_model_matrix(X: np.ndarray, force_intercept: bool = False) -> np.ndarray:
    X_var = X.var(axis=0)
    assert sum(X_var == 0.) == 1 and not X_var[0]
    p = X.shape[1]
    model_matrix = np.repeat(False, 2**(p - 1 * force_intercept) * p).reshape(2**(p - 1 * force_intercept), p)
    for model_num in range(2**(p - 1 * force_intercept)):
        model_id = bin(model_num)[2:].lstrip('0')
        model_matrix[model_num, :] = model_id_to_vector(model_id, p, force_intercept)
    return model_matrix


def model_id_to_model_num(model_id: str, force_intercept: bool = False) -> int:
    start = 1 if force_intercept else 0
    return int('0b' + model_id[start:], 2)


def has_intercept(X: np.ndarray) -> bool:
    return any(X.var(axis=0) == 0.)


def prepare_X(X: np.ndarray) -> np.ndarray:
    if has_intercept(X):
        column_means = X.mean(axis=0)
        where_intercepts = X.var(axis=0) == 0.
        intercept_value = column_means[where_intercepts][0]
        X = X[:, ~where_intercepts].copy()
        X = add_intercept(X, intercept_value)
    else:
        X = add_intercept(X)
    return X


def add_intercept(X: np.ndarray, intercept_value: float or int = 1.) -> np.ndarray:
    return np.column_stack([np.repeat(intercept_value, X.shape[0]), X])


def full_postProb(postProb: dict, p: int, force_intercept: bool = False) -> np.ndarray:
    postProb_full = np.zeros(2**(p - 1 * force_intercept))
    for model_id in postProb:
        model_num = model_id_to_model_num(model_id, force_intercept)
        postProb_full[model_num] = postProb[model_id]
    return postProb_full


def chi_squared_distance(x: np.ndarray, y: np.ndarray):
    sum_x_y = x + y
    support = sum_x_y != 0.
    return 1 / 2 * np.sum((x - y)[support]**2 / sum_x_y[support])


def euclidean_distance(x: np.ndarray, y: np.ndarray):
    return np.sum((x - y)**2)**0.5


def create_data(n: int, rho: float, model: object, beta_true: np.ndarray, intercept: float = 0.) -> tuple:
    n = n
    rho = rho
    n_covariates = len(beta_true)
    sigma_x = np.diag([1.0] * n_covariates)
    sigma_x[np.triu_indices(n_covariates, 1)] = rho
    sigma_x[np.tril_indices(n_covariates, -1)] = rho
    X = np.random.multivariate_normal(np.zeros(n_covariates), sigma_x, n)
    X = add_intercept(X)
    y = model.sample(X, np.append(intercept, beta_true))
    return X, y


def l0_norm(vector: np.ndarray) -> float:
    return np.sum(vector != 0.)
