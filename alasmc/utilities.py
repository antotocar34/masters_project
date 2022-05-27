import numpy as np


def get_model_id(model: np.ndarray) -> str:
    """
    Returns the string of zeros and ones (without left zeros) associated to a model, i.e.:
    array([False, True, True]) -> '11'.

    Note that model without covariates corresponds to an empty string, that is: array([False, False, False]) -> ''.
    """
    return ''.join((model * 1).astype(str)).lstrip('0')


# def get_model_id(model: np.ndarray):
#     """
#     Returns the binary number associated to a model
#     """
#     return int('0b' + ''.join((model * 1).astype(str)), 2)

def model_id_to_vector(model_id: str, p: int) -> np.ndarray:
    return np.array(list(map(int, model_id.rjust(p, '0'))), dtype=bool)

# def model_id_to_vector(model_id: int, p: int):
#     return np.array(list(np.binary_repr(model_id).zfill(p))).astype(np.int8)


def unzip(li: list) -> tuple[np.ndarray]:
    return tuple([np.array(elem, dtype=object) for elem in zip(*li)])


def create_model_matrix(X: np.ndarray) -> np.ndarray:
    p = X.shape[1]
    model_matrix = np.repeat(False, 2**p * p).reshape(2**p, p)
    for model_num in range(2**p):
        model_id = bin(model_num)[2:].lstrip('0')
        model_matrix[model_num, :] = model_id_to_vector(model_id, p)
    return model_matrix


def model_id_to_model_num(model_id: str) -> int:
    return int('0b' + model_id, 2)


def has_intercept(X: np.ndarray) -> bool:
    return any(X.var(axis=0) == 0.)


def prepare_X(X: np.ndarray) -> np.ndarray:
    pass


def full_postProb(postProb: dict, p: int) -> np.ndarray:
    postProb_full = np.zeros(2**p)
    for model_id in postProb:
        model_num = model_id_to_model_num(model_id)
        postProb_full[model_num] = postProb[model_id]
    return postProb_full


def chi_squared_distance(x: np.ndarray, y: np.ndarray):
    return 1 / 2 * np.sum((x - y)**2 / (x + y))


def euclidean_distance(x: np.ndarray, y: np.ndarray):
    return np.sum((x - y)**2)**0.5

def create_data(n: int, n_covariates: int, n_active: int, rho: float, model: object, beta_true: np.ndarray) -> tuple:
    n = n
    rho = rho
    sigma_x = np.diag([1.0] * n_covariates)
    sigma_x[np.triu_indices(n_covariates, 1)] = rho
    sigma_x[np.tril_indices(n_covariates, -1)] = rho
    X = np.random.multivariate_normal(np.zeros(n_covariates), sigma_x, n)
    y = model.sample(X, beta_true)
    return X, y


def l0_norm(vector: np.ndarray, atol: float = 1e-12) -> float:
    return np.sum(~np.isclose(vector, 0, atol=atol))
