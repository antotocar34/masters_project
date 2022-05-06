import numpy as np

def get_model_id(model: np.ndarray):
    """
    Returns the binary number associated to a model
    """
    return int('0b' + ''.join((model * 1).astype(str)), 2)

def model_id_to_vector(model_id: int, p: int):
    return np.array(list(np.binary_repr(model_id).zfill(p))).astype(np.int8)
