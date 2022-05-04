import numpy as np

def get_model_id(model: np.ndarray):
    """
    Returns the binary number associated to a model
    """
    return int('0b' + ''.join((model * 1).astype(str)), 2)
