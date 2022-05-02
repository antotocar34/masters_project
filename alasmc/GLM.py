from abc import ABC, abstractmethod

import numpy as np

class GLM(ABC):

    @staticmethod
    @abstractmethod
    def loglikelihood(self, y, linpred):
        pass

    @staticmethod
    @abstractmethod
    def db(linpred):
        pass

    @staticmethod
    @abstractmethod
    def d2b(linpred):
        pass

    @staticmethod
    def gradient(Xt: np.ndarray, y: np.ndarray, db_dlinpred: np.ndarray):
        return Xt @ (db_dlinpred - y)

    @staticmethod
    def hessian(X: np.ndarray, Xt: np.ndarray, d2b_dlinpred2: np.ndarray):
        return (Xt * d2b_dlinpred2) @ X


class LogisticGLM(GLM):
    @staticmethod 
    def loglikelihood(y, linpred):
        p = 1 / (1 + np.exp(-linpred))
        if np.isclose(p, 0).any() or np.isclose(p, 1).any():
            raise Exception("Exact 0 and 1 occur in the computed probabilities.")
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    @staticmethod
    def db(linpred):
        return 1 / (1 + np.exp(-linpred))

    @staticmethod
    def d2b(linpred):
        p = 1 / (1 + np.exp(-linpred))
        return p * (1 - p)
