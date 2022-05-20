from abc import ABC, abstractmethod

import numpy as np


class GLM(ABC):

    @staticmethod
    @abstractmethod
    def loglikelihood(y, linpred):
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
    @abstractmethod
    def sample(X: np.ndarray, beta_true: np.ndarray):
        ...

    @classmethod
    def gradient(cls, Xt: np.ndarray, y: np.ndarray, linpred: np.ndarray):
        return Xt @ (cls.db(linpred) - y)

    @classmethod
    def hessian(cls, X: np.ndarray, Xt: np.ndarray, linpred: np.ndarray):
        return (Xt * cls.d2b(linpred)) @ X


class BinomialLogit(GLM):

    @staticmethod
    def sample(X: np.ndarray, beta_true: np.ndarray):
        p = 1 / (1 + np.exp(- X @ beta_true))
        n = X.shape[0]
        y = np.random.binomial(1, p, n)
        return y

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
