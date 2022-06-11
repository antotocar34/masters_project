from abc import ABC, abstractmethod

import numpy as np
from .utilities import l0_norm


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

    @staticmethod
    @abstractmethod
    def mean_func(linpred):
        ...

    @classmethod
    def gradient(cls, Xt: np.ndarray, y: np.ndarray, linpred: np.ndarray):
        return Xt @ (cls.db(linpred) - y)

    @classmethod
    def hessian(cls,
                X: np.ndarray,
                Xt: np.ndarray,
                linpred: np.ndarray,
                phi: float = 1,
                adjusted_curvature: bool = False,
                coef: np.ndarray = None,
                y: np.ndarray = None):
        if adjusted_curvature and (coef is None or y is None):
            raise ValueError("The curvature adjustment requires additional arguments: coef and y.")
        d2b = cls.d2b(linpred)
        rho = 1 if not adjusted_curvature else np.sum((y - cls.mean_func(linpred))**2 / (phi * d2b)) / (len(y) -
                                                                                                        l0_norm(coef))
        return rho * (Xt * d2b) @ X / phi


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

    @staticmethod
    def mean_func(linpred):
        return 1 / (1 + np.exp(-linpred))


class PoissonRegression(GLM):

    @staticmethod
    def sample(X: np.ndarray, beta_true: np.ndarray):
        _lambda = np.exp(X @ beta_true)
        n = X.shape[0]
        y = np.random.poisson(lam=_lambda, size=n)
        return y

    @staticmethod
    def loglikelihood(y, linpred):
        _lambda = np.exp(linpred)
        return y.dot(linpred) - sum(_lambda)  # without the part which depends solely on y: - sum(y_i!)

    @staticmethod
    def mean_func(linpred):
        return np.exp(linpred)

    @staticmethod
    def db(linpred):
        return np.exp(linpred)

    @staticmethod
    def d2b(linpred):
        return np.exp(linpred)
