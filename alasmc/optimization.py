import numpy
import abc

from multipledispatch import dispatch
from glm import GLM
from collections.abc import Callable


class NewtonRaphson:
    # Now we only implement the algorithm for generalized linear models.

    def __init__(self):
        pass

    @staticmethod
    @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, object)
    def iteration(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef: numpy.ndarray, linpred: numpy.ndarray,
                  glm: GLM):
        gradient = glm.gradient(Xt, y, linpred)
        hessian = glm.hessian(X, Xt, linpred)
        hessian_inv = numpy.linalg.inv(hessian)
        coef = coef - hessian_inv @ gradient
        return coef, gradient, hessian_inv

    @staticmethod
    @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, Callable, Callable)
    def iteration(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef: numpy.ndarray, linpred: numpy.ndarray,
                  gradient: callable, hessian):
        gradient = gradient(Xt, y, linpred)
        hessian = hessian(X, Xt, linpred)
        hessian_inv = numpy.linalg.inv(hessian)
        coef = coef - hessian_inv @ gradient
        return coef, gradient, hessian_inv

    @staticmethod
    @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, object, float, int)
    def optimize(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef_init: numpy.ndarray, glm: GLM,
                 tol_grad: float, maxit: int = 300):
        coef = coef_init
        linpred = X @ coef
        gradient = glm.gradient(Xt, y, linpred)
        iteration = 0
        while sum(gradient**2) / len(gradient) >= tol_grad and iteration < maxit:
            coef, gradient, _ = NewtonRaphson.iteration(glm, y, X, Xt, coef, linpred)
            linpred = X @ coef
            iteration += 1
            if iteration == maxit:
                print("Newton method has not converged hitting the maximum number of iterations: ", maxit)
        hessian = glm.hessian(X, Xt, linpred)
        hessian_inv = numpy.linalg.inv(hessian)
        return coef, linpred, gradient, hessian_inv

    @staticmethod
    @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, Callable, Callable, float, int)
    def optimize(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef_init: numpy.ndarray, gradient: callable,
                 hessian: callable, tol_grad: float, maxit: int = 300):
        coef = coef_init
        linpred = X @ coef
        gradient = gradient(Xt, y, linpred)
        iteration = 0
        while sum(gradient**2) / len(gradient) >= tol_grad and iteration < maxit:
            coef, gradient, _ = NewtonRaphson.iteration(y, X, Xt, coef, linpred, gradient, hessian)
            linpred = X @ coef
            iteration += 1
            if iteration == maxit:
                print("Newton method has not converged hitting the maximum number of iterations: ", maxit)
        hessian = hessian(X, Xt, linpred)
        hessian_inv = numpy.linalg.inv(hessian)
        return coef, linpred, gradient, hessian_inv
