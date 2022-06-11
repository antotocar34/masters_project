import numpy
import abc

from multipledispatch import dispatch
from collections.abc import Callable

from .glm import GLM


class NewtonRaphson:
    # Now we only implement the algorithm for generalized linear models.

    def __init__(self):
        pass

    #@staticmethod
    #@dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, object)
    #def iteration(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef: numpy.ndarray, linpred: numpy.ndarray,
    #              glm: GLM):
    #    gradient = glm.gradient(Xt, y, linpred)
    #    hessian = glm.hessian(X, Xt, linpred)
    #    hessian_inv = numpy.linalg.inv(hessian)
    #    coef = coef - hessian_inv @ gradient
    #    return coef, gradient, hessian_inv

    @staticmethod
    def iteration(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef: numpy.ndarray, linpred: numpy.ndarray,
                  gradient_func: Callable, hessian_func: Callable, **kwargs):
        adjusted_curvature = kwargs.get("adjusted_curvature")
        gradient = gradient_func(Xt, y, linpred)
        if adjusted_curvature:
            hessian = hessian_func(X, Xt, linpred, adjusted_curvature=adjusted_curvature, coef=coef, y=y)
        else:
            hessian = hessian_func(X, Xt, linpred)
        hessian_inv = numpy.linalg.inv(hessian)
        coef = coef - hessian_inv @ gradient
        return coef, gradient, hessian_inv

    # @staticmethod
    # @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, Callable, Callable, int)
    # def iteration(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef: numpy.ndarray, linpred: numpy.ndarray,
    #               gradient_func: Callable, hessian_func: Callable, iter: int, **kwargs):
    #     gradient = gradient_func(Xt, y, linpred)
    #     hessian = hessian_func(X, Xt, linpred)
    #     hessian_inv = numpy.linalg.inv(hessian)
    #     coef = coef - hessian_inv @ gradient / iter**0.5
    #     return coef, gradient, hessian_inv

    # @staticmethod
    # @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, object, float, maxit=int)
    # def optimize(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef_init: numpy.ndarray, glm: GLM,
    #              tol_grad: float, maxit: int = 1000):
    #     coef = coef_init
    #     linpred = X @ coef
    #     gradient = glm.gradient(Xt, y, linpred)
    #     iteration = 0
    #     while sum(gradient**2) / len(gradient) >= tol_grad and iteration < maxit:
    #         coef, gradient, _ = NewtonRaphson.iteration(glm, y, X, Xt, coef, linpred)
    #         linpred = X @ coef
    #         iteration += 1
    #         if iteration == maxit:
    #             print("Newton method has not converged hitting the maximum number of iterations: ", maxit)
    #     hessian = glm.hessian(X, Xt, linpred)
    #     hessian_inv = numpy.linalg.inv(hessian)
    #     return coef, linpred, gradient, hessian_inv

    @staticmethod
    def optimize(y: numpy.ndarray, X: numpy.ndarray, Xt: numpy.ndarray, coef_init: numpy.ndarray, gradient_func: Callable,
                 hessian_func: Callable, tol_grad: float, maxit: int = 1000, **kwargs):
        adjusted_curvature = kwargs.get("adjusted_curvature")
        coef = coef_init
        linpred = X @ coef
        gradient = gradient_func(Xt, y, linpred)
        iteration = 0
        while sum(gradient**2) / len(gradient) >= tol_grad and iteration < maxit:
            coef, gradient, _ = NewtonRaphson.iteration(y, X, Xt, coef, linpred, gradient_func, hessian_func,
                                                        adjusted_curvature=adjusted_curvature)
            linpred = X @ coef
            iteration += 1
            if iteration == maxit:
                print("Newton method has not converged hitting the maximum number of iterations: ", maxit)
        if adjusted_curvature:
            hessian = hessian_func(X, Xt, linpred, adjusted_curvature=adjusted_curvature, coef=coef, y=y)
        else:
            hessian = hessian_func(X, Xt, linpred)
        hessian_inv = numpy.linalg.inv(hessian)
        return coef, linpred, gradient, hessian_inv
