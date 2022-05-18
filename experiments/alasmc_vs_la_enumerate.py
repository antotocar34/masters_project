from alasmc.main import ModelSelectionSMC, ModelKernel, normal_prior_log, beta_binomial_prior_log, ApproxIntegral
from alasmc.glm import BinomialLogit, GLM
from alasmc.utilities import get_model_id, model_id_to_vector, create_model_matrix
from alasmc.optimization import NewtonRaphson
from collections.abc import Callable

import numpy as np


def create_data(n, n_covariates, n_active, rho):
    beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.ones(n_active)])
    n = n
    rho = rho
    sigma_x = np.diag([1.0] * n_covariates)
    sigma_x[np.triu_indices(n_covariates, 1)] = rho
    sigma_x[np.tril_indices(n_covariates, -1)] = rho
    X = np.random.multivariate_normal(np.zeros(n_covariates), sigma_x, n)
    p = 1 / (1 + np.exp(- X @ beta_true))
    y = np.random.binomial(1, p, n)
    return X, y, beta_true


class ModelSelectionLA:
    def __init__(self, X: np.ndarray,
                 y: np.ndarray,
                 glm: GLM,
                 optimization_procedure: callable,
                 coef_init: np.array,
                 coef_prior_log: callable,
                 model_prior_log: callable,
                 # kernel: callable,
                 # kernel_steps: int,
                 # burnin: int,
                 full_enumeration: bool = True,
                 tol_grad: float = 1e-8):
        self.X = X
        self.Xt = X.transpose()
        self.y = y
        self.glm = glm
        self.optimization_procedure = optimization_procedure
        self.coef_init = coef_init
        self.coef_prior_log = coef_prior_log
        self.model_prior_log = model_prior_log
        # self.kernel = kernel
        # self.kernel_steps = kernel_steps
        # self.burnin = burnin
        self.full_enumeration = full_enumeration
        self.tol_grad = tol_grad  # Let p_t = p_{t+1} once norm of gradient is smaller than this value.
        self.integrated_loglikes = None
        self.postProb = None
        self.marginal_postProb = None
        self.postMode = None
        self.model_matrix = None

    def _run_enumeration(self):
        p_full = self.X.shape[1]
        self.integrated_loglikes = np.repeat(np.nan, 2**p_full)
        self.postProb = np.repeat(np.nan, 2**p_full)
        self.marginal_postProb = np.repeat(np.nan, p_full)
        self.model_matrix = create_model_matrix(self.X)
        for model_id in range(2**p_full):
            model = self.model_matrix[model_id, :]
            coef_init = self.coef_init[model]
            self.integrated_loglikes[model_id] = ApproxIntegral.la_log(self.y, self.X[:, model], self.Xt[model, :],
                                                                       coef_init, self.glm.loglikelihood,
                                                                       self.coef_prior_log, self.glm.gradient,
                                                                       self.glm.hessian, self.tol_grad)
            self.postProb[model_id] = np.exp(self.integrated_loglikes[model_id] + self.model_prior_log(model))
        self.postProb = self.postProb / sum(self.postProb)
        self.marginal_postProb = self.postProb @ self.model_matrix
        self.postMode = model_id_to_vector(np.argmax(self.postProb), p_full)

    def run(self):
        if self.full_enumeration:
            self._run_enumeration()
        else:
            raise NotImplementedError("Only the full enumeration of the models is implemented at the moment")


if __name__ == "__main__":
    X, y, beta_true = create_data(n=1000, n_covariates=10, n_active=3, rho=0.)
    coef_init = np.repeat(0, X.shape[1])
    model_selection = ModelSelectionLA(X, y, glm=BinomialLogit, optimization_procedure=NewtonRaphson(), coef_init=coef_init,
                                       coef_prior_log=normal_prior_log, model_prior_log=beta_binomial_prior_log)
    model_selection.run()
    print(model_selection.marginal_postProb)
