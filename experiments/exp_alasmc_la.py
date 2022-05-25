import json
import time

from alasmc.main import ModelSelectionLA, ModelSelectionSMC, ModelKernel, normal_prior_log, beta_binomial_prior_log
from alasmc.glm import BinomialLogit, PoissonRegression, GLM
from alasmc.utilities import create_data
from alasmc.optimization import NewtonRaphson
from collections.abc import Callable
from sklearn.linear_model import PoissonRegressor, LogisticRegression

import numpy as np

import warnings
warnings.filterwarnings("error")


def single_dataset(n: int,
                   n_covariates: int,
                   n_active: int,
                   rho: float,
                   glm: GLM,
                   optimization_procedure: object,
                   coef_init: np.ndarray,
                   model_init: np.ndarray,
                   coef_prior_log: Callable,
                   model_prior_log: Callable,
                   kernel: object,
                   burnin: int,
                   particle_number: int,
                   n_draws: int,
                   tol_grad: float,
                   tol_loglike: float,
                   dataset: int = None):
    results = []
    beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.ones(n_active)])
    X, y = create_data(n=n, n_covariates=n_covariates, n_active=n_active, rho=rho, model=glm, beta_true=beta_true)
    model_selection_LA = ModelSelectionLA(X=X,
                                          y=y,
                                          glm=glm,
                                          optimization_procedure=optimization_procedure,
                                          coef_init=coef_init,
                                          coef_prior_log=coef_prior_log,
                                          model_prior_log=model_prior_log,
                                          tol_grad=tol_grad)
    start = time.time()
    model_selection_LA.run()
    end = time.time()

    for feature_id in range(n_covariates):
        results.append({'dataset': dataset + 1,
                        'method': 'LA',
                        'n': n,
                        'p': n_covariates,
                        'p_true': n_active,
                        'rho': rho,
                        'model': 'Poisson Regression',
                        'feature': feature_id + 1,
                        'marginalPP': model_selection_LA.marginal_postProb[feature_id],
                        'time': end - start})
    print("LA results are ready. Starting", n_draws, "tries of ALASMC.")
    for draw in range(n_draws):
        model_selection_ALASMC = ModelSelectionSMC(X=X,
                                                   y=y,
                                                   glm=glm,
                                                   optimization_procedure=optimization_procedure,  # brackets are important
                                                   coef_init=coef_init,
                                                   model_init=model_init,
                                                   coef_prior_log=normal_prior_log,
                                                   model_prior_log=beta_binomial_prior_log,
                                                   kernel=kernel,
                                                   kernel_steps=1,
                                                   burnin=burnin,
                                                   particle_number=particle_number,
                                                   tol_grad=tol_grad,
                                                   tol_loglike=tol_loglike,
                                                   verbose=False)
        start = time.time()
        model_selection_ALASMC.run()
        end = time.time()
        for feature_id in range(n_covariates):
            results.append({'dataset': dataset + 1,
                            'method': 'ALASMC',
                            'draw': draw,
                            'n': n,
                            'p': n_covariates,
                            'p_true': n_active,
                            'rho': rho,
                            'model': 'Poisson Regression',
                            'particle_number': particle_number,
                            'burn_in': burnin,
                            'feature': feature_id + 1,
                            'marginalPP': model_selection_ALASMC.marginal_postProb[feature_id],
                            'time': end - start})
        print(f"ALASMC done! [{draw + 1} / {n_draws}]")
    return results


def multiple_datasets(n: int,
                      n_covariates: int,
                      n_active: int,
                      rho: float,
                      glm: GLM,
                      optimization_procedure: object,
                      coef_init: np.ndarray,
                      model_init: np.ndarray,
                      coef_prior_log: Callable,
                      model_prior_log: Callable,
                      kernel: object,
                      burnin: int,
                      particle_number: int,
                      n_draws: int,
                      n_datasets: int,
                      tol_grad: float,
                      tol_loglike: float):
    results = []
    for dataset in range(n_datasets):
        results = results + single_dataset(n=n,
                                           n_covariates=n_covariates,
                                           n_active=n_active,
                                           rho=rho,
                                           glm=glm,
                                           optimization_procedure=optimization_procedure,
                                           coef_init=coef_init,
                                           model_init=model_init,
                                           coef_prior_log=coef_prior_log,
                                           model_prior_log=model_prior_log,
                                           kernel=kernel,
                                           burnin=burnin,
                                           particle_number=particle_number,
                                           n_draws=n_draws,
                                           dataset=dataset,
                                           tol_grad=tol_grad,
                                           tol_loglike=tol_loglike)
        print(f"Sampled datasets: [{dataset + 1} / {n_datasets}]")
    return results


#if __name__ == "__main__":
#    n_draws = 500
#    n_datasets = 1
#    n_covariates_list = [10, 15]
#    n_active = 3
#    n_list = [500, 1000, 2000, 4000]
#    rho_list = [0.0, 0.5]
#    coef_init_large = np.repeat(0, n_covariates_list[-1])
#    model_init_large = np.array([False] * n_covariates_list[-1])
#    kernel = ModelKernel()
#    tol_grad = 1e-10
#    tol_loglike = 1e-10
#
#    param_grid = np.array(np.meshgrid(n_covariates_list, n_list, rho_list)).T.reshape(-1, 3)
#    n_experiments = len(param_grid)
#    experiment = 1
#
#    for n_covariates, n, rho in param_grid:
#        n_covariates = int(n_covariates)
#        burnin = 1000 if n_covariates <= 10 else 5000
#        particle_number = 500 if n_covariates <= 10 else 1000
#        n = int(n)
#        particle_number = int(particle_number)
#        print(f"Starting experiment [{experiment} / {n_experiments}]")
#        model_init = model_init_large[:n_covariates]
#        coef_init = coef_init_large[:n_covariates]
#        res = multiple_datasets(n=n, n_covariates=n_covariates, n_active=n_active, rho=rho, glm=BinomialLogit,
#                                optimization_procedure=NewtonRaphson(), coef_init=coef_init, model_init=model_init,
#                                coef_prior_log=normal_prior_log, model_prior_log=beta_binomial_prior_log,
#                                kernel=kernel, burnin=burnin, particle_number=particle_number, n_draws=n_draws,
#                                n_datasets=n_datasets, tol_grad=tol_grad, tol_loglike=tol_loglike)
#
#        with open(f'results/single_PoissonRegression_{n_covariates}covariates_{n}n_{rho}rho_{particle_number}particles_{tol_loglike}loglike_{tol_grad}grad_results.json', 'w') as file:
#            json.dump(res, file)
#        print(f"The experiment [{experiment} / {n_experiments}] is finished.")
#        experiment += 1
#    #X, y, beta_true = create_data(n=1000, n_covariates=10, n_active=3, rho=0., model=BinomialLogit())
#    #coef_init = np.repeat(0, X.shape[1])
#    #model_selection = ModelSelectionLA(X, y, glm=BinomialLogit(), optimization_procedure=NewtonRaphson(), coef_init=coef_init,
#    #                                   coef_prior_log=normal_prior_log, model_prior_log=beta_binomial_prior_log)
#    #model_selection.run()
#    #print(model_selection.marginal_postProb)
#

if __name__ == "__main__":
    n_covariates = 15
    n_active = 3
    n = 10000
    rho = 0.
    beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.ones(n_active)])
    X, y = create_data(n, n_covariates, n_active, rho, PoissonRegression, beta_true)
    kernel = ModelKernel()
    particle_number = 5000
    coef_init = np.zeros(n_covariates)  # PoissonRegressor(fit_intercept=False, tol=1e-10, max_iter=1000).fit(X, y).coef_
    model_init = np.repeat(False, n_covariates)

    smc = ModelSelectionSMC(X, y,
                            glm=PoissonRegression,
                            optimization_procedure=NewtonRaphson(),  # brackets are important
                            coef_init=coef_init, model_init=model_init, coef_prior_log=normal_prior_log,
                            model_prior_log=beta_binomial_prior_log, kernel=kernel, kernel_steps=1, burnin=5000,
                            particle_number=particle_number, verbose=2, tol_grad=1e-10, tol_loglike=1e-10,
                            adjusted_curvature=True)

    # model_selection_LA = ModelSelectionLA(X=X,
    #                                       y=y,
    #                                       glm=PoissonRegression,
    #                                       optimization_procedure=optimization_procedure,
    #                                       coef_init=coef_init,
    #                                       coef_prior_log=coef_prior_log,
    #                                       model_prior_log=model_prior_log,
    #                                       tol_grad=tol_grad)

    smc.run()
    print(smc.marginal_postProb, smc.postMode)
