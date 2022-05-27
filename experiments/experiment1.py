import json
import time

from alasmc.main import ModelSelectionLA, ModelSelectionSMC, normal_prior_log, beta_binomial_prior_log
from alasmc.kernels import SimpleGibbsKernel
from alasmc.glm import BinomialLogit, PoissonRegression, GLM
from alasmc.utilities import create_data, full_postProb, chi_squared_distance, euclidean_distance
from alasmc.optimization import NewtonRaphson
from collections.abc import Callable
from sklearn.linear_model import PoissonRegressor, LogisticRegression

import numpy as np

# import warnings
# warnings.filterwarnings("error")


def single_dataset(data_creation: Callable,
                   n: int,
                   rho: float,
                   glm: GLM,
                   optimization_procedure: object,
                   coef_prior_log: Callable,
                   model_prior_log: Callable,
                   kernel: object,
                   kernel_steps: int,
                   burnin: int,
                   particle_number: int,
                   smc_runs: int,
                   adjusted_curvature: bool,
                   adaptive_move: bool,
                   force_intercept: bool,
                   tol_grad: float,
                   tol_loglike: float,
                   dataset: int = None):
    X, y, beta_true = data_creation(n=n, rho=rho, glm=glm)
    true_model = (beta_true != 0.)
    n_covariates = len(beta_true)
    n_active = sum(true_model)
    model_init = np.repeat(False, n_covariates)
    if isinstance(glm, BinomialLogit):
        MLE_full_fit = LogisticRegression(fit_intercept=False, penalty='none').fit(X, y)
        coef_init = MLE_full_fit.coef_[0]
    else:
        MLE_full_fit = PoissonRegressor(fit_intercept=False, alpha=0.).fit(X, y)
        coef_init = MLE_full_fit.coef_
    results = []
    if isinstance(glm, BinomialLogit):
        model_name = 'Binomial Logit'
    elif isinstance(glm, PoissonRegression):
        model_name = 'Poisson'
    model_selection_LA = ModelSelectionLA(X=X,
                                          y=y,
                                          glm=glm,
                                          optimization_procedure=optimization_procedure,
                                          coef_init=coef_init,
                                          coef_prior_log=coef_prior_log,
                                          model_prior_log=model_prior_log,
                                          force_intercept=force_intercept,
                                          adjusted_curvature=adjusted_curvature,
                                          tol_grad=tol_grad)
    start = time.time()
    model_selection_LA.run()
    end = time.time()

    results.append({'dataset': dataset + 1,
                    'method': 'LA',
                    'n': n,
                    'p': n_covariates,
                    'p_true': n_active,
                    'rho': rho,
                    'beta_true': beta_true,
                    'coef_init': coef_init,
                    'model': model_name,
                    'force_intercept': force_intercept,
                    'marginalPProb': model_selection_LA.marginal_postProb,
                    'postProb': model_selection_LA.postProb,
                    'recovers_true': all(model_selection_LA.postMode[force_intercept:] == true_model[force_intercept:]),
                    'time': end - start})
    print("LA results are ready. Starting", smc_runs, "tries of ALASMC.")
    for smc_run in range(smc_runs):
        model_selection_ALASMC = ModelSelectionSMC(X=X,
                                                   y=y,
                                                   glm=glm,
                                                   optimization_procedure=optimization_procedure,
                                                   coef_init=coef_init,
                                                   model_init=model_init,
                                                   coef_prior_log=normal_prior_log,
                                                   model_prior_log=beta_binomial_prior_log,
                                                   kernel=kernel,
                                                   kernel_steps=kernel_steps,
                                                   adaptive_move=adaptive_move,
                                                   adjusted_curvature=adjusted_curvature,
                                                   burnin=burnin,
                                                   particle_number=particle_number,
                                                   force_intercept=force_intercept,
                                                   tol_grad=tol_grad,
                                                   tol_loglike=tol_loglike,
                                                   verbose=0)
        start = time.time()
        model_selection_ALASMC.run()
        end = time.time()
        postProb_full = full_postProb(model_selection_ALASMC.postProb, n_covariates, force_intercept)
        results.append({'dataset': dataset + 1,
                        'method': 'ALASMC',
                        'run': smc_run,
                        'n': n,
                        'p': n_covariates,
                        'p_true': n_active,
                        'rho': rho,
                        'beta_true': beta_true,
                        'coef_init': coef_init,
                        'model': model_name,
                        'particle_number': particle_number,
                        'force_intercept': force_intercept,
                        'adaptive_move': adaptive_move,
                        'adjusted_curvature': adjusted_curvature,
                        'kernel_steps': kernel_steps,
                        'burn_in': burnin,
                        'marginalPProb': model_selection_ALASMC.marginal_postProb,
                        'postProb': postProb_full,
                        'postProb_chi2dist_to_LA': chi_squared_distance(postProb_full, model_selection_LA.postProb),
                        'marginalPProb_euqldist_to_LA': euclidean_distance(model_selection_ALASMC.marginal_postProb[force_intercept:],
                                                                           model_selection_LA.marginal_postProb[force_intercept:]),
                        'recovers_LA': all(model_selection_ALASMC.postMode == model_selection_LA.postMode),
                        'recovers_true': all(model_selection_ALASMC.postMode[force_intercept:] ==
                                             true_model[force_intercept:]),
                        'time': end - start})
        if smc_run % 10 == 0:
            print(f"ALASMC done! [{smc_run + 1} / {smc_runs}]")
    return results


def multiple_datasets(data_creation: Callable,
                      n: int,
                      rho: float,
                      glm: GLM,
                      optimization_procedure: object,
                      coef_prior_log: Callable,
                      model_prior_log: Callable,
                      kernel: object,
                      burnin: int,
                      particle_number: int,
                      smc_runs: int,
                      n_datasets: int,
                      adjusted_curvature: bool,
                      adaptive_move: bool,
                      tol_grad: float,
                      tol_loglike: float,
                      force_intercept: bool):
    results = []
    for dataset in range(n_datasets):
        results = results + single_dataset(n=n,
                                           rho=rho,
                                           glm=glm,
                                           optimization_procedure=optimization_procedure,
                                           coef_prior_log=coef_prior_log,
                                           model_prior_log=model_prior_log,
                                           kernel=kernel,
                                           burnin=burnin,
                                           adaptive_move=adaptive_move,
                                           adjusted_curvature=adjusted_curvature,
                                           particle_number=particle_number,
                                           smc_runs=smc_runs,
                                           dataset=dataset,
                                           tol_grad=tol_grad,
                                           tol_loglike=tol_loglike,
                                           kernel_steps=1,
                                           data_creation=data_creation,
                                           force_intercept=force_intercept)
        print(f"Sampled datasets: [{dataset + 1} / {n_datasets}]")
    return results


def exp1_data_creation(glm: GLM, rho: float, n: int):
    if isinstance(glm, BinomialLogit):
        beta_true = np.append(np.zeros(7), [0.5, 1])
        X, y = create_data(n=n, rho=rho, model=glm, beta_true=beta_true, intercept=2)
        beta_true = np.append(2, beta_true)
    elif isinstance(glm, PoissonRegression):
        beta_true = np.array([0., 0., 0., 0.5, 1])
        X, y = create_data(n=n, rho=rho, model=glm, beta_true=beta_true)
        X = np.column_stack([X, X[:, 1:]**2])
        beta_true = np.concatenate([np.array([0.]), beta_true, np.zeros(5)])
    else:
        return None
    return X, y, beta_true


if __name__ == "__main__":
    tol_grad = 1e-13
    tol_loglike = 1e-10
    smc_runs = 50
    n_datasets = 30
    rho = 0.5
    n = 1000
    kernel = SimpleGibbsKernel
    particle_number = 5000
    burnin = 5000
    result = []
    for glm in [BinomialLogit(), PoissonRegression()]:
        adjusted_curvature = isinstance(glm, PoissonRegression)
        force_intercept = isinstance(glm, PoissonRegression)
        result = result + multiple_datasets(exp1_data_creation, n, rho, glm, NewtonRaphson(), normal_prior_log,
                                            beta_binomial_prior_log, SimpleGibbsKernel, burnin, particle_number,
                                            smc_runs, n_datasets, adjusted_curvature, False, tol_grad,
                                            tol_loglike, force_intercept)
    with open(f'results/experiment1.json', 'w') as file:
        json.dump(result, file)
    print(f"The Experiment 1 is finished.")

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

# if __name__ == "__main__":
#     tol_grad = 1e-13
#     tol_loglike = 1e-10
#     n_draws = 500
#     n_datasets = 1
#     n_covariates = 9
#     n_active = 2
#     n = 1000
#     # beta_true = np.array([0., 0., 0., 0.5, 1])
#     beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.repeat(1, n_active)])
#     X, y = create_data(n=n, rho=0.5, model=BinomialLogit, beta_true=beta_true, intercept=2)
#     # X = np.column_stack([X, X**2])
#     rho = 0.5
#     kernel = SimpleGibbsKernel
#     particle_number = 5000
#     full_fit = LogisticRegression(fit_intercept=True, penalty='l2', C=0.5).fit(X, y)
#     # full_fit = PoissonRegressor(fit_intercept=False, alpha=0.5, max_iter=1000).fit(X, y)
#     coef_init = np.append(full_fit.intercept_, full_fit.coef_) if full_fit.coef_.ndim == 1 else \
#         np.append(full_fit.intercept_, full_fit.coef_[0])
#     # alpha = 0.3 works well for Poisson, p = 100, n = 1000
#     model_init = np.repeat(False, n_covariates)
#
#     smc = ModelSelectionSMC(X, y,
#                             glm=BinomialLogit(),
#                             optimization_procedure=NewtonRaphson(),  # brackets are important
#                             coef_init=coef_init, model_init=model_init, coef_prior_log=normal_prior_log,
#                             model_prior_log=beta_binomial_prior_log, kernel=kernel, kernel_steps=1, burnin=5000,
#                             particle_number=particle_number, verbose=2, tol_grad=1e-13, tol_loglike=1e-10,
#                             adjusted_curvature=True, adaptive_move=False, force_intercept=False)
#
#     # model_selection_LA = ModelSelectionLA(X=X,
#     #                                       y=y,
#     #                                       glm=PoissonRegression(),
#     #                                       optimization_procedure=NewtonRaphson(),
#     #                                       coef_init=coef_init,
#     #                                       coef_prior_log=normal_prior_log,
#     #                                       model_prior_log=beta_binomial_prior_log,
#     #                                       tol_grad=1e-13)
#     #
#     smc.run()
#     # model_selection_LA.run()
#     print(smc.marginal_postProb, smc.postMode)
#     # print(model_selection_LA.marginal_postProb, model_selection_LA.postMode)
