import json

from alasmc.main import ModelSelectionLA, ModelSelectionSMC, ModelKernel, normal_prior_log, beta_binomial_prior_log
from alasmc.glm import BinomialLogit, GLM
from alasmc.utilities import create_data
from alasmc.optimization import NewtonRaphson
from collections.abc import Callable

import numpy as np


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
                   kernel: Callable,
                   burnin: int,
                   particle_number: int,
                   n_draws: int,
                   dataset: int = None):
    results = []
    X, y, beta_true = create_data(n=n, n_covariates=n_covariates, n_active=n_active, rho=rho, model=glm)
    model_selection_LA = ModelSelectionLA(X=X,
                                          y=y,
                                          glm=glm,
                                          optimization_procedure=optimization_procedure,
                                          coef_init=coef_init,
                                          coef_prior_log=coef_prior_log,
                                          model_prior_log=model_prior_log)
    model_selection_LA.run()

    for feature_id in range(n_covariates):
        results.append({'dataset': dataset + 1,
                        'method': 'LA',
                        'n': n,
                        'p': n_covariates,
                        'p_true': n_active,
                        'rho': rho,
                        'model': 'Binomial Logit',
                        'feature': feature_id + 1,
                        'marginalPP': model_selection_LA.marginal_postProb[feature_id]})
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
                                                   verbose=False)
        model_selection_ALASMC.run()

        for feature_id in range(n_covariates):
            results.append({'dataset': dataset + 1,
                            'method': 'ALASMC',
                            'draw': draw,
                            'n': n,
                            'p': n_covariates,
                            'p_true': n_active,
                            'rho': rho,
                            'model': 'Binomial Logit',
                            'feature': feature_id + 1,
                            'marginalPP': model_selection_ALASMC.marginal_postProb[feature_id]})
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
                      n_datasets: int):
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
                                           dataset=dataset)
        print(f"Sampled datasets: [{dataset + 1} / {n_datasets}]")
    return results


if __name__ == "__main__":
    n_draws = 500
    n_datasets = 1
    n_covariates = 10
    n_active = 3
    n = 1000
    rho = 0.0
    coef_init = np.repeat(0, n_covariates)
    model_init = np.array([False] * n_covariates)
    kernel = ModelKernel()
    particle_number = 1000
    burnin = 5000

    res = multiple_datasets(n=n, n_covariates=n_covariates, n_active=n_active, rho=rho, glm=BinomialLogit(),
                            optimization_procedure=NewtonRaphson(), coef_init=coef_init, model_init=model_init,
                            coef_prior_log=normal_prior_log, model_prior_log=beta_binomial_prior_log,
                            kernel=kernel, burnin=burnin, particle_number=particle_number, n_draws=n_draws,
                            n_datasets=n_datasets)

    with open('results/single_dataset_results.json', 'w') as file:
        json.dump(res, file)
    print(f"The experiment is finished.")
    #X, y, beta_true = create_data(n=1000, n_covariates=10, n_active=3, rho=0., model=BinomialLogit())
    #coef_init = np.repeat(0, X.shape[1])
    #model_selection = ModelSelectionLA(X, y, glm=BinomialLogit(), optimization_procedure=NewtonRaphson(), coef_init=coef_init,
    #                                   coef_prior_log=normal_prior_log, model_prior_log=beta_binomial_prior_log)
    #model_selection.run()
    #print(model_selection.marginal_postProb)
