import json
import time
import numpy as np

from alasmc.main import ModelSelectionLA, ModelSelectionSMC, normal_prior_log, beta_binomial_prior_log
from alasmc.kernels import SimpleGibbsKernel
from alasmc.glm import BinomialLogit, PoissonRegression, GLM
from alasmc.utilities import create_data, full_postProb, chi_squared_distance, euclidean_distance, has_intercept
from alasmc.optimization import NewtonRaphson
from collections.abc import Callable
from sklearn.linear_model import PoissonRegressor, LogisticRegression


def get_coef_init(coef_str: str, glm: GLM, X, y):
    if coef_str == "zero":
        if has_intercept(X):
            p = X.shape[1]
            return np.zeros(p)
        else:
            raise Exception("X should have a constant included")
    elif coef_str == "MLE":
        if isinstance(glm, BinomialLogit):
            MLE_full_fit = LogisticRegression(fit_intercept=False, penalty='none').fit(X, y)
            coef_init = MLE_full_fit.coef_[0]
        else:
            MLE_full_fit = PoissonRegressor(fit_intercept=False, alpha=0.).fit(X, y)
            coef_init = MLE_full_fit.coef_
        return coef_init
    else:
        raise NotImplementedError("Only two modes of work are now implemented for the coef_init.")


if __name__ == "__main__":
    results = []

    np.random.seed(100)

    tol_grad = 1e-5
    tol_loglike = 1e-5
    p_active = 3
    n_datasets = 50
    rho = 0.5
    kernel = SimpleGibbsKernel
    particle_number = 5000
    burnin = 5000
    adaptive_move = False
    kernel_steps = 1
    optimization_procedure = NewtonRaphson()
    force_intercept = False
    adjusted_curvature = True

    p_list = [10, 20, 30, 50]
    coefs_to_try = ["zero", "MLE"]
    glm_list = [BinomialLogit(), PoissonRegression()]
    n_list = [500, 1000, 2500, 5000]

    param_grid = [(1000, p, coef_str, glm) for p, coef_str, glm in
                  np.array(np.meshgrid(p_list, coefs_to_try, glm_list)).T.reshape(-1, 3)]
    param_grid = param_grid + [(n, 10, coef_str, glm) for n, coef_str, glm in
                               np.array(np.meshgrid(n_list, coefs_to_try, glm_list)).T.reshape(-1, 3)
                               if (n, 10, coef_str, glm) not in param_grid]

    n_cases = len(param_grid)
    case = 1

    for n, p, coef_str, glm in param_grid:
        case_time = 0
        for dataset in range(n_datasets):
            n = int(n)
            p = int(p)
            beta_true = np.append(np.zeros(p - p_active), np.repeat(0.5, p_active))
            X, y = create_data(n=n, rho=rho, model=glm, beta_true=beta_true[1:], intercept=beta_true[0])
            true_model = (beta_true != 0.)
            n_covariates = len(beta_true)
            n_active = int(sum(true_model))
            model_init = np.repeat(False, n_covariates)
            coef_init = get_coef_init(coef_str, glm, X, y)

            if isinstance(glm, BinomialLogit):
                model_name = 'Binomial Logit'
            elif isinstance(glm, PoissonRegression):
                model_name = 'Poisson'
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

            n_iterations = model_selection_ALASMC.iteration
            included_true = int(sum(model_selection_ALASMC.postMode[true_model]))
            discarded_spurious = int(sum(~model_selection_ALASMC.postMode[~true_model]))
            results.append({'dataset': dataset + 1,
                            'method': 'ALASMC',
                            'n': n,
                            'p': n_covariates,
                            'p_true': n_active,
                            'rho': rho,
                            'coef_init': coef_str,
                            'iterations': n_iterations,
                            'included_true': included_true,
                            'discarded_spurious': discarded_spurious,
                            'model': model_name,
                            'particle_number': particle_number,
                            'force_intercept': force_intercept,
                            'adaptive_move': adaptive_move,
                            'adjusted_curvature': adjusted_curvature,
                            'kernel_steps': kernel_steps,
                            'burn_in': burnin,
                            'mean_marginalPProb_active': np.mean(model_selection_ALASMC.marginal_postProb[true_model]),
                            'mean_marginalPProb_spurious': np.mean(model_selection_ALASMC.marginal_postProb[~true_model]),
                            'recovers_true': all(model_selection_ALASMC.postMode ==
                                                 true_model),
                            'time': end - start})
            print(f"Sampled datasets: [{dataset + 1} / {n_datasets}]")
            case_time += end - start
        print(f"Covered cases: [{case} / {n_cases}], elapsed time (sec.): {case_time}")
        case += 1

    with open(f'results/experiment2.json', 'w') as file:
        json.dump(results, file)
    print(f"The Experiment 2 is finished.")
