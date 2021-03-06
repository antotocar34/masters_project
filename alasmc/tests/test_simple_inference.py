import pytest
from collections import Counter


import numpy as np
from sklearn.linear_model import LogisticRegression
# from scipy.spatial.distance import hamming

from termcolor import colored

from alasmc.main import ModelSelectionSMC, normal_prior_log, beta_binomial_prior_log
from alasmc.kernels import SimpleGibbsKernel, LogisticKernel
from alasmc.glm import BinomialLogit
from alasmc.utilities import get_model_id, model_id_to_vector
from alasmc.optimization import NewtonRaphson


def create_data(n_covariates, n_active):
    beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.ones(n_active)])
    n = 1000
    rho = 0.0
    sigma_x = np.diag([1.0] * n_covariates)
    sigma_x[np.triu_indices(n_covariates, 1)] = rho
    sigma_x[np.tril_indices(n_covariates, -1)] = rho
    X = np.random.multivariate_normal(np.zeros(n_covariates), sigma_x, n)
    p = 1 / (1 + np.exp(- X @ beta_true))
    y = np.random.binomial(1, p, n)
    return (X, y, beta_true)

def easy_data(adaptive, kernel):
    n_covariates = 50
    n_active = 3
    X, y, beta_true = create_data(n_covariates,n_active)

    initial_kernel = SimpleGibbsKernel()
    particle_number = 1000
    model_init = np.array([False] * n_covariates)
    model_init[np.random.choice(n_covariates)] = True

    smc = ModelSelectionSMC(X, y, 
                            glm=BinomialLogit,
                            optimization_procedure=NewtonRaphson(),
                            # coef_init=np.array([0] * n_covariates), 
                            coef_init=LogisticRegression(fit_intercept=False, penalty="none").fit(X,y).coef_[0],
                            model_init=model_init, 
                            coef_prior_log=normal_prior_log,
                            model_prior_log=beta_binomial_prior_log,
                            burnin=1000,
                            kernel=kernel, 
                            initial_kernel=initial_kernel,
                            adaptive_move=adaptive,
                            kernel_steps=1, 
                            particle_number=particle_number, 
                            verbose=True)
    return smc, beta_true, n_covariates


# Simple test, 5 covariates, only one is significant.
# Tests the functionality of the whole package on a very simple case.
@pytest.mark.parametrize(('adaptive', 'kernel'), 
    [
        # (True, SimpleGibbsKernel()), (False, SimpleGibbsKernel()),
        (True, LogisticKernel()), (False, LogisticKernel())
    ]
    )
def test_simple_inference(capsys, adaptive, kernel):
    smc, beta_true, n_covariates = easy_data(adaptive, kernel)
    with capsys.disabled():
        smc.run()
    sampled_models = Counter([get_model_id(model) for model in smc.particles])
    true_model_id = get_model_id(beta_true != 0)
    selected_model_id = max(sampled_models, key=sampled_models.get)

    true_gamma = model_id_to_vector(true_model_id, n_covariates)
    selected_gamma = model_id_to_vector(selected_model_id, n_covariates)

    equal = true_model_id == selected_model_id
    subset = all([ 
        selected_gamma[i] == b 
        for (i, b) in enumerate(true_gamma) 
        if b 
        ])
    with capsys.disabled():
        if subset and not equal:
            differences = np.sum(np.abs(true_gamma - selected_gamma))
            print(colored("Number of spurious variables in most probable model: ", "green") + colored(differences, "yellow"))
        if equal:
            print(colored("Perfect match!", "green"))

    assert equal or subset

