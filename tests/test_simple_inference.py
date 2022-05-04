import pytest

from collections import Counter

import numpy as np

from alasmc.main import ModelSelectionSMC, ModelKernel, normal_prior
from alasmc.GLM import LogisticGLM
from alasmc.utilities import get_model_id
from alasmc.optimization import newton_iteration


def create_data(n_covariates, n_active):
    n_covariates = 5
    n_active = 1
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

@pytest.fixture
def easy_data():
    n_covariates = 5
    n_active = 1
    X, y, beta_true = create_data(n_covariates,n_active)

    kernel = ModelKernel()
    particle_number = 1000
    model_init = np.array([False] * n_covariates)
    model_init[np.random.choice(n_covariates)] = True

    smc = ModelSelectionSMC(X, y, 
                            glm=LogisticGLM,
                            optimization_procedure=newton_iteration,
                            coef_init=np.array([0] * n_covariates), model_init=model_init, coef_prior=normal_prior,
                            kernel=kernel, kernel_steps=5, particle_number=particle_number, verbose=True)
    return smc, beta_true

# Simple test, 5 covariates, only one is significant.
# Tests the functionality of the whole package on a very simple case.
def test_simple_inference(easy_data):
    smc, beta_true = easy_data
    smc.run()
    sampled_models = Counter([get_model_id(model) for model in smc.particles])
    # print(get_model_id(beta_true != 0))
    # print(smc.particles)
    # print(sampled_models)
    true_model_id = get_model_id(beta_true != 0)
    selected_model_id = max(sampled_models, key=sampled_models.get)
    # print("Selected model: ", bin(selected_model_id)[2:])
    assert true_model_id == selected_model_id
