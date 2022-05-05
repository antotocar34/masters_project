"""
Students: Antoine Carnec, Maxim Fedotov

Description TODO
"""
import math

import numpy as np
from scipy.stats import multinomial
from scipy.optimize import root_scalar
from scipy.special import softmax
from copy import deepcopy
from collections import Counter

from smc import SMC
from GLM import GLM, BinomialLogit
from utilities import get_model_id
from optimization import newton_iteration


class ApproxIntegral:
    @staticmethod
    # version for GLMs
    def ala_log(y, linpred_old, coef_new, gradient, hessian_inv, loglikelihood, coef_prior):
        p = len(coef_new)
        return loglikelihood(y, linpred_old) + np.log(coef_prior(coef_new, p)) + (p / 2) * np.log(2 * np.pi) + \
               0.5 * np.log(np.linalg.det(hessian_inv)) + 0.5 * gradient.transpose() @ hessian_inv @ gradient


# Here we set default g = 1. In fact, we might also think about adding rho parameter
def normal_prior(beta: np.ndarray, p: int, g=1):
    return 1 / (2 * np.pi * g)**(p / 2) * np.exp(- 1 / g * beta.dot(beta) / 2)


class ModelKernel:
    """
    TODO

    Methods:
        sample: Samples new matrix given the current one by interchanging a pair of elements.    @static
    """
    def __init__(self):
        pass

    @staticmethod
    def sample(model_cur: np.ndarray) -> np.ndarray:
        """
        ...

        Parameters:
            model_cur:  Current particle to provide a new particle in re-sampling procedure.        [numpy.ndarray]

        Returns:
            New particle obtained by sampling form the Markov kernel given the current particle.    [numpy.ndarray]
        """
        p = len(model_cur)
        model_new = deepcopy(model_cur)
        i = np.random.choice(p)
        model_new[i] = not model_cur[i]
        return model_new


class ModelSelectionSMC(SMC):
    """
        ...

        Attributes:
            kernel:          An object with method 'sample'; Markov kernel that draws a new    [callable]
                             sample given the current sample (particle).
            kernel_steps:    Number of times the kernel is applied to a particle; defines a     [int]
                             'depth' of MCMC resampling.
            particle_number: Size of the sample                                                 [int]
            ess_min_ratio:   Ratio that defines the min Effective Sample Size that the          [float]
                             algorithm maintains at each step.
            verbose:         If True, the methods will print information about the process.     [bool]
            ess_min:         Minimal ESS defined by ess_min_ratio and particle_number.          [float]
            iteration:       Tracks number of iteration.                                        [int]
            w_log:           Unnormalized logarithmic weights. To calculate normalized ones.    [numpy.ndarray]
            w_normalized:    Normalized weights at time t. For t > 0 are used to sample         [numpy.ndarray]
                             ancestors at step t+1 from Multinomial(w_normalized).
            _lambda:         Tempering parameter 'lambda' at iteration t; it defines            [float]
                             the sequence of distributions.
            delta:           Defines lambda update, i.e. lambda_t - lambda_(t-1); is chosen     [float]
                             such that the updated sample maintains ESS_min.
            logLt:           Logarithm of the estimated normalized constant, basically:         [float]
                             \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n ).
    """
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 glm: GLM,
                 optimization_procedure: callable,
                 coef_init: np.array, 
                 model_init: np.array,
                 coef_prior: callable, 
                 kernel: callable,
                 kernel_steps: int,
                 burnin: int,
                 particle_number: int,
                 tol_loglike: float = 10e-8,
                 maxit_smc: int = 40,
                 ess_min_ratio: float = 1/2, 
                 verbose: bool = False) -> None:

        super().__init__(
            kernel=kernel,
            kernel_steps=kernel_steps,
            particle_number=particle_number,
            maxit_smc=maxit_smc,
            ess_min_ratio=ess_min_ratio,
            verbose=verbose
        )

        self.X = X
        self.Xt = X.transpose()
        self.y = y
        self.glm = glm
        self.coef_init = coef_init
        self.model_init = model_init
        self.burnin = burnin

        self.coef_prior = coef_prior  # This is the distribution that you start with.
        self.optimization_procedure = optimization_procedure
        self.tol_loglike = tol_loglike


        self.coefs = {}  # Used to save computed coefficients for models. 
                         # TODO add type annotation to make clear what this dictionary is
        self.integrated_loglikes = {}  # Used to save integrated likelihoods for models.
        self.integrated_loglike_changes = np.zeros(particle_number)
        self.computed_at = {}  # Used to save the number of the latest iteration where the coefficients and LL were upd.

    def compute_integrated_loglike(self, model: np.ndarray):
        model_id = get_model_id(model)
        model_seen = True if self.computed_at.get(model_id, None) else False
        if model_seen:
            n_iterations = self.iteration - self.computed_at[model_id]
            coef_new = self.coefs[model_id][1]
        else:
            n_iterations = self.iteration
            coef_new = self.coef_init[model]
            self.coefs[model_id] = [None, None] # TODO make this a mutable fixed size array?
            self.integrated_loglikes[model_id] = [None, None]
        integrated_loglike = self.integrated_loglikes[model_id][1]
        for iter in range(n_iterations): # TODO abstract this into a function
            coef_old = coef_new
            linpred_old = self.X[:, model] @ coef_old
            gradient = self.glm.gradient(self.Xt[model, :], self.y, linpred_old)
            hessian = self.glm.hessian(self.X[:, model], self.Xt[model, :], linpred_old)
            hessian_inv = np.linalg.inv(hessian)
            coef_new = self.optimization_procedure(coef_old, gradient, hessian_inv)
            if n_iterations - iter <= 2:
                integrated_loglike = ApproxIntegral.ala_log(self.y, linpred_old, coef_new,
                                                            gradient, hessian_inv,
                                                            self.glm.loglikelihood, self.coef_prior)
                self.coefs[model_id][0] = self.coefs[model_id][1]
                self.coefs[model_id][1] = coef_new
                self.integrated_loglikes[model_id][0] = self.integrated_loglikes[model_id][1]
                self.integrated_loglikes[model_id][1] = integrated_loglike
        self.computed_at[model_id] = self.iteration
        return integrated_loglike

    # Does this need to be part of the class?
    def gibbs_iteration(self, model: np.ndarray):
        model_new = model
        for _ in range(self.kernel_steps):
            model_old = model_new
            model_new = self.kernel.sample(model_old)  # Draw a new sample from kernel
            integrated_loglike_model_old = self.compute_integrated_loglike(model_old)
            integrated_loglike_model_new = self.compute_integrated_loglike(model_new)
            accept_prob = 1 / (1 + np.exp(integrated_loglike_model_old - integrated_loglike_model_new))
            accept = np.random.binomial(1, accept_prob)
            model_new = model_new if accept else model_old
        return model_new

    def sample_init(self):
        """
        Sample Inital Particles
        """
        model_new = self.model_init  # Todo abstract away
        for i in range(self.particle_number + self.burnin):
            model_old = model_new
            model_new = self.gibbs_iteration(model_old)
            if (j := i - self.burnin) >= 0:
                self.particles[j] = model_new

    def update_weights(self) -> None:
        """
        Weight update according to a new delta found by 'update_lambda'.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates attributes 'w' and 'w_normalized'.
        """
        def diff(integrated_loglikes):
            return integrated_loglikes[1] - integrated_loglikes[0]
        self.integrated_loglike_changes[:] = np.array([diff(self.integrated_loglikes[get_model_id(model)])
                                                       for model in self.particles])
        self.w_log = self.w_hat_log + self.integrated_loglike_changes
        self.w_normalized = softmax(self.w_log)

    def run(self):
        """
        Runs the Adaptive SMC algorithm. See Algorithm 2 in the report.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates all attributes. The logarithm of the estimate of the final normalising constant is kept in 'logLt'.
        """
        if self.verbose:
            print('---SMC started---')
        self.sample_init()
        self.w_log = np.zeros(self.particle_number)
        self.w_normalized = np.repeat(1 / self.particle_number, self.particle_number)
        if self.verbose:
            print('Iteration 1 done! The initial particles sampled.')
        while np.isclose(self.integrated_loglike_changes, 0, atol=self.tol_loglike).all() and self.iteration < self.maxit_smc:
            self.iteration += 1
            if self.ess() < self.ess_min:
                ancestor_idxs = self.resample()  # Get indexes of ancestors
                self.w_hat_log = np.zeros(self.particle_number)
            else:
                ancestor_idxs = np.arange(self.particle_number)
                self.w_hat_log = self.w_log
            self.particles = [self.gibbs_iteration(self.particles[ancestor_idx]) for ancestor_idx in ancestor_idxs]  # Update particles
            self.update_weights()  # Recalculate weights
            if self.verbose:
                print(f"Iteration {self.iteration} done!")
        if self.verbose:
            print('---SMC finished---\n')


if __name__ == '__main__':
    n_covariates = 1000
    n_active = 3
    beta_true = np.concatenate([np.zeros(n_covariates - n_active), np.ones(n_active)])
    n = 1000
    rho = 0.0
    sigma_x = np.diag([1.0] * n_covariates)
    sigma_x[np.triu_indices(n_covariates, 1)] = rho
    sigma_x[np.tril_indices(n_covariates, -1)] = rho
    X = np.random.multivariate_normal(np.zeros(n_covariates), sigma_x, n)
    p = 1 / (1 + np.exp(- X @ beta_true))
    y = np.random.binomial(1, p, n)
    kernel = ModelKernel()
    particle_number = 1000
    model_init = np.array([False] * n_covariates)
    smc = ModelSelectionSMC(X, y,
                            glm=BinomialLogit,
                            optimization_procedure=newton_iteration,
                            coef_init=np.array([0] * n_covariates), model_init=model_init, coef_prior=normal_prior,
                            kernel=kernel, kernel_steps=1, burnin=5000, particle_number=particle_number, verbose=True)
    smc.run()
    sampled_models = Counter([get_model_id(model) for model in smc.particles])
    print(get_model_id(beta_true != 0))
    # print(smc.particles)
    print(sampled_models)
    selected_model_id = max(sampled_models, key=sampled_models.get)
    print("Selected model: ", bin(selected_model_id)[2:])
