"""
Students: Antoine Carnec, Maxim Fedotov

Description TODO
"""
import math

import numpy as np
from scipy.stats import multinomial
from scipy.optimize import root_scalar
from scipy.special import softmax, comb
from copy import deepcopy
from collections import Counter

from .smc import SMC
from .glm import GLM, BinomialLogit, PoissonRegression
from .utilities import get_model_id, unzip, model_id_to_vector, create_model_matrix, create_data
from .optimization import NewtonRaphson


class ApproxIntegral:
    @staticmethod
    def ala_log(y: np.ndarray, linpred_old: np.ndarray, coef_new: np.ndarray, gradient: np.ndarray,
                hessian_inv: np.ndarray, loglikelihood: callable, coef_prior_log: callable):
        p = len(coef_new)
        return loglikelihood(y, linpred_old) + coef_prior_log(coef_new) + p / 2 * np.log(2 * np.pi) + \
               0.5 * np.log(np.linalg.det(hessian_inv)) + 0.5 * gradient.transpose() @ hessian_inv @ gradient

    @staticmethod
    def la_log(y: np.ndarray, X: np.ndarray, Xt: np.ndarray, coef_init: np.ndarray, loglikelihood: callable,
               coef_prior_log: callable, gradient_func: callable, hessian_func: callable, tol_grad: float):
        p = len(coef_init)
        if p == 0:
            return loglikelihood(y, X @ coef_init)
        coef, linpred, _, hessian_inv = NewtonRaphson.optimize(y, X, Xt, coef_init, gradient_func, hessian_func, tol_grad)
        return loglikelihood(y, linpred) + coef_prior_log(coef) + (p / 2) * np.log(2 * np.pi) + 0.5 * \
               np.log(np.linalg.det(hessian_inv))


# Here we set default g = 1. In fact, we might also think about adding rho parameter.
def normal_prior_log(beta: np.ndarray, g=1):
    p = len(beta)
    return -1 / 2 * (p * np.log(2 * np.pi * g) + 1 / g * beta.dot(beta))


def beta_binomial_prior_log(model: np.ndarray):
    p_model = sum(model)
    p = len(model)
    return - np.log(p + 1) - np.log(comb(p, p_model))


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
                             sum_{s=0}^{t} log( sum_{n=1}^{N} w_s^n ).
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 glm: GLM,
                 optimization_procedure: callable,
                 coef_init: np.array,
                 model_init: np.array,
                 coef_prior_log: callable,
                 model_prior_log: callable,
                 kernel: callable,
                 kernel_steps: int,
                 burnin: int,
                 particle_number: int,
                 tol_loglike: float = 1e-8,
                 maxit_smc: int = 40,
                 ess_min_ratio: float = 0.5,
                 tol_grad: float = 1e-13,
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
        self.particle_ids = np.zeros(particle_number)
        self.burnin = burnin
        self.tol_grad = tol_grad  # Let p_t = p_{t+1} once norm of gradient is smaller than this value.
        self.coef_prior_log = coef_prior_log
        self.model_prior_log = model_prior_log
        self.optimization_procedure = optimization_procedure
        self.tol_loglike = tol_loglike
        self.coefs = {}  # Used to save computed coefficients for models. 
                         # TODO add type annotation to make clear what this dictionary is
        self.integrated_loglikes = {}  # Used to save integrated likelihoods for models.
        self.integrated_loglike_changes = np.repeat(np.nan, particle_number)
        self.computed_at = {}  # Used to save the number of the latest iteration where the coefficients and LL were upd.
        self.postProb = {}
        self.marginal_postProb = None
        self.postMode = None

    def compute_integrated_loglike(self, model: np.ndarray, model_id: int):
        converged = False
        computed_at = self.computed_at.get(model_id, None)
        if computed_at == -1:  # If it's converged, just return the latest log-likelihood
            if self.integrated_loglikes[model_id][0] != self.integrated_loglikes[model_id][1]:
                self.integrated_loglikes[model_id][0] = self.integrated_loglikes[model_id][1]
            return self.integrated_loglikes[model_id][1]
        elif computed_at:
            n_iterations = self.iteration - self.computed_at[model_id]
            coef_new = self.coefs[model_id][1]
        else:
            n_iterations = self.iteration
            coef_new = self.coef_init[model]
            self.coefs[model_id] = [None, None]
            self.integrated_loglikes[model_id] = [None, None]
        integrated_loglike = self.integrated_loglikes[model_id][1]

        for iter in range(n_iterations):
            coef_old = coef_new
            linpred_old = self.X[:, model] @ coef_old
            if isinstance(self.optimization_procedure, NewtonRaphson):
                coef_new, gradient, hessian_inv = self.optimization_procedure.iteration(self.y,
                                                                                        self.X[:, model],
                                                                                        self.Xt[model, :],
                                                                                        coef_old,
                                                                                        linpred_old,
                                                                                        self.glm.gradient,
                                                                                        self.glm.hessian)
            else:
                raise NotImplementedError("Only Newton method is implemented at the moment.")
            if n_iterations - iter <= 2:
                self.coefs[model_id][0] = self.coefs[model_id][1]
                self.coefs[model_id][1] = coef_new
                integrated_loglike = ApproxIntegral.ala_log(self.y, linpred_old, coef_new,
                                                            gradient, hessian_inv,
                                                            self.glm.loglikelihood, self.coef_prior_log)
                self.integrated_loglikes[model_id][0] = self.integrated_loglikes[model_id][1]
                self.integrated_loglikes[model_id][1] = integrated_loglike

        if not model.any():
            converged = True
        elif n_iterations != 0:
            converged = sum(gradient**2) / len(gradient) < self.tol_grad  # Check for convergence of optimization problem.
        self.computed_at[model_id] = self.iteration if not converged else -1
        return integrated_loglike

    # Does this need to be part of the class? ANSWER: It does not. We'll create another script with kernels.
    def gibbs_iteration(self, model: np.ndarray):
        model_new = model
        model_id_new = get_model_id(model_new)
        for _ in range(self.kernel_steps):
            model_old = model_new
            model_id_old = model_id_new
            model_new = self.kernel.sample(model_old)  # Draw a new sample from kernel
            model_id_new = get_model_id(model_new)
            postLLRatio = self.compute_integrated_loglike(model_old, model_id_old) + self.model_prior_log(model_old) - \
                self.compute_integrated_loglike(model_new, model_id_new) - self.model_prior_log(model_new)
            uniform = np.random.uniform(0, 1)
            if uniform <= 1e-200:  # For the sake of dealing with overflow.
                accept = False
            else:
                accept = np.log(1 / uniform - 1) >= postLLRatio
            model_new = model_new if accept else model_old
            model_id_new = model_id_new if accept else model_id_old
        return model_id_new, model_new

    def sample_init(self):
        """
        Sample Initial Particles
        """
        model_new = self.model_init  # Todo abstract away
        model_new_id = get_model_id(model_new)
        for i in range(self.particle_number + self.burnin):
            model_old_id, model_old = model_new_id, model_new
            model_new_id, model_new = self.gibbs_iteration(model_old)
            if (j := i - self.burnin) >= 0:
                self.particle_ids[j] = model_new_id
                self.particles[j] = model_new

    def move(self, ancestor_idxs: np.array):
        particle_ids, particles = unzip([self.gibbs_iteration(self.particles[ancestor_idx])
                                         for ancestor_idx in ancestor_idxs])
        self.particle_ids, self.particles = np.array(particle_ids), np.array(particles)

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
        def change(integrated_loglikes):
            return integrated_loglikes[1] - integrated_loglikes[0]
        self.integrated_loglike_changes = np.array([change(self.integrated_loglikes[get_model_id(model)])
                                                    for model in self.particles])
        self.w_log = self.w_hat_log + self.integrated_loglike_changes
        self.w_normalized = softmax(self.w_log)

    def compute_postProb(self):
        postProb = {}
        for id in np.unique(self.particle_ids):
            postProb[id] = sum(self.w_normalized[self.particle_ids == id])
        return postProb

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
        while not np.allclose(self.integrated_loglike_changes, 0, atol=self.tol_loglike) and self.iteration < self.maxit_smc:
            self.iteration += 1
            if self.ess() < self.ess_min:
                ancestor_idxs = self.resample()  # Get indexes of ancestors
                self.w_hat_log = np.zeros(self.particle_number)
            else:
                ancestor_idxs = np.arange(self.particle_number)
                self.w_hat_log = self.w_log
            self.move(ancestor_idxs)
            self.update_weights()  # Recalculate weights
            if self.verbose:
                print(f"Iteration {self.iteration} done!")
        if self.iteration == self.maxit_smc:
            print("SMC hits the pre-specified maximum of iterations.")
        self.postProb = self.compute_postProb()
        self.marginal_postProb = self.w_normalized @ self.particles
        self.postMode = model_id_to_vector(max(self.postProb, key=self.postProb.get), len(self.marginal_postProb))
        if self.verbose:
            print('---SMC finished---\n')


class ModelSelectionLA:
    def __init__(self, X: np.ndarray,
                 y: np.ndarray,
                 glm: GLM,
                 optimization_procedure: object,
                 coef_init: np.array,
                 coef_prior_log: callable,
                 model_prior_log: callable,
                 # kernel: callable,
                 # kernel_steps: int,
                 # burnin: int,
                 full_enumeration: bool = True,
                 tol_grad: float = 1e-13):
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
            self.postProb[model_id] = self.integrated_loglikes[model_id] + self.model_prior_log(model)
        self.postProb = softmax(self.postProb)
        self.marginal_postProb = self.postProb @ self.model_matrix
        self.postMode = model_id_to_vector(np.argmax(self.postProb), p_full)

    def run(self):
        if self.full_enumeration:
            self._run_enumeration()
        else:
            raise NotImplementedError("Only the full enumeration of the models is implemented at the moment")


if __name__ == '__main__':
    n_covariates = 15
    n_active = 3
    n = 1000
    rho = 0.5
    X, y, beta_true = create_data(n, n_covariates, n_active, rho, BinomialLogit)
    kernel = ModelKernel()
    particle_number = 5000
    model_init = np.array([False] * n_covariates)

    smc = ModelSelectionSMC(X, y,
                            glm=BinomialLogit,
                            optimization_procedure=NewtonRaphson(),  # brackets are important
                            coef_init=np.array([0] * n_covariates), model_init=model_init, coef_prior_log=normal_prior_log,
                            model_prior_log=beta_binomial_prior_log, kernel=kernel, kernel_steps=1, burnin=5000,
                            particle_number=particle_number, verbose=True, tol_grad=1e-13, tol_loglike=1e-13)
    smc.run()
    print(smc.marginal_postProb, smc.postMode)
    # sampled_models = Counter([get_model_id(model) for model in smc.particles])
    # print(get_model_id(beta_true != 0))
    # print(smc.particles)
    # print(sampled_models)
    # selected_model_id = max(sampled_models, key=sampled_models.get)
    # print("Selected model: ", bin(selected_model_id)[2:])
