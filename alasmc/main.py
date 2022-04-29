"""
Students: Antoine Carnec, Maxim Fedotov

This is an implementation of the latin square sampler described in
``
Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." arXiv preprint arXiv:2011.02328 (2020)
``

We first implement a generic Adaptive SMC Sampler Class.
This class samples from a posterior `p` of the form
`p(x) \propto m(x) exp(-V(x))`, where m is the prior and exp(-V) is the likelihood
"""
import math

import numpy
from scipy.stats import multinomial
from scipy.optimize import root_scalar
from scipy.special import softmax
from copy import deepcopy
from collections import Counter


def get_model_id(model: numpy.ndarray):
    return int('0b' + ''.join((model * 1).astype(str)), 2)


def binomial_logit_logll(y, linpred):
    p = 1 / (1 + numpy.exp(-linpred))
    if numpy.isclose(p, 0).any() or numpy.isclose(p, 1).any():
        raise Exception("Exact 0 and 1 occur in the computed probabilities.")
    return numpy.sum(y * numpy.log(p) + (1 - y) * numpy.log(1 - p))


class GLM:
    def __init__(self):
        pass

    @staticmethod
    def gradient(Xt: numpy.ndarray, y: numpy.ndarray, db_dlinpred: numpy.ndarray):
        return Xt @ (db_dlinpred - y)

    @staticmethod
    def hessian(X: numpy.ndarray, Xt: numpy.ndarray, d2b_dlinpred2: numpy.ndarray):
        return (Xt * d2b_dlinpred2) @ X


class ApproxIntegral:
    def __init__(self):
        pass

    @staticmethod
    # version for GLMs
    def ala_log(y, linpred_old, coef_new, gradient, hessian_inv, loglikelihood, coef_prior):
        p = len(coef_new)
        return loglikelihood(y, linpred_old) + numpy.log(coef_prior(coef_new, p)) + (p / 2) * numpy.log(2 * numpy.pi) + \
               0.5 * numpy.log(numpy.linalg.det(hessian_inv)) + 0.5 * gradient.transpose() @ hessian_inv @ gradient


# Here we set default g = 1. In fact, we might also think about adding rho parameter
def normal_prior(beta: numpy.ndarray, p: int, g=1):
    return 1 / (2 * numpy.pi * g)**(p / 2) * numpy.exp(- 1 / g * beta.dot(beta) / 2)


class ModelKernel:
    """
    Implements a Markov kernel to be used in SMC for the Latin square enumeration problem. See
    Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." (2020).

    Methods:
        sample: Samples new matrix given the current one by interchanging a pair of elements.    @static
    """
    def __init__(self):
        pass

    @staticmethod
    def sample(model_cur: numpy.ndarray) -> numpy.ndarray:
        """
        Takes a d x d matrix and selects a row i and two columns j1 and j2 at random.
        Then it swaps the values of x[i,j1] and x[i,j2].

        Parameters:
            model_cur:  Current particle to provide a new particle in re-sampling procedure.        [numpy.ndarray]

        Returns:
            New particle obtained by sampling form the Markov kernel given the current particle.    [numpy.ndarray]
        """
        p = len(model_cur)
        model_new = deepcopy(model_cur)
        i = numpy.random.choice(p)
        model_new[i] = not model_cur[i]
        while not model_new.any():
            i = numpy.random.choice(p)
            model_new[i] = not model_cur[i]
        return model_new


class ModelSelectionSMC:
    """
        Implements Adaptive SMC algorithm (Algorithm 17.3) form the book:
        Papaspiliopoulos, Chopin (2020) An introduction to sequential Monte Carlo

        Attributes:
            prior:           A prior distribution according to which the initial sample is     [callable]
                             drawn. Corresponds to pi_0 distribution.
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
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, likelihood: callable, db: callable, d2b: callable,
                 coef_init: numpy.array, model_init: numpy.array, coef_prior: callable, kernel: callable,
                 kernel_steps: int, particle_number: int, ess_min_ratio: float = 1/2, verbose: bool = False) -> None:
        """
        Instantiates Adaptive SMC sampler.

        Parameters:
            prior:           A prior distribution according to which the initial sample is      [callable]
                             drawn. Corresponds to pi_0 distribution.
            kernel:          An object with method 'sample'; Markov kernel that draws a new     [callable]
                             sample given the current sample (particle).
            kernel_steps:    Number of times the kernel is applied to a particle; defines a     [int]
                             'depth' of MCMC resampling.
            particle_number: Size of the sample                                                 [int]
            ess_min_ratio:   Ratio that defines the min Effective Sample Size that the          [float]
                             algorithm maintains at each step.
            verbose:         If True, the methods will print information about the process.     [bool]
        """
        self.X = X
        self.Xt = X.transpose()
        self.y = y
        self.likelihood = likelihood
        self.db = db
        self.d2b = d2b
        self.coef_init = coef_init
        self.model_init = model_init
        self.kernel = kernel
        self.kernel_steps = kernel_steps
        self.coef_prior = coef_prior  # This is the distribution that you start with.
        self.particle_number = particle_number
        self.verbose = verbose
        self.ess_min = particle_number * ess_min_ratio  # Papaspiliopoulos & Chopin states that the performance
                                                        # of the algorithm is pretty robust to this choice.
        # Initializing useful quantities for later
        self.iteration = 1  # Tracks the t variable
        self.particles = [None] * self.particle_number
        self.w_log = None  # unnormalized logweights
        self.w_normalized = None  # normalized weights
        self.w_hat_log = None  # reweighing multipliers
        self.logLt = 0.  # This will hold the cumulative value of the log normalising constant at time t.
        self.coefs = {}  # Used to save computed coefficients for models.
        self.integrated_loglikes = {}  # Used to save integrated likelihoods for models.
        self.computed_at = {}  # Used to save the number of the latest iteration where the coefficients and LL were upd.

    def multinomial_draw(self):
        """
        Returns an array of indices.

        For example:
        If we have 5 particles, then we might draw [1,0,0,2,2], which means we will resample particle 1 once
        and particles 4 and 5 two times.

        Returns:
            Sample of size n from ( 0, 1, ..., len(w_normalized) ) with replacement according to    [numpy.ndarray]
            probabilities given by w_normalized.
        """
        return multinomial(n=self.particle_number, p=self.w_normalized).rvs()[0]

    def compute_integrated_loglike(self, model: numpy.ndarray):
        model_id = get_model_id(model)
        model_seen = model_id in self.computed_at
        if model_seen:
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
            db = self.db(linpred_old)  # 1 / (1 + numpy.exp(-linpred_old))
            d2b = self.d2b(linpred_old)  # db * (1 - db)
            gradient = GLM.gradient(self.Xt[model, :], self.y, db)
            hessian = GLM.hessian(self.X[:, model], self.Xt[model, :], d2b)
            hessian_inv = numpy.linalg.inv(hessian)
            coef_new = coef_old - hessian_inv @ gradient
            if n_iterations - iter <= 2:
                integrated_loglike = ApproxIntegral.ala_log(self.y, linpred_old, coef_new,
                                                            gradient, hessian_inv,
                                                            self.likelihood, self.coef_prior)
                self.coefs[model_id][0] = self.coefs[model_id][1]
                self.coefs[model_id][1] = coef_new
                self.integrated_loglikes[model_id][0] = self.integrated_loglikes[model_id][1]
                self.integrated_loglikes[model_id][1] = integrated_loglike
        self.computed_at[model_id] = self.iteration
        return integrated_loglike

    def gibbs_iteration(self, model: numpy.ndarray):
        model_new = model
        for _ in range(self.kernel_steps):
            model_old = model_new
            model_new = self.kernel.sample(model_old)  # Draw a new sample from kernel
            integrated_loglike_model_old = self.compute_integrated_loglike(model_old)
            integrated_loglike_model_new = self.compute_integrated_loglike(model_new)
            accept_prob = numpy.exp(min(0, integrated_loglike_model_new - integrated_loglike_model_old))
            accept = numpy.random.binomial(1, accept_prob)
            model_new = model_new if accept else model_old
        return model_new

    def sample_init(self, cut: int):
        model_new = self.model_init
        for i in range(self.particle_number + cut):
            model_old = model_new
            model_new = self.gibbs_iteration(model_old)
            if i >= cut:
                self.particles[i] = model_new

    def resample(self) -> list:
        """
        Resample particles with repetition form the existing cloud.

        Returns:
            List of ancestors (with repetitions) of length given by "particle_number".
        """
        resample_indices = self.multinomial_draw()
        # Apply the metropolis step k times to each resampled particles
        ancestors = numpy.repeat(None, self.particle_number)  # Initialize vector of new particles
        if self.verbose:
            print("Doing Metropolis Resampling...")
        j = 0
        # n = number of times the particle has been resampled
        for particle_idx in (counter := Counter(resample_indices)):
            n = counter[particle_idx]
            ancestors[j:(j + n)] = particle_idx
            j += n
        if self.verbose:
            print("Resampling done!")
        return ancestors

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
        self.w_log = self.w_hat_log + numpy.array([self.integrated_loglikes[get_model_id(model)][1] -
                                                  self.integrated_loglikes[get_model_id(model)][0]
                                                  for model in self.particles])
        self.w_normalized = softmax(self.w_log)

    def update_logLt(self):
        """
        Updates the logarithm of the normalising constant by accumulating logarithm of mean of weights at each
        iteration. We do it this way since we are interested solely in the normalizing constant of the final
        distribution in the sequence.

        See pg 305 of Papaspiliopoulos / Chopin. I cross referenced with the `particles` library by Chopin.

        We can caluculate logLt by
        $$logLt = \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n )$$

        So for every iteration, we calculate the log normalising constant and add it to `self.LogLt`.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates attribute 'logLt'.
        """
        self.logLt += numpy.log(numpy.mean(numpy.exp(self.w_log)))

    def ess(self):
        """
        Calculates the effective sample size.
        """
        return 1 / sum(self.w_normalized**2)

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
        self.sample_init(cut=0)
        self.w_log = numpy.zeros(self.particle_number)
        self.w_normalized = numpy.repeat(1 / self.particle_number, self.particle_number)
        if self.verbose:
            print('Iteration 1 done! The initial particles sampled.')
        while self.iteration < 10:  # change to unnormalized weights being close to 1
            self.iteration += 1
            if self.ess() < self.ess_min:
                ancestor_idxs = self.resample()  # Get indexes of ancestors
                self.w_hat_log = numpy.zeros(self.particle_number)
            else:
                ancestor_idxs = numpy.arange(self.particle_number)
                self.w_hat_log = self.w_log
            self.particles = [self.gibbs_iteration(self.particles[ancestor_idx]) for ancestor_idx in ancestor_idxs]  # Update particles
            self.update_weights()  # Recalculate weights
            if self.verbose:
                print(f"Iteration {self.iteration} done!")
        if self.verbose:
            print('---SMC finished---\n')


if __name__ == '__main__':
    n_covariates = 30
    n_active = 3
    beta_true = numpy.concatenate([numpy.zeros(n_covariates - n_active), numpy.ones(n_active)])
    n = 1000
    rho = 0.0
    sigma_x = numpy.diag([1.0] * n_covariates)
    sigma_x[numpy.triu_indices(n_covariates, 1)] = rho
    sigma_x[numpy.tril_indices(n_covariates, -1)] = rho
    X = numpy.random.multivariate_normal(numpy.zeros(n_covariates), sigma_x, n)
    p = 1 / (1 + numpy.exp(- X @ beta_true))
    y = numpy.random.binomial(1, p, n)
    kernel = ModelKernel()
    particle_number = 500
    model_init = numpy.array([False] * n_covariates)
    model_init[numpy.random.choice(n_covariates)] = True

    def db(linpred):
        return 1 / (1 + numpy.exp(-linpred))

    def d2b(linpred):
        p = 1 / (1 + numpy.exp(-linpred))
        return p * (1 - p)

    smc = ModelSelectionSMC(X, y, likelihood=binomial_logit_logll, db=db, d2b=d2b,
                            coef_init=numpy.array([0] * n_covariates), model_init=model_init, coef_prior=normal_prior,
                            kernel=kernel, kernel_steps=5, particle_number=particle_number, verbose=True)
    smc.run()
    sampled_models = Counter([get_model_id(model) for model in smc.particles])
    print(get_model_id(beta_true != 0))
    # print(smc.particles)
    print(sampled_models)
    selected_model_id = max(sampled_models, key=sampled_models.get)
    print("Selected model: ", bin(selected_model_id)[2:])
