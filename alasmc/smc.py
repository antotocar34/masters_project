from abc import ABC, abstractmethod
from scipy.stats import multinomial
from collections import Counter
from scipy.special import softmax
import numpy as np


class SMC(ABC):
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
            logLt:           Logarithm of the estimated normalized constant, basically:         [float]
                             \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n ).
    """
    def __init__(self, 
                 kernel: callable,
                 kernel_steps: int,
                 particle_number: int,
                 maxit_smc: int = 40,
                 ess_min_ratio: float = 1/2, 
                 verbose: bool = False) -> None:
        self.kernel = kernel
        self.kernel_steps = kernel_steps
        self.particle_number = particle_number
        self.maxit_smc = maxit_smc
        self.verbose = verbose
        self.ess_min = particle_number * ess_min_ratio  # Papaspiliopoulos & Chopin states that the performance
                                                        # of the algorithm is pretty robust to this choice.
        # Initializing useful quantities for later
        self.iteration = 1  # Tracks the t variable
        self.particles = [None] * self.particle_number  # Make into np array?
        self.w_log = None  # unnormalized logweights
        self.w_normalized = None  # normalized weights
        self.w_hat_log = None  # reweighing multipliers
        self.logLt = 0.  # This will hold the cumulative value of the log normalising constant at time t.

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


    @abstractmethod
    def sample_init(self):
        """
        Sample Inital Particles.
        """
        pass

    def resample(self) -> list:
        """
        Resample particles with repetition form the existing cloud.

        Returns:
            List of ancestors (with repetitions) of length given by "particle_number".
        """
        resample_indices = self.multinomial_draw()
        # Apply the metropolis step k times to each resampled particles
        ancestors = np.repeat(None, self.particle_number)  # Initialize vector of new particles
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

    @abstractmethod
    def update_weights(self) -> None:
        """
        Weight update.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates attributes 'w' and 'w_normalized'.
        """
        self.w_log = self.w_hat_log + ...
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
        self.logLt += np.log(np.mean(np.exp(self.w_log)))

    def ess(self):
        """
        Calculates the effective sample size.
        """
        return 1 / sum(self.w_normalized**2)

    @abstractmethod
    def run(self):  # The code is never run---it is merely suggestive
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
        while self.iteration < self.maxit_smc:  # change to unnormalized weights being close to 1
            self.iteration += 1
            if self.ess() < self.ess_min:
                ancestor_idxs = self.resample()  # Get indexes of ancestors
                self.w_hat_log = np.zeros(self.particle_number)
            else:
                ancestor_idxs = np.arange(self.particle_number)
                self.w_hat_log = self.w_log
            self.particles = ...  # Update particles with MCMC kernel
            self.update_weights()  # Recalculate weights
            if self.verbose:
                print(f"Iteration {self.iteration} done!")
        if self.verbose:
            print('---SMC finished---\n')
