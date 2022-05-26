import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.stats import bernoulli
from scipy.special import softmax

from alasmc.smc import SMC
from alasmc.utilities import get_model_id


class Kernel(ABC):
    @abstractmethod
    def initialize(self, smc, ancestor_idxs):
        ...

    @abstractmethod
    def sample(self, model):
        ...

    @abstractmethod
    def accept_reject(self, model, model_id, smc):
        model_new = ...
        model_id = get_model_id(model_new)
        return model_new, model_id

    def sweep(self, particles, smc):
        L = [self.accept_reject(p, get_model_id(p), smc) for p in particles]
        new_particles = np.array([tup[0] for tup in L])
        new_particle_ids = np.array([tup[1] for tup in L])
        return new_particles, new_particle_ids
        

    
class SimpleGibbsKernel(Kernel):
    def initialize(self, smc, ancestor_idxs):
        return

    def sample(self, model_cur: np.ndarray) -> np.ndarray:
        """

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

    def accept_reject(self, model, model_id, smc:SMC):
        model_id = get_model_id(model)
        model_new = self.sample(model)  # Draw a new sample from kernel
        model_id_new = get_model_id(model_new)
        postLLRatio = smc.compute_integrated_loglike(model, model_id) + smc.model_prior_log(model) - \
            smc.compute_integrated_loglike(model_new, model_id_new) - smc.model_prior_log(model_new)
        uniform = np.random.uniform()
        if uniform <= 1e-200:  # For the sake of dealing with overflow.
            accept = False
        else:
            accept = np.log(1 / uniform - 1) >= postLLRatio
        model_new = model_new if accept else model
        model_id_new = model_id_new if accept else model_id
        return model_new, model_id_new

class LogisticKernel(Kernel):
    def __init__(self):
        self.model_matrix = None
        self.coef_dict = None
        self.log_kernel_func = None
        
    def solve_regression_problem(self, i: int, particle_weights, regressor_index):
        y = self.model_matrix[:, i] ##
        X = self.model_matrix[:, regressor_index] ##
       
        reg = LogisticRegression(fit_intercept=True, penalty="none")
        reg.fit(X, y, particle_weights)
        intercept = reg.intercept_
        coefs = np.reshape(reg.coef_, reg.coef_.size) # The reshape is to make the vector one dimensional

        # Return a vector with 0 outside the `regressor index` indices
        return_vector = np.zeros(i)
        return_vector[regressor_index] = coefs
        return np.concatenate( [intercept, return_vector] )

    def build_kernel(self, particles, particle_weights):
        self.model_matrix = np.stack(particles, axis=0) ##
        p = self.model_matrix.shape[1] ##
    
        var_average = np.average(particles, weights=particle_weights, axis=0) # Take the weights average of marginals

        def r(i,j):
            x_i = self.model_matrix[:,i] ##
            x_j = self.model_matrix[:,j] ##
            bar_x_i = var_average[i]
            bar_x_j = var_average[j]
            x_ij = np.average(x_i * x_j, weights=particle_weights)

            # Not sure how to handle dividing by 0
            if (x_ij - (bar_x_i*bar_x_j)) == 0:
                return 0
            return ( 
                    x_ij - (bar_x_i * bar_x_j) 
                    / 
                    np.sqrt(bar_x_i*(1-bar_x_i)*bar_x_j*(1-bar_x_j)) 
                    )

        epsilon = 0.02
        # Choose variables to avoid doing a logistic regression for.
        include = (var_average < 1-epsilon) & (var_average > epsilon)
        # delta = 0.075
        delta = 0.075

        # Regressor index is an dictionary containing that maps an variable index
        # To an array of the indicies to regress that variable upon
        regressor_indices = {
                i :(
                    np.array([ 
                      j for j in range(i) if (abs(r(i,j)) >= delta)
                    ]) if include[i] else np.empty(0, dtype=np.int64)
                    )
                for i in range(1,p)
                }
        regressor_indices[0] = np.empty(0, dtype=np.int64)

        # Make a dictionary that gives the coefficients 
        self.coef_dict = {
            i:(
                self.solve_regression_problem(i, particle_weights, regressor_indices[i]) 
                if (regressor_indices[i].size > 0) else
                np.concatenate([np.array([var_average[i]]), np.zeros(i)])
              )
            for i in range(1,p)
                }

        self.coef_dict[0] = np.array([var_average[0]])

        def log_kernel_func(gamma):
            coef_dict = self.coef_dict ##

            acc = 0
            for i in range(len(gamma)):
                prob = expit(-coef_dict[i][0] - np.dot(coef_dict[i][1:], gamma[0:i]))

                # acc += np.log(prob)*(gamma[i]) + np.log((1-prob))*(1-gamma[i])
                acc += np.log(prob) if gamma[i] else np.log((1-prob))
            return acc

        return log_kernel_func

    def initialize(self, smc, ancestor_idxs):
        print("Building Logistic Kernel...")
        particles = np.array(smc.particles)[ancestor_idxs]
        particle_weights = softmax(np.array(smc.w_log)[ancestor_idxs]) # Take the weights of the resampled particles
        self.log_kernel_func = self.build_kernel(particles, particle_weights)
        print("Logistic Kernel Built.")

    def sample(self, model_cur):
        """
        Returns the sample model and it's probability in one go

        See procedure 6 in Schafer, Chopin (2011)
        """
        coef_dict = self.coef_dict
        new_model = np.zeros(model_cur.size, dtype=bool)
        prob=1
        for i in range(model_cur.size):
            q = expit(-coef_dict[i][0] - np.dot(coef_dict[i][1:], new_model[0:i]))
            new_model[i] = bool(bernoulli.rvs(q))
            prob = prob*q if new_model[i] else prob*(1-q)
        return new_model, prob

    def accept_reject(self, model, model_id, smc):
        model_new, prob = self.sample(model)
        model_new_id = get_model_id(model_new)
        uniform = np.random.uniform()
        logratio = (
                (self.log_kernel_func(model)-np.log(prob))
                +
                (smc.compute_integrated_loglike(model_new, model_new_id) - smc.compute_integrated_loglike(model, model_id))
                )
        accept = np.exp(logratio) > uniform
        model_new = model_new if accept else model
        model_new_id = model_new_id if accept else model_id
        return model_new, model_new_id
