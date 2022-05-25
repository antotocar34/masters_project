import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

from .smc import SMC
from .utilities import get_model_id

class Kernel(ABC):
    @abstractmethod
    def sample(self, model):
        ...

    @abstractmethod
    def accept_reject(self, model, model_id, smc):
        model_new = ...
        model_id = get_model_id(model_new)
        return model_new, model_id

    def sweep(self, particles, smc):
        L = np.array([self.accept_reject(p, get_model_id(p), smc) for p in particles])
        new_particles = np.array([tup[0] for tup in L])
        new_particle_ids = np.array([tup[1] for tup in L])
        return new_particles, new_particle_ids
        

    
class SimpleGibbsKernel(Kernel):
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
        uniform = np.random.uniform(0, 1)
        if uniform <= 1e-200:  # For the sake of dealing with overflow.
            accept = False
        else:
            accept = np.log(1 / uniform - 1) >= postLLRatio
        model_new = model_new if accept else model
        model_id_new = model_id_new if accept else model_id
        return model_new, model_id_new



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
