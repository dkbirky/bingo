import numpy as np

from .generalized_crowding import GeneralizedCrowding


class BayesCrowding(GeneralizedCrowding):
    """Crowding using Bayesian model selection

    Fitness of individuals are assumed to be a measure of model evidence, such
    that a ratio between two fitness values gives the Bayes Factor.

    Parameters
    ----------
    logscale : bool
        Whether fitnesses of the individuals is in log space. Note that
        fitnesses are assumed to be *negative* log probabilities. Default True.
    """
    def __init__(self, logscale=True):
        self._logscale = logscale
        super().__init__()

    def _return_most_fit(self, child, parent):
        if np.isnan(parent.fitness):
            return child
        if np.isnan(child.fitness):
            return parent

        if self._logscale:
            prob = np.exp(parent.fitness - child.fitness)
            prob = prob / (prob + 1)
        else:
            prob = child.fitness / (parent.fitness + child.fitness)
        print(np.random.random(), prob, np.random.random() < prob)
        return child if np.random.random() < prob else parent