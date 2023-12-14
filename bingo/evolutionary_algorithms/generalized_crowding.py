"""The generalized crowding evolutionary algorithm

This module defines the basis of the generalized crowding evolutionary
algorithm in bingo analyses. The next generation is selected by pairing parents
with their offspring and advancing the most fit of the two.
"""
import numpy as np

from .evolutionary_algorithm import EvolutionaryAlgorithm
from ..variation.var_and import VarAnd
from ..selection.deterministic_crowding import DeterministicCrowding


class GeneralizedCrowdingEA(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the generalized crowding evolutionary algorithm in bingo.

    Parameters
    ----------
    evaluation : evaluation
        The evaluation algorithm that sets the fitness on the population.
    crossover : Crossover
        The algorithm that performs crossover during variation.
    mutation : Mutation
        The algorithm that performs mutation during variation.
    crossover_probability : float
        Probability that crossover will occur on an individual.
    mutation_probability : float
        Probability that mutation will occur on an individual.
    selection : CrowdingSelection
        Selection phase. Default DeterministicCrowding.

    Attributes
    ----------
    evaluation : Evaluation
        evaluation instance to perform evaluation on a population
    selection : CrowdingSelection
        Performs selection on a population via deterministic crowding
    variation : VarAnd
        Performs VarAnd variation on a population
    diagnostics : `bingo.evolutionary_algorithms.ea_diagnostics.EaDiagnostics`
        Public to the EA diagnostics
    """
    def __init__(self, evaluation, crossover, mutation, crossover_probability,
                 mutation_probability, selection=None, multiobject=False, outliers=True):
        self._multiobject = multiobject
        self._outliers = outliers
        if selection is None:
            selection = DeterministicCrowding()

        super().__init__(variation=VarAnd(crossover, mutation,
                                          crossover_probability,
                                          mutation_probability),
                         evaluation=evaluation,
                         selection=selection)

    def generational_step(self, population):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of chromosomes
            The population at the start of the generational step

        Returns
        -------
        list of chromosomes :
            The next generation of the population
        """
        offspring = self.variation(population, len(population))
        self.evaluation(population)
        self.evaluation(offspring)
        if self._multiobject:
            nmll_store = []
            for i in range(len(population)):
                nmll_store.append(population[i].nmll)
            for i in range(len(offspring)):
                nmll_store.append(offspring[i].nmll)
            nmll_store = np.array(nmll_store)
            if self._outliers:
                mu = np.nanmean(nmll_store)
                std = np.nanstd(nmll_store)
                nmll_store[np.where(nmll_store>mu+3*std)] = np.nan
                nmll_store[np.where(nmll_store<mu-3*std)] = np.nan
            nmll_store_new = (nmll_store - np.nanmin(nmll_store))/(np.nanmax(nmll_store) - np.nanmin(nmll_store))
            nmll_store_new[np.where(nmll_store<0)] *= -1
            nmll_store = nmll_store_new
            #nmll_store = (self._nmll_range[1]-self._nmll_range[0])/(np.nanmax(nmll_store) - np.nanmin(nmll_store))* \
            #                        (nmll_store-np.nanmax(nmll_store))+self._nmll_range[1]
           # print(nmll_store)
            #print(np.exp(nmll_store))
            #nmll_store = np.exp(nmll_store)/np.nansum(np.exp(nmll_store))
            #print(nmll_store)
            count = 0
            for i in range(len(population)):
                population[i].norm_nmll = nmll_store[count]
                count += 1
            for i in range(len(offspring)):
                offspring[i].norm_nmll = nmll_store[count]
                count += 1
            self.evaluation(population)
            self.evaluation(offspring)
        self.update_diagnostics(population, offspring)
        next_gen = self.selection(population + offspring, len(population))
        np.random.shuffle(next_gen)
        return next_gen
