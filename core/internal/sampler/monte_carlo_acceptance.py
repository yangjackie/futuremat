"""
Module contains various functions to calculate the acceptance/rejection probabilities in Monte Carlo Simulations.
"""
import core.models.constants as cnst
import math
import random
from typing import Optional


def boltzmann_distribution(E: float, T: float, unit: str = 'eV') -> Optional[float]:
    if unit is 'eV':
        return math.exp(-E / (cnst.KB_ev * T))
    else:
        raise NotImplementedError


class MetropolisAcceptance(object):

    def __init__(self,
                 temperature: float = 300.0,
                 unit: str = 'eV'):
        self.temperature = temperature
        self.unit = unit

    def accept(self, e_new: float, e_old: float) -> bool:
        if e_new <= e_old:
            return True
        else:
            p = boltzmann_distribution(e_new, self.temperature, unit=self.unit)
            p = p / boltzmann_distribution(e_old, self.temperature, unit=self.unit)
            r = random.random()
            if p >= r:
                return True
            else:
                return False


class WangLandauAcceptance(object):

    def __init__(self, temperature: float=300, unit: str='eV'):
        self.temperature = temperature
        self.unit = unit

    def accept(self, e_new: float, e_old: float) -> bool:
        #TODO: to be implemented
        raise NotImplementedError
