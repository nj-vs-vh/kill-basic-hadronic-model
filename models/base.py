from __future__ import annotations

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Optional, Callable, ClassVar


@dataclass
class ModelSED(ABC):
    name: str
    color: str

    allows_njit: ClassVar[bool] = False

    @abstractmethod
    def get_average_func(self) -> Callable[..., float]:
        """Must return function that calculates model SED average over a given energy interval.
        
        Expected signature is model_average(E_min, E_max, *model_params)
        """
        pass

    @abstractmethod
    def set_parameters(self, *args):
        """
        ModelSED instance is **both** parametrized model and a particular state of that model.
        This function provides interface to set parameters, and its signature should specify their meaning.
        """

    @abstractmethod
    def plot(self, ax: plt.Axes, E_min: Optional[float] = None, E_max: Optional[float] = None):
        pass