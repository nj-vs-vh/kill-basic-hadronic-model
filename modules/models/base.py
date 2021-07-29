from __future__ import annotations

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Optional, Callable, ClassVar, Tuple


@dataclass
class ModelSED(ABC):
    name: str
    color: str
    linestyle: str

    allows_njit: ClassVar[bool] = False

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def get_average_func(self) -> Callable[..., float]:
        """Must return function that calculates model SED average over a given energy interval.
        
        Expected signature is model_average(E_min, E_max, *model_params)
        """
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        pass

    @abstractmethod
    def set_parameters(self, *args):
        """
        ModelSED instance is **both** parametrized model and a particular state of that model.
        This function provides interface to set parameters, and its signature should specify their meaning.
        """

    @abstractmethod
    def get_parameters(self) -> Tuple[float]:
        pass

    @abstractmethod
    def plot(self, ax: plt.Axes, E_min: Optional[float] = None, E_max: Optional[float] = None):
        pass