from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from dataclasses import dataclass

from nptyping import NDArray
from typing import Callable, ClassVar, Tuple

import utils
from .base import ModelSED
from experiment import Object

import astropy.units as u
from agnprocesses.processes.ebl import tau_gilmore


@dataclass
class AnalyticalSED(ModelSED):
    sed_func: Callable[..., float]  # SED(E, *model_params)
    sed_integral: Callable[..., float]  # SED_int(E_min, E_max, *model_params)

    model_params: NDArray = None

    allows_njit: ClassVar[bool] = False

    def get_average_func(self) -> Callable[..., float]:
        sed_int = self.sed_integral

        def avg(E_min, E_max, *model_params):
            return sed_int(E_min, E_max, *model_params) / (E_max - E_min)
        
        return njit(avg) if self.allows_njit else avg

    def set_parameters(self, *model_params):
        self.model_params = model_params
        
    def plot(self, ax: plt.Axes, E_min: float, E_max: float):
        if self.model_params is None:
            raise ValueError(f"You must set model parameters before plotting!")
        utils.format_axes(ax)
        E = np.logspace(np.log10(E_min), np.log10(E_max), 100)
        ax.plot(E, self.sed_func(E, *self.model_params), color=self.color, label=self.name)


class LogparabolaThroughEblSED(AnalyticalSED):
    COLOR = '#e09758'
    allows_njit = True

    def __init__(self, E_min: float, E_max: float, redshift: float):
        E_tau_lookup = np.logspace(np.log10(E_min), np.log10(E_max), 100)
        tau_lookup = tau_gilmore(E_tau_lookup * u.TeV, redshift)

        @njit
        def sed(E: float, A: float, c1: float, c2: float) -> float:
            logparabola = A * E ** (-c1 - c2 * np.log10(E))
            tau = np.interp(E, E_tau_lookup, tau_lookup)
            return logparabola * np.exp(-tau)
        
        super().__init__(
            name=f'Logparabola source + EBL abs. (z={redshift})',
            color=self.__class__.COLOR,
            sed_func=sed,
            sed_integral=utils.trapz_integral_func(sed, allows_njit=self.allows_njit, n_pts=5),
        )

    @classmethod
    def for_object(cls, obj: Object) -> LogparabolaThroughEblSED:
        return cls(E_min=obj.E_min, E_max=obj.E_max, redshift=obj.z)

    def estimate_params(self, obj: Object) -> Tuple[float, float, float]:
        c1_est = 0.05
        c2_est = 0.1

        E_ref = 1.0  # TeV
        E_center_joint = np.concatenate(tuple(0.5 * (sed.E_left + sed.E_right) for sed in obj.seds))
        sed_joint = np.concatenate(tuple(sed.sed_mean for sed in obj.seds))
        sed_join_mask = np.concatenate(tuple(np.isfinite(sed.sed_lower) for sed in obj.seds))
        i_closest = np.argmin(np.abs(E_center_joint[sed_join_mask] - E_ref))

        A_no_absorption = sed_joint[sed_join_mask][i_closest]
        # compensating for EBL attenuation
        A_est = A_no_absorption ** 2 / self.sed_func(1.0, A_no_absorption, c1_est, c2_est)
        return A_est, c1_est, c2_est

    def set_parameters(self, A: float, c1: float, c2: float):
        self.model_params = (A, c1, c2)
