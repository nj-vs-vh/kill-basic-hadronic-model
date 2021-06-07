from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import re
from pathlib import Path

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Optional, Dict, Callable, Tuple, ClassVar
from nptyping import NDArray

CUR_DIR = Path(__file__).parent

import utils
from experiment import ExperimentalSED


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
    def plot(ax: plt.Axes, E_min: Optional[float] = None, E_max: Optional[float] = None):
        pass


# @dataclass
# class AnalyticalSED(ModelSED):
#     sed_func: Callable[..., float]  # SED(E, *model_params)
#     sed_integral: Callable[[float, float], float]


@dataclass
class TableLookupSED(ModelSED):
    E_lookup: NDArray[(Any,), float]
    sed_lookup: NDArray[(Any,), float]

    sed_scale: float = 1.0

    def __post_init__(self):
        assert self.E_lookup.shape == self.sed_lookup.shape

    def set_parameters(self, normalization: float):
        self.sed_scale = normalization

    allows_njit: ClassVar[bool] = True

    def get_average_func(self):
        E = self.E_lookup
        sed = self.sed_lookup

        @njit
        def model_mean(E_min, E_max, normalization):
            # simple mean works only for dense enough lookup tables
            return np.mean(normalization * sed[np.logical_and(E >= E_min, E <= E_max)])

        return model_mean

    def normalize_to_experimental_sed(self, exp: ExperimentalSED):
        """Modify model so that normalization of 1.0 gives same integral value as a given experimental SED"""
        E_min = np.min(exp.E_left)
        E_max = np.max(exp.E_right)
        good_exp_sed = exp.sed_mean[np.isfinite(exp.sed_lower)]
        exp_mean = np.mean(good_exp_sed)
        model_mean = np.mean(self.sed_lookup[np.logical_and(self.E_lookup > E_min, self.E_lookup < E_max)])
        self.sed_lookup *= exp_mean / model_mean
        self.sed_scale = 1.0

    def plot(self, ax: plt.Axes, E_min: Optional[float] = None, E_max: Optional[float] = None):
        utils.format_axes(ax)
        plot_label = self.name + f" ($\\times {self.sed_scale:.2f}$)"
        mask = np.ones_like(self.E_lookup, dtype=bool)
        if E_min is not None:
            mask[self.E_lookup < E_min] = False
        if E_max is not None:
            mask[self.E_lookup > E_max] = False
        ax.plot(self.E_lookup[mask], self.sed_scale * self.sed_lookup[mask], color=self.color, label=plot_label)


class BasicHadronicModelSED(TableLookupSED):
    @classmethod
    def _from_file(cls, filename: str, name: str, color: str) -> BasicHadronicModelSED:
        """File format by Timur, e.g. data/basic-hadronic-model/SED-KD10-Basic-0.140-Combined"""
        data = np.loadtxt(filename)
        return cls(name=name, color=color, E_lookup=data[:, 0], sed_lookup=data[:, 1])

    INTERP_FILES_DIR = CUR_DIR / 'data/basic-hadronic-model'
    data_for_interp: Optional[Dict[float, BasicHadronicModelSED]] = None

    @classmethod
    def _read_data_for_interpolation(cls):
        data_for_interp = dict()
        filename_patt = r'SED-KD10-Basic-(.*)-Combined'
        for filename in cls.INTERP_FILES_DIR.iterdir():
            z_match = re.match(filename_patt, filename.name)
            if z_match:
                z = float(z_match.groups()[0])
                data_for_interp[z] = cls._from_file(filename, str(z), color='b')
        cls.data_for_interp = data_for_interp

    @classmethod
    def at_z(cls, z: float):
        """Main method for getting model SED at a given z by interpolating known values"""
        if cls.data_for_interp is None:
            cls._read_data_for_interpolation()
        available_z_vals = np.array(list(cls.data_for_interp.keys()))
        if not min(available_z_vals) <= z <= max(available_z_vals):
            raise ValueError(f"Requested z value falls outside of interpolation range")
        
        find_nearest = lambda arr, value: arr[(np.abs(arr - value)).argmin()]
        z_nearest = find_nearest(available_z_vals, z)
        E_interp = cls.data_for_interp[z_nearest].E_lookup
        if np.abs(z_nearest - z) < 0.0001:
            sed_interp = cls.data_for_interp[z_nearest].sed_lookup
        else:
            z_left = find_nearest(available_z_vals[available_z_vals < z], z)
            z_right = find_nearest(available_z_vals[available_z_vals >= z], z)
            left_contrib = (z_right - z) / (z_right - z_left)
            right_contrib = (z - z_left) / (z_right - z_left)
            left_model = cls.data_for_interp[z_left]
            right_model = cls.data_for_interp[z_right]
            sed_interp = left_model.sed_lookup * left_contrib + right_model.sed_lookup * right_contrib
        return cls(
            name=f'Basic hadronic model ($z = {z}$)',
            color='#794ece',
            E_lookup=E_interp,
            sed_lookup=sed_interp,
        )


if __name__ == "__main__":
    BasicHadronicModelSED.at_z(0.169)
