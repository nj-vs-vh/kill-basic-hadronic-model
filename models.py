from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

from dataclasses import dataclass

from typing import Any, Optional, Dict
from nptyping import NDArray

CUR_DIR = Path(__file__).parent

import utils
from experiment import ExperimentalSED


@dataclass
class ModelSED:
    name: str
    color: str
    E: NDArray[(Any,), float]
    sed: NDArray[(Any,), float]
    E_scale: Optional[float] = None

    def __post_init__(self):
        assert self.E.shape == self.sed.shape

    def normalized_to_experimental_sed(self, exp: ExperimentalSED):
        E_min = np.min(exp.E_left)
        E_max = np.max(exp.E_right)
        good_exp_sed = exp.sed_mean[np.isfinite(exp.sed_lower)]
        exp_mean = np.mean(good_exp_sed)
        model_mean = np.mean(self.sed[np.logical_and(self.E > E_min, self.E < E_max)])
        return self.with_normalization(exp_mean / model_mean)

    def with_normalization(self, k: float) -> ModelSED:
        return ModelSED(
            name=self.name,
            color=self.color,
            E=self.E,
            sed=self.sed * k,
            E_scale=k if self.E_scale is None else k * self.E_scale,
        )

    def plot(self, ax: plt.Axes, E_min: Optional[float] = None, E_max: Optional[float] = None):
        utils.format_axes(ax)
        plot_label = self.name if self.E_scale is None else self.name + f" x {self.E_scale:.2f}"
        mask = np.ones_like(self.E, dtype=bool)
        if E_min is not None:
            mask[self.E < E_min] = False
        if E_max is not None:
            mask[self.E > E_max] = False
        ax.plot(self.E[mask], self.sed[mask], color=self.color, label=plot_label)


class BasicHadronicModelSED(ModelSED):

    @classmethod
    def _from_file(cls, filename: str, name: str, color: str) -> ModelSED:
        """File format by Timur, e.g. data/basic-hadronic-model/SED-KD10-Basic-0.140-Combined"""
        data = np.loadtxt(filename)
        return cls(name=name, color=color, E=data[:, 0], sed=data[:, 1])

    INTERP_FILES_DIR = CUR_DIR / 'data/basic-hadronic-model'
    data_for_interp: Optional[Dict[float, ModelSED]] = None

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
        E_interp = cls.data_for_interp[z_nearest].E
        if np.abs(z_nearest - z) < 0.0001:
            sed_interp = cls.data_for_interp[z_nearest].sed
        else:
            z_left = find_nearest(available_z_vals[available_z_vals < z], z)
            z_right = find_nearest(available_z_vals[available_z_vals >= z], z)
            left_contrib = (z_right - z) / (z_right - z_left)
            right_contrib = (z - z_left) / (z_right - z_left)
            left_model = cls.data_for_interp[z_left]
            right_model = cls.data_for_interp[z_right]
            sed_interp = left_model.sed * left_contrib + right_model.sed * right_contrib
        return cls(
            name=f'Basic hadronic model ($z = {z}$)',
            color='#794ece',
            E=E_interp,
            sed=sed_interp,
        )


if __name__ == "__main__":
    BasicHadronicModelSED.at_z(0.169)
