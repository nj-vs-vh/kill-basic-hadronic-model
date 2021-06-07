from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import re
import yaml
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from typing import Any, Callable, ClassVar, List, Tuple
from nptyping import NDArray

from models.base import ModelSED
import utils


CUR_DIR = Path(__file__).parent


@dataclass
class ExperimentalSED(ABC):
    # All energies in TeV!
    name: str
    color: str
    # bin edges
    E_left: NDArray[(Any,), float]
    E_right: NDArray[(Any,), float]
    # SED distribution characteristics
    sed_mean: NDArray[(Any,), float]
    sed_lower: NDArray[(Any,), float]  # sigma or 95% CL???
    sed_upper: NDArray[(Any,), float]

    E_uncertainty: float = 0.0001  # essentialy no uncertainty by default

    DATA_DIR: ClassVar[Path] = CUR_DIR / "data"

    @classmethod
    @abstractmethod
    def _data_file_pattern(cls, obj_name: str) -> str:
        """Regular expression pattern for finding data file from object name"""
        pass

    @classmethod
    @abstractmethod
    def _from_file(cls, filename: str, name: str) -> ExperimentalSED:
        pass

    @classmethod
    def for_object(cls, obj_name: str) -> ExperimentalSED:
        """Object name is e.g. 1ES0229+200"""
        dir_patt = f".*{re.escape(obj_name)}"
        file_patt = cls._data_file_pattern(obj_name)
        object_path = None
        for dir in cls.DATA_DIR.iterdir():
            if re.match(dir_patt, dir.name):
                for file in dir.iterdir():
                    if re.match(file_patt, file.name):
                        object_path = file
                        break
        if object_path is None:
            raise FileExistsError(f"File with data for {obj_name} not found")
        return cls._from_file(filename=str(object_path), name=obj_name)

    def get_logprior(self) -> Callable[[float], float]:
        log_E_uncertainty_disp: float = np.log(1 + self.E_uncertainty) ** 2

        @njit
        def logprior(E_factor: float) -> float:
            log_E_factor = np.log(E_factor)
            if np.isnan(log_E_factor) or np.abs(log_E_factor) > 5 * log_E_uncertainty_disp:
                return -np.inf  # cutting prior on 5 sigma to allow skip likelihood evaluation
            else:
                return log_E_factor ** 2 / (2 * log_E_uncertainty_disp)

        return logprior

    def get_loglikelihood(self, model: ModelSED):
        # to allow njit-ted function to access these without going to class instance
        bin_count = self.sed_mean.size
        E_left = self.E_left
        E_right = self.E_right
        sed_mean = self.sed_mean
        sed_upper = self.sed_upper
        sed_lower = self.sed_lower

        model_average = model.get_average_func()

        def loglikelihood(model_params: Tuple[float], E_factor: float):
            sed_mean_shifted = sed_mean * E_factor ** 2
            sed_upper_shifted = sed_upper * E_factor ** 2
            sed_lower_shifted = sed_lower * E_factor ** 2
            logL = 0.0
            for i in range(bin_count):
                # shifting experimental bin on E_factor and computing average model SED in shifted bin
                model_sed_in_shifted_bin = model_average(E_factor * E_left[i], E_factor * E_right[i], *model_params)
                if np.isnan(model_sed_in_shifted_bin):
                    return -np.inf
                model_minus_mean = model_sed_in_shifted_bin - sed_mean_shifted[i]
                if model_minus_mean > 0:  # using upper sigma
                    logL -= model_minus_mean ** 2 / (
                        2 * (sed_upper_shifted[i] - sed_mean_shifted[i]) ** 2
                    )
                else:  # using lower sigma, if exists
                    if np.isfinite(sed_lower_shifted[i]):
                        logL -= model_minus_mean ** 2 / (
                            2 * (sed_mean_shifted[i] - sed_lower_shifted[i]) ** 2
                        )
            return logL
        
        if model.allows_njit:
            loglikelihood = njit(loglikelihood)

        return loglikelihood

    def with_E_factor(self, E_factor: float, color: str = None) -> ExperimentalSED:
        return self.__class__(
            name=self.name + f" shifted {100 * (E_factor - 1):.2f} %",
            color=color or self.color,
            E_right=self.E_right * E_factor,
            E_left=self.E_left * E_factor,
            sed_mean=self.sed_mean * E_factor ** 2,
            sed_upper=self.sed_upper * E_factor ** 2,
            sed_lower=self.sed_lower * E_factor ** 2,
        )

    def plot(self, ax: plt.Axes):
        utils.format_axes(ax)
        with_both_bounds = np.logical_and(np.isfinite(self.sed_lower), np.isfinite(self.sed_upper))
        E_mean = np.sqrt(self.E_left * self.E_right)
        E_err_left = E_mean - self.E_left
        E_err_right = self.E_right - E_mean

        fmt = "o"
        ax.errorbar(
            E_mean[with_both_bounds],
            self.sed_mean[with_both_bounds],
            xerr=[E_err_left[with_both_bounds], E_err_right[with_both_bounds]],
            yerr=(
                (self.sed_mean - self.sed_lower)[with_both_bounds],
                (self.sed_upper - self.sed_mean)[with_both_bounds],
            ),
            fmt=fmt,
            color=self.color,
            label=self.name,
        )
        with_upper_bound = np.logical_not(with_both_bounds)
        ax.errorbar(
            E_mean[with_upper_bound],
            self.sed_upper[with_upper_bound],
            xerr=[E_err_left[with_upper_bound], E_err_right[with_upper_bound]],
            yerr=self.sed_upper[with_upper_bound] / 2,
            uplims=True,
            fmt=fmt,
            color=self.color,
        )


class FermiSED(ExperimentalSED):
    COLOR = "#dd4040"
    SYSTEMATIC_E_UNCERAINTY = 0.1

    @classmethod
    def _from_file(cls, filename: str, name: str) -> FermiSED:
        with open(filename) as fermifile:
            fermifile.readline()  # skipping first line
            fermi_data = []
            for line in fermifile:
                fermi_data.append(line.split())
        fermi_data = np.array(fermi_data, dtype=float)
        fermi_data *= 1e-6  # MeV -> TeV
        fermi_data[fermi_data < 0] = -np.inf  # -1 is used to signify 'no lower bound'
        return cls(
            name=name + " (Fermi)",
            color=cls.COLOR,
            E_left=fermi_data[:, 0],
            E_right=fermi_data[:, 2],
            sed_mean=fermi_data[:, 3],
            sed_lower=fermi_data[:, 3] - fermi_data[:, 4],
            sed_upper=fermi_data[:, 3] + fermi_data[:, 5],
            E_uncertainty=cls.SYSTEMATIC_E_UNCERAINTY,
        )

    @classmethod
    def _data_file_pattern(cls, obj_name: str) -> str:
        """Regular expression pattern for finding data file from object name"""
        obj_name = obj_name[3:]
        return fr"1ES\s?{re.escape(obj_name)}-Fermi"


class IactSED(ExperimentalSED):
    COLOR = "#4ece7d"
    SYSTEMATIC_E_UNCERAINTY = 0.35

    @classmethod
    def _from_file(cls, filename: str, name: str):
        with open(filename) as iactfile:
            iactfile.readline()  # skipping first line
            iact_data = []
            for line in iactfile:
                iact_data.append(line.split())
        iact_data = np.array(iact_data, dtype=float)

        # WTF?????
        col_offset = 1 if re.match('.*' + re.escape('1ES0229+200-IACT'), filename) else 0

        E_mean_bin = iact_data[:, col_offset + 0]

        def dnde2sed(dnde):
            return dnde * np.square(E_mean_bin)

        sed_mean = dnde2sed(iact_data[:, col_offset + 3])
        return cls(
            name=name + " (IACT)",
            color=cls.COLOR,
            E_left=iact_data[:, col_offset + 1],
            E_right=iact_data[:, col_offset + 2],
            sed_mean=sed_mean,
            sed_upper=sed_mean + dnde2sed(iact_data[:, col_offset + 4]),
            sed_lower=sed_mean + dnde2sed(iact_data[:, col_offset + 5]),
            E_uncertainty=cls.SYSTEMATIC_E_UNCERAINTY,
        )

    @classmethod
    def _data_file_pattern(cls, obj_name: str) -> str:
        """Regular expression pattern for finding data file from object name"""
        obj_name = obj_name[3:]
        return fr"1ES\s?{re.escape(obj_name)}-IACT"


with open(CUR_DIR / "data/redshifts.yaml") as f:
    redshift_data = yaml.safe_load(f)


@dataclass
class Object:
    name: str
    z: float
    seds: List[ExperimentalSED]

    @classmethod
    def by_name(cls, name: str):
        return cls(
            name=name,
            z=redshift_data[name],
            seds=[
                FermiSED.for_object(name),
                IactSED.for_object(name),
            ]
        )

    def get_joint_logprior(self):
        logpriors = [sed.get_logprior() for sed in self.seds]
        
        return lambda E_factors: sum(
            logprior(E_factor) for logprior, E_factor in zip(logpriors, E_factors)
        )

    def get_joint_loglike(self, model: ModelSED):
        loglikes = [sed.get_loglikelihood(model) for sed in self.seds]

        return lambda model_params, E_factors: sum(
            loglike(model_params, E_factor)
            for loglike, E_factor in zip(loglikes, E_factors)
        )

    def get_joint_logposterior(self, model: ModelSED):

        joint_logprior = self.get_joint_logprior()
        joint_loglike = self.get_joint_loglike(model)

        def logposterior(model_params: Tuple[float], E_factors: Tuple[float]):
            logp = joint_logprior(E_factors)
            if np.isinf(logp):
                return -np.inf
            else:
                return logp + joint_loglike(model_params, E_factors)

        return logposterior

    def with_E_factors(self, *E_factors) -> Object:
        if len(E_factors) != len(self.seds):
            raise ValueError(f"Number of E factors must be the same as the number of SEDs for the object")
        return self.__class__(
            name=self.name + ' (shifted SEDs)',
            z=self.z,
            seds=[sed.with_E_factor(E_factor) for sed, E_factor in zip(self.seds, E_factors)]
        )

    @property
    def E_min(self):
        return min(np.min(sed.E_left) for sed in self.seds)
    
    @property
    def E_max(self):
        return max(np.max(sed.E_right) for sed in self.seds)

    def plot(self, ax: plt.Axes):
        for sed in self.seds:
            sed.plot(ax)


all_objects = [Object.by_name(name) for name in redshift_data.keys()]


if __name__ == "__main__":
    FermiSED.for_object("1ES0229+200")
    IactSED.for_object("1ES0229+200")

    Object.by_name('1ES0347-121')
