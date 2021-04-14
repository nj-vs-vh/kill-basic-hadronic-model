from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

from numba import njit

from typing import Any, Tuple, Callable
from nptyping import NDArray


def format_axes(ax: plt.Axes):
    ax.set_xlabel("E, $TeV$")
    ax.set_xscale("log")
    ax.set_ylabel("SED, $TeV \, cm^{-2}, \, s^{-1}$")
    ax.set_yscale("log")


@dataclass
class ModelSED:
    name: str
    color: str
    E: NDArray[(Any,), float]
    sed: NDArray[(Any,), float]

    def with_normalization(self, k: float) -> ModelSED:
        return ModelSED(
            name=self.name + f" x {k:.2f}",
            color=self.color,
            E=self.E,
            sed=self.sed * k,
        )

    def plot(self, ax: plt.Axes):
        format_axes(ax)
        ax.plot(self.E, self.sed, color=self.color, label=self.name)


@dataclass
class ExperimentalSED:
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

    def set_systematic_energy_uncertainty(self, unc: float):
        self.E_uncertainty = unc

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
        bin_count = self.sed_mean.size
        model_E = model.E
        model_sed = model.sed
        # to allow njit-ted function to access these without going to class instance
        E_left = self.E_left
        E_right = self.E_right
        sed_mean = self.sed_mean
        sed_upper = self.sed_upper
        sed_lower = self.sed_lower

        @njit
        def loglikelihood(model_normalization: float, E_factor: float):
            sed_mean_shifted = sed_mean * E_factor ** 2
            sed_upper_shifted = sed_upper * E_factor ** 2
            sed_lower_shifted = sed_lower * E_factor ** 2
            logL: float = 0
            for i in range(bin_count):
                # shifting experimantal bin on E_factor and computing average model SED in shifted bin
                model_sed_in_shifted_bin = np.mean(
                    model_normalization
                    * model_sed[
                        np.logical_and(
                            model_E >= E_factor * E_left[i],
                            model_E <= E_factor * E_right[i],
                        )
                    ]
                )
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

        return loglikelihood

    def with_E_factor(self, E_factor: float, color: str = None) -> ExperimentalSED:
        return ExperimentalSED(
            name=self.name + f" shifted {100 * (E_factor - 1):.2f} %",
            color=color or self.color,
            E_right=self.E_right * E_factor,
            E_left=self.E_left * E_factor,
            sed_mean=self.sed_mean * E_factor ** 2,
            sed_upper=self.sed_upper * E_factor ** 2,
            sed_lower=self.sed_lower * E_factor ** 2,
        )

    def plot(self, ax: plt.Axes):
        format_axes(ax)
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


def read_seds() -> Tuple[ModelSED, ExperimentalSED, ExperimentalSED]:
    model = np.loadtxt("data/SED-KD10-Basic-0.140-Combined")
    model_sed = ModelSED(
        name="Basic hadronic model", color="#794ece", E=model[:, 0], sed=model[:, 1]
    )

    with open("data/1ES0229+200-Fermi") as fermifile:
        fermifile.readline()  # skipping first line
        fermi_data = []
        for line in fermifile:
            fermi_data.append(line.split())
    fermi_data = np.array(fermi_data, dtype=float)
    fermi_data *= 1e-6  # MeV -> TeV
    fermi_data[fermi_data < 0] = -np.inf  # -1 is used to signify 'no lower bound'
    fermi_sed = ExperimentalSED(
        name="Fermi LAT data",
        color="#dd4040",
        E_left=fermi_data[:, 0],
        E_right=fermi_data[:, 2],
        sed_mean=fermi_data[:, 3],
        sed_lower=fermi_data[:, 3] - fermi_data[:, 4],
        sed_upper=fermi_data[:, 3] + fermi_data[:, 5],
    )

    with open("data/1ES0229+200-IACT") as iactfile:
        iactfile.readline()  # skipping first line
        iact_data = []
        for line in iactfile:
            iact_data.append(line.split())
    iact_data = np.array(iact_data, dtype=float)
    E_mean_bin = iact_data[:, 1]

    def dnde2sed(dnde):
        return dnde * np.square(E_mean_bin)

    sed_mean = dnde2sed(iact_data[:, 4])
    iact_sed = ExperimentalSED(
        name="IACT data",
        color="#4ece7d",
        E_left=iact_data[:, 2],
        E_right=iact_data[:, 3],
        sed_mean=sed_mean,
        sed_upper=sed_mean + dnde2sed(iact_data[:, 5]),
        sed_lower=sed_mean + dnde2sed(iact_data[:, 6]),
    )

    return model_sed, fermi_sed, iact_sed


if __name__ == "__main__":
    read_seds()
