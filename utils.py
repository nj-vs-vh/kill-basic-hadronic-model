import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from nptyping import NDArray


def format_axes(ax: plt.Axes):
    ax.set_xlabel("E, $TeV$")
    ax.set_xscale("log")
    ax.set_ylabel("SED, $TeV \, cm^{-2}, \, s^{-1}$")
    ax.set_yscale("log")


def _evaluate_loglikes(sample: NDArray, loglike: Callable[[NDArray], float], progress: bool = False) -> NDArray:
    return np.array([loglike(p) for p in sample])


def max_loglike(sample: NDArray, loglike: Callable[[NDArray], float], progress: bool = False) -> float:
    return np.max(_evaluate_loglikes(sample, loglike, progress))


def max_loglike_point(sample: NDArray, loglike: Callable[[NDArray], float], progress: bool = False) -> NDArray:
    i_max = np.argmax(_evaluate_loglikes(sample, loglike, progress))
    return sample[i_max, :]


def bic(k: int, n: int, sample: NDArray, loglike: Callable[[NDArray], float]) -> float:
    return k * np.log(n) - 2 * max_loglike(sample, loglike)
