import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

from typing import Callable, Optional
from nptyping import NDArray


def format_axes(ax: plt.Axes):
    ax.set_xlabel("E, $TeV$")
    ax.set_xscale("log")
    ax.set_ylabel("SED, $TeV \, cm^{-2}, \, s^{-1}$")
    ax.set_yscale("log")


def _evaluate_loglikes(
    sample: NDArray, loglike: Callable[[NDArray], float], progress: bool = False
) -> NDArray:
    res = np.zeros(shape=(sample.shape[0],))
    if progress:
        sample = tqdm(sample)
    for i, point in enumerate(sample):
        res[i] = loglike(point)
    return res


def max_loglike(
    sample: NDArray, loglike: Callable[[NDArray], float], progress: bool = False
) -> float:
    return np.max(_evaluate_loglikes(sample, loglike, progress))


def max_loglike_point(
    sample: NDArray, loglike: Callable[[NDArray], float], progress: bool = False
) -> NDArray:
    i_max = np.argmax(_evaluate_loglikes(sample, loglike, progress))
    return sample[i_max, :]


def AIC(
    k: int,
    max_loglikelihood: Optional[float] = None,
    sample: Optional[NDArray] = None,
    loglike: Optional[Callable[[NDArray], float]] = None,
) -> float:
    if max_loglikelihood is None:
        max_loglikelihood = max_loglike(sample, loglike)
    return 2 * k - 2 * max_loglikelihood


def trapz_integral_func(
    f: Callable[..., float], allows_njit: bool = False, n_pts: int = 4
) -> Callable[..., float]:
    def f_integral(x_min: float, x_max: float, *params: float) -> float:
        xs = np.exp(np.linspace(np.log(x_min), np.log(x_max), n_pts))
        return np.trapz(f(xs, *params), xs)

    return f_integral if not allows_njit else njit(f_integral)


def enlarge_log_interval(left, right, pad = 0.05):
    if left < 0 or right < 0:
        return left, right
    mid = np.sqrt(left * right)
    hr = right / mid

    return mid / (hr ** (1 + pad)), mid * (hr ** (1 + pad))
