import numpy as np
from scipy import stats

from matplotlib import cm
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import List, Callable, Optional, Tuple
from matplotlib.pyplot import Axes


@dataclass
class DataPoint(ABC):
    x: float

    @property
    @abstractmethod
    def mean(self) -> float:
        pass

    @property
    @abstractmethod
    def std(self) -> float:
        pass

    @abstractmethod
    def plot_point(self, ax: Axes, col=None):
        pass

    @abstractmethod
    def plot_pdf(self, ax: Axes, col=None):
        pass

    @abstractmethod
    def logpdf(self, x) -> float:
        pass


@dataclass
class NormalDataPoint(DataPoint):
    mu: float
    sigma: float

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        return self.sigma

    def logpdf(self, x) -> float:
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)

    def plot_point(self, ax: Axes, col=None):
        ax.errorbar(self.x, self.mu, yerr=self.sigma, marker="o", color=col, capsize=3)

    def plot_pdf(self, ax: Axes, col=None):
        pdf_x = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma)
        ax.plot(pdf_x, stats.norm.pdf(pdf_x, loc=self.mu, scale=self.sigma), color=col)
        ax.axvline(self.mu, color=col)
        for sign in (-1, 1):
            ax.axvline(self.mu + sign * self.sigma, color=col, linestyle="--")


@dataclass
class UpperLimitDataPoint(DataPoint):
    top_edge: float
    slope_hw: float

    @property
    def mean(self) -> float:
        return self.top_edge

    @property
    def std(self) -> float:
        return np.inf

    def logpdf(self, x) -> float:
        res = np.zeros_like(x)
        above_top = x > self.top_edge
        res[above_top] = stats.norm.logpdf(x[above_top], loc=self.top_edge, scale=self.slope_hw)
        res[np.logical_not(above_top)] = stats.norm.logpdf(self.top_edge, loc=self.top_edge, scale=self.slope_hw)
        return res

    def plot_point(self, ax: Axes, col=None):
        UPLIM_CAPSIZE = 0.5

        ax.errorbar(
            self.x,
            self.top_edge + self.slope_hw,
            yerr=self.slope_hw,
            xerr=UPLIM_CAPSIZE / 2,
            uplims=True,
            color=col,
        )

    def plot_pdf(self, ax: Axes, col=None):
        LOWER_BOUND = -10  # quick hack

        pdf_x = np.linspace(LOWER_BOUND, self.top_edge + 3 * self.slope_hw)
        pdf_y = stats.norm.pdf(pdf_x, loc=self.top_edge, scale=self.slope_hw)
        pdf_y[pdf_x < self.top_edge] = pdf_y.max()
        ax.plot(pdf_x, pdf_y, color=col)
        ax.axvline(self.top_edge + self.slope_hw, color=col, linestyle="--")



@dataclass
class UniformDataPoint(DataPoint):
    mid: float
    hw: float

    @property
    def mean(self) -> float:
        return self.mid

    @property
    def std(self) -> float:
        return 2 * self.hw / np.sqrt(12)

    def logpdf(self, x) -> float:
        return np.log((np.abs(x - self.mid) < self.hw) / (2 * self.hw))

    def plot_point(self, ax: Axes, col=None):
        ax.errorbar(
            self.x,
            self.mid,
            yerr=self.hw,
            capsize=6,
            marker=None,
            color=col,
        )

    def plot_pdf(self, ax: Axes, col=None):
        pdf_x = np.linspace(self.mid - 1.1 * self.hw, self.mid + 1.1 * self.hw, 100)
        pdf_y = np.zeros_like(pdf_x)
        pdf_y[np.abs(pdf_x - self.mid) < self.hw] = 1 / (2 * self.hw)
        ax.plot(pdf_x, pdf_y, color=col)


@dataclass
class DataSet:
    points: List[DataPoint]

    cmap: Optional[Callable[[float], "color"]] = cm.get_cmap("cool")  # type: ignore

    def __post_init__(self):
        self.points = sorted(self.points, key=lambda pt: pt.x)
        if isinstance(self.cmap, str):
            self.cmap = cm.get_cmap(self.cmap)

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def x(self) -> List[float]:
        return [pt.x for pt in self.points]

    @property
    def x_bounds(self) -> Tuple[float, float]:
        pt_xs = self.x
        return min(pt_xs), max(pt_xs)

    def plot_points(self, ax):
        for i, pt in enumerate(self.points):
            pt.plot_point(ax, col=self.cmap(i / self.size))

    def plot_point_pdf(self, ax, i):
        self.points[i].plot_pdf(ax, col=self.cmap(i / self.size))
