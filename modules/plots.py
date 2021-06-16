import matplotlib.pyplot as plt

from typing import List

from .models.base import ModelSED
from .experiment import Object


def plot_objects_with_model(objects: List[Object], models: List[ModelSED]):
    if len(objects) != len(models):
        raise ValueError(f"objects and models must be lists of the same length")

    n_plots = len(objects)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, sharex=True, figsize=(10, n_plots*3))

    for ax, obj, model in zip(axes, objects, models):
        obj.plot(ax)
        # E_center = 0.5 * (obj.E_min + obj.E_max)
        # E_range = obj.E_max - E_center
        model.plot(ax, E_min=obj.E_min, E_max=obj.E_max)
        ax.legend()

    plt.show()
    return fig, axes
