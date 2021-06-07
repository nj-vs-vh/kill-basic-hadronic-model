import matplotlib.pyplot as plt


def format_axes(ax: plt.Axes):
    ax.set_xlabel("E, $TeV$")
    ax.set_xscale("log")
    ax.set_ylabel("SED, $TeV \, cm^{-2}, \, s^{-1}$")
    ax.set_yscale("log")
