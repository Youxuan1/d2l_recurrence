from typing import Optional, Union

import matplotlib.axes
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l, torch


def use_svg_display() -> None:
    """Use .svg format to show plot in Jupyter notebook"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize: tuple[float, float]=(3.5, 2.5)) -> None:
    """Set the size of matplotlib figures"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes: matplotlib.axes.Axes,
             xlabel: str,
             ylabel: str,
             xlim: tuple[float, float],
             ylim: tuple[float, float],
             xscale: str,
             yscale: str,
             legend: Optional[list[str]]) -> None:
    """Set the axes of matplotlib"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(True)

def myPlot(X: Union[torch.tensor, np.ndarray],
           Y: Optional[Union[torch.tensor, np.ndarray]]=None,
           xlabel: Optional[str]=None,
           ylabel: Optional[str]=None,
           xlim: Optional[tuple[float, float]]=None,
           ylim: Optional[tuple[float, float]]=None,
           xscale: str='linear',
           yscale: str='linear',
           legend: Optional[list[str]]=None,
           fmts: tuple[str, ...]=('-', 'm--', 'g-.', 'r:'),
           figsize: tuple[float, float]=(3.5, 2.5),
           axes: Optional[matplotlib.axes.Axes]=None) -> None:
    """Plot data points"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    def has_one_axis(X):
        """To figure it out whether X is a scalar or 1-D list or not"""
        return (hasattr(X, 'ndim') and X.ndim == 1
                or
                isinstance(X, list) and not hasattr(X[0], '__len__'))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[] for _ in range(len(X))], X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X *= len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    d2l.plt.show()

if __name__ == "__main__":
    x = np.arange(-5, 5, .1)
    f = lambda x: 3 * x**2 - 4 * x
    myPlot(x, [f(x), 2 * x - 3], 'x', 'f(x)',
           legend=['f(x)', 'Tagent line (x=1)'])
