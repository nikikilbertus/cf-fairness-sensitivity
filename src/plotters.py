"""Plotting functionality"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logzero import logger

from utils import data_to_tensor


def plot_pvals_cfus(pvals, cfus, path, save=True, suff='', log=True):
    """Plot counterfactual unfairness for different values of p in the bivariate
    case.

    Args:
        pvals: The values of p
        cfus: The corresponding cfu values (array with baselines)
        path: Where to store the plot (only used if save=True)
        save: Whether to save the plot
        suff: Suffix of the figure file name
        log: Whether to plot the y axis on log scale
    """
    plt.figure()
    if log:
        plt.semilogy(pvals, cfus)
    else:
        plt.plot(pvals, cfus)
    plt.xlabel('p')
    plt.ylabel('mean squared CFU')
    plt.tight_layout()

    plt.legend(['cfu', 'unconstrained', 'blind unconstrained'])
    if save:
        if len(suff) > 0:
            suff = '_' + suff
        figname = "cfu" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()


def plot_scatter_matrix(data, graph, path, save=True, suff='', max_points=3000):
    """Plot scatter matrix of features for quick inspection."""
    columns = graph.vertices()
    df = pd.DataFrame(data_to_tensor(data, columns, numpy=True),
                      columns=columns)
    if len(df) > max_points:
        df = df.sample(n=max_points)
    figdim = len(columns) * 5
    plt.figure()
    pd.plotting.scatter_matrix(df, figsize=(figdim, figdim), diagonal='kde')
    plt.tight_layout()
    plt.title('feature scatter matrix')
    if save:
        if len(suff) > 0:
            suff = '_' + suff
        figname = "scatter" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()


def plot_conditional_histograms(data, target, condition, path, save=True,
                                suff=''):
    """Plot conditional histograms."""
    c_vals = sorted(np.unique(data[condition]))
    if len(c_vals) != 2:
        logger.warning("No conditional histrograms for non-binary condition.")
    columns = [target, condition]
    df = pd.DataFrame(data_to_tensor(data, columns, numpy=True),
                      columns=columns)
    plt.figure()
    plt.hist(df[df[condition] == c_vals[0]][target], bins=100, density=True,
             alpha=0.5, label=f'{target} | {condition}={c_vals[0]}')
    plt.hist(df[df[condition] == c_vals[1]][target], bins=100, density=True,
             alpha=0.5, label=f'{target} | {condition}={c_vals[1]}')
    plt.legend()
    plt.title(f"Distribution of {target}")
    plt.tight_layout()
    if save:
        if len(suff) > 0:
            suff = '_' + suff
        figname = f"{target}_given_{condition}{suff}.pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()


def plot_training_progress(pvals, cfus, path, save=True, suff=''):
    """Plot training curves on epoch level."""
    plt.figure()
    for cfu in cfus:
        plt.plot(cfu)
    plt.xlabel('epochs')
    plt.ylabel('CFU')
    plt.title("Training progress")
    plt.legend([f"{pmax:.2f}" for pmax in pvals], loc="center left",
               bbox_to_anchor=(1.05, 0.5))
    if save:
        if len(suff) > 0:
            suff = '_' + suff
        figname = "training" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()
