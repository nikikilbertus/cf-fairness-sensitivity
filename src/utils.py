"""This module contains various utility functions."""

import os
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from typing import Optional, Dict, Text, Any, Sequence, Union, Callable, Tuple

import numpy as np
import torch
from logzero import logger
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import graph


# -------------------------------------------------------------------------
# region Setup and initializiation
# -------------------------------------------------------------------------
def setup_directories(config: Dict[Text, Any]) -> Tuple[Text, Text]:
    """Create required directories to collect results.

    Args:
        config: The configuration dictionary.
    """
    logger.info("Setup directories for results...")
    result_dir = os.path.abspath(config['results']['result_dir'])
    if config['results']['name'] is None:
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config['data']['type'] == 'custom':
            dir_name += config['data']['custom_type']
        else:
            dir_name += '_' + config['data']['type'] + '_' + \
                        config['data']['protected']
    else:
        dir_name = config['results']['name']
    result_dir = os.path.join(result_dir, dir_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    config['results']['resolved'] = result_dir

    fig_dir = os.path.join(result_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    config['results']['figure_dir'] = fig_dir
    return result_dir, fig_dir


def construct_graph(config: Dict[Text, Any]) -> Tuple[graph.Graph, graph.Graph]:
    g = graph.Graph(config['graph'])
    g_noy = graph.Graph({k: list(tuple(v))
                         for k, v in g.graph.items() if k != 'Y'})
    g.render(path=config['results']['figure_dir'], save=True)
    return g, g_noy
# endregion


# -------------------------------------------------------------------------
# region Data handling and feature map construction
# -------------------------------------------------------------------------
def to_vec(x: np.ndarray) -> np.ndarray:
    """Stack (n, d) matrix into (n*d,) vector."""
    return x.T.ravel()


def data_to_tensor(data: Dict[Text, torch.Tensor],
                   nodes: Sequence[Text],
                   numpy: Optional[bool] = False) -> Union[torch.Tensor,
                                                           np.ndarray]:
    """Combine data from various nodes in a graph into one tensor.

    Args:
        data: The data dictionary.
        nodes: The names of the nodes to combine in the tensor (in this order).
        numpy: Whether to convert the output to a numpy array.

    Returns:
        torch.tensor/np.ndarray with the data.
    """
    if numpy:
        if isinstance(data[list(nodes)[0]], torch.Tensor):
            result = torch.cat(tuple(data[node][:, None]
                                     for node in nodes), dim=1).numpy()
        else:
            result = np.stack([data[node] for node in nodes], axis=1)
    else:
        result = torch.cat(tuple(data[node].double() for node in nodes), dim=1)
    return result


def is_binary(x: np.ndarray) -> bool:
    """Check whether a tensor is binary.

    Args:
        x: tensor
    """
    vals = np.unique(x)
    if len(vals) == 2:
        vals = sorted(vals)
        if vals[0] == 0 and vals[1] == 1:
            return True
        else:
            logger.warn(f"Is binary with values {vals}!")
            return True
    return False


def get_featuremap(powers: Sequence[Sequence[int]]) -> Callable[[torch.Tensor],
                                                                torch.Tensor]:
    """Decorator for creating polynomial feature maps.

    Args:
        powers: List of list of powers for the polynomial

    Returns:
        The feature map
    """
    def transform(x):
        """Apply polynomial feature map.

        Args:
            x: Input tensor.

        Returns:
            polynomial featuers
        """
        return torch.stack(tuple(reduce(torch.mul, [x[:, i].pow(int(p))
                                                    for i, p in enumerate(fp)])
                                 for fp in powers), dim=1)
    return transform


def construct_feature_maps(model_a: Any) -> Dict[Text, Callable[[torch.Tensor],
                                                                torch.Tensor]]:
    """Construct polynomial feature maps for torch tensors from the learned
    model.

    Args:
        model_a: The optimal fitted model A.

    Returns:
        feature_maps: A dictionary of feature maps.
    """
    powers = model_a.powers
    fm = OrderedDict()
    for target_pows, v in zip(powers, model_a.targets.keys()):
        fm[v] = get_featuremap(target_pows)
    return fm


def create_counterfactual_value(attr: torch.Tensor) -> torch.Tensor:
    """For a tensor with binary values only, swap them.

    Args:
        attr: torch.tensor in which to swap the values.
    """
    x = torch.ones_like(attr)
    vals = attr.unique()
    if len(vals) == 2:
        x[attr == vals[0]] = vals[1]
        x[attr == vals[1]] = vals[0]
    else:
        x *= attr.mean()
    return x[:, None]


def counterfactual_data(model_a: Any,
                        weights: torch.Tensor,
                        eps: torch.Tensor,
                        a: torch.Tensor) -> Tuple[Dict[Text, torch.Tensor],
                                                  torch.Tensor,
                                                  torch.Tensor]:
    """Compute the counterfactual data.

    Args:
        model_a: Optimal fitted model A.
        weights: The weights with which to compute the counterfactuals.
        eps: The residuals from which to compute the counterfactuals.
        a: The originally observed protected values.

    Returns:
        data: dictionary with the counterfactual data
    """
    data_cf = {'A': create_counterfactual_value(a)}
    n_feats = model_a.feature_dims()
    phi_cf = torch.zeros(a.shape[0], len(n_feats), sum(n_feats))
    x_cf = torch.zeros(a.shape[0], len(n_feats))
    fm = construct_feature_maps(model_a)

    i1 = 0
    for i, target in enumerate(model_a.targets.keys()):
        i2 = i1 + n_feats[i]
        inputs = model_a.g_noy.parents(target)
        inputs = data_to_tensor(data_cf, inputs, numpy=False)
        phi = fm[target](inputs)
        data_cf[target] = torch.matmul(phi, weights[i1:i2]) + eps[:, i]
        phi_cf[:, i, i1:i2] = phi
        x_cf[:, i] = torch.squeeze(data_cf[target])
        i1 = i2
    return data_cf, x_cf, phi_cf
# endregion


# -------------------------------------------------------------------------
# region Model fitting
# -------------------------------------------------------------------------
@torch.jit.script
def weighted_ridge(
        phi: torch.Tensor,
        x: torch.Tensor,
        sigma: torch.Tensor,
        alpha: float,
        n_original: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit a weighted ridge regression (operates on torch tensors).

        n is the number of examples in the data set
        m is the dimensionality of inputs x
        d is the dimensionality of the features Phi
        k is the dimensionality of the protected attributes a

    Args:
        phi: The feature matrices 3 dimensional (n, m, d)
        x: The target variable values 2 dimensional (n, m)
        sigma: The covariance matrix of the targets (d, d)
        alpha: The regularization parameter (scalar).
        n_original: The number of training examples used to find alpha.

    Returns:
        weights, residuals
    """
    d = phi.shape[2]
    reg = alpha * torch.eye(d)
    if n_original is not None:
        reg *= x.shape[0] / n_original
    # invert (m,m) --> (m,m)
    sigma_inv = torch.inverse(sigma)
    # (m,m) times (n,m,d) --> (n,m,d)
    mat = torch.matmul(sigma_inv, phi)
    # (n,d,m) times (n,m,d) --> (n,d,d)
    mat = torch.matmul(torch.transpose(phi, 1, 2), mat)
    # sum (n,d,d) along 0 --> (d,d)
    mat = torch.sum(mat, 0) + reg
    # (m,m) times (n,m,1) --> (n,m,1)
    vec = torch.matmul(sigma_inv, x[:, :, None])
    # (n,d,m) times (n,m,1) --> (n,d,1)
    vec = torch.matmul(torch.transpose(phi, 1, 2), vec)
    # sum (n,d,1) along 0 --> (d,1)
    vec = torch.sum(vec, 0)
    # weights for model B --> (d,1)
    wstar = torch.solve(vec, mat)[0]

    # compute residuals (n,m,1) - (n,m,d) times (d,1) --> (n,m,1)
    eps = x[:, :, None] - torch.matmul(phi, wstar)
    return wstar, eps


def torch_predict(model: Any, x: torch.Tensor) -> torch.Tensor:
    """Evaluate an sklearn model on torch tensor for autodiff.

    Args:
        model: The sklearn model.
        x: Inputs to the model.
    """
    w = torch.tensor(model.named_steps['regressor'].coef_)
    if model.named_steps['regressor']._estimator_type == 'classifier':
        w = torch.transpose(w, 0, 1)
    powers = model.named_steps['features'].powers_
    fm = get_featuremap(powers)
    return torch.matmul(fm(x), w)


def simple_cv_fit_numpy(x: np.ndarray,
                        y: np.ndarray,
                        config: np.ndarray,
                        verbose: Optional[int] = 0) -> Tuple[GridSearchCV,
                                                             np.ndarray,
                                                             np.ndarray]:
    """A simple cross validated ridge regression fit (operates on numpy arrays).

    Args:
        x: Input features.
        y: Labels.
        config: Configuration dictionary.
        verbose: Verbosity of logging.

    Returns:
        The cross validation results.
    """
    cv = config['cv']
    degrees = config['poly_degrees']
    n_jobs = config['n_jobs']
    intercept = config['intercept']
    alphas = config['alphas']
    alphas = np.logspace(*alphas)
    fit_type = config['type']

    parts = [('features', PolynomialFeatures(interaction_only=False,
                                             include_bias=intercept))]
    param_grid = {'features__degree': degrees}
    if fit_type == 'regression':
        logger.info("Training Ridge...")
        parts.append(('regressor', Ridge(fit_intercept=False)))
        param_grid['regressor__alpha'] = alphas
    else:
        logger.info("Training regularized Logistic...")
        parts.append(('regressor', LogisticRegression(fit_intercept=False,
                                                      solver='lbfgs')))
        param_grid['regressor__C'] = alphas

    pipe = Pipeline(parts)
    grid = GridSearchCV(pipe,
                        param_grid,
                        iid=False,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose)
    grid.fit(x, y)
    yhat = grid.predict(x)
    return grid, yhat, y - yhat


def compute_cfu(y_cf: np.ndarray,
                yhat: np.ndarray,
                config: Dict[Text, Any]) -> float:
    """Compute counterfactual unfairness between two predictions depending on
    the estimation type.

    Args:
        y_cf: Counterfactually fair labels.
        yhat: Other labels.
        config: Configuration dictionary (for type of estimation).

    Returns:
        measure of counterfactual unfairness
    """
    if config['type'] == 'regression':
        return mean_squared_error(y_cf, yhat)
    else:
        return log_loss(y_cf, yhat)
# endregion
