"""This module fits model A via cross validation for Ridge regression."""

from collections import OrderedDict
from itertools import product

import numpy as np
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.utils._joblib import Parallel, delayed

import utils


# -------------------------------------------------------------------------
# region Falsely assumed model A (without unobserved confounding)
# -------------------------------------------------------------------------
class ModelA:
    """The falsely assumed model A."""

    def __init__(self, g_noy):
        """Initialize a falsely assumed model.

        Args:
            g_noy: The causal graph without the label vertex y.
        """
        self.g_noy = g_noy
        self.inputs = None
        self.targets = None
        self.columns = None
        self.grid = None
        self.model = None
        self.set_inputs_targets()

    def set_inputs_targets(self):
        """Get parent child relations needed to fit the overall model."""
        inputs = OrderedDict((v, i) for i, v in
                             enumerate(self.g_noy.non_leafs()))
        targets = OrderedDict((v, i) for i, v in
                              enumerate(self.g_noy.non_roots()))
        columns = OrderedDict(((v, [inputs[pa] for pa in self.g_noy.parents(v)])
                               for v in targets.keys()))
        self.inputs = inputs
        self.targets = targets
        self.columns = columns

    def fit(self, data, config, verbose=0):
        """Fit entire causal graph with cross validation for hyperparameter
        selection.

        Args:
            data: The data for the graph.
            config: Configuration dictionary.
            verbose: Verbosity of logging.

        Returns:
            grid: the fitted gridsearch results.
            x: The input data of the model fit.
            y: The targets of the model fit.
            residuals: The residuals of the fit.
        """
        cv = config['cv']
        n_jobs = config['n_jobs']
        degrees = config['poly_degrees']
        intercept = config['intercept']
        alphas = config['alphas']
        alphas = np.logspace(*alphas)

        input_names = list(self.inputs.keys())
        target_names = list(self.targets.keys())
        degree_grid = []
        for degs in product(degrees, repeat=len(self.targets)):
            deg_dict = OrderedDict(((target_names[i], deg)
                                    for i, deg in enumerate(degs)))
            degree_grid.append(deg_dict)

        estim = MultipleRidge(columns=self.columns, intercept=intercept)
        param_grid = {
            'alpha': alphas,
            'degrees': degree_grid
        }

        grid = GridSearchCV(estim,
                            param_grid,
                            iid=False,
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose, pre_dispatch=1)

        x = utils.data_to_tensor(data, input_names, numpy=True)
        y = utils.data_to_tensor(data, target_names, numpy=True)
        self.grid = grid.fit(x, y)
        self.model = grid.best_estimator_

        yhat = self.from_vec(self.model.predict(x), y.shape[0])
        residuals = y - yhat
        feat = self.model.transform(x)
        n, n_targets = y.shape
        phi = np.zeros((n, n_targets, feat.shape[1]))
        for i in range(n_targets):
            phi[:, i, :] = feat[i * n:(i + 1) * n, :]

        return yhat, phi, residuals

    @property
    def alpha(self):
        """Get the regularization parameter of the trained model."""
        return self.model.alpha

    @property
    def powers(self):
        """Get the powers of polynomials used in the feature transformation."""
        return [trafo[1].named_steps['poly'].powers_
                for trafo in self.model.feature_pipe_.transformer_list]

    @property
    def weights(self):
        """Return the weight vector of the regression."""
        return self.model.regressor_.coef_

    def individual_weights(self):
        """Get a dictionary of the individual weights for each target."""
        weights = self.weights
        feature_dims = self.feature_dims()
        indiv_weights = {}
        i1 = 0
        for i, v in enumerate(self.targets.keys()):
            i2 = i1 + feature_dims[i]
            indiv_weights[v] = weights[i1:i2]
            i1 = i2
        return indiv_weights

    def feature_dims(self):
        """Get the dimensions of the features for each target."""
        return [trafo[1].named_steps['poly'].n_output_features_
                for trafo in self.model.feature_pipe_.transformer_list]

    @property
    def best_parameters(self):
        """The parameters of the best model from the gridsearch."""
        if self.model is None:
            raise RuntimeError("Must fit models first.")
        return self.grid.best_params_

    @staticmethod
    def from_vec(x, n):
        """Unstack a (n*d,) vector into a (n, d) matrix, by chunks of n."""
        if len(x.shape) != 1:
            raise RuntimeError("Expected 1D np.ndarray.")
        nd = len(x)
        if nd % n != 0:
            raise RuntimeError(f"Length of x {len(x)} not divisible by {n}.")
        d = nd // n
        y = np.zeros((n, d))
        for i in range(d):
            y[:, i] = x[i * n:(i + 1) * n]
        return y
# endregion


# -------------------------------------------------------------------------
# region Perform multiple independent ridge regressions as a single one
# -------------------------------------------------------------------------
class MultipleRidge(BaseEstimator, RegressorMixin):

    def __init__(self, columns=None, degrees=None, alpha=1.0, intercept=True):
        """Initialize MultipleRidge regression.

        Args:
            columns: A iterable of columns (represented as list).
            degrees: A iterable of degrees (represented as lists).
            alpha: The regularization parameters alpha.
            intercept: Whether to use an intercept in the fit.
        """
        self.degrees = degrees
        self.alpha = alpha
        self.columns = columns
        self.intercept = intercept
        self.feature_pipe_ = None
        self.regressor_ = None

    def fit(self, x, y=None):
        if not isinstance(self.columns, OrderedDict):
            raise RuntimeError("Targets must be of type OrderedDict.")
        if not isinstance(self.degrees, OrderedDict):
            raise RuntimeError("Degrees must be of type OrderedDict.")

        trafos = []
        for v, col in self.columns.items():
            pipe = Pipeline([('get_col',
                              FunctionTransformer(self.select_column,
                                                  kw_args={'col': col},
                                                  validate=False)),
                             ('poly',
                              PolynomialFeatures(degree=self.degrees[v],
                                                 interaction_only=False,
                                                 include_bias=self.intercept))])
            trafos.append((v, pipe))
        self.feature_pipe_ = FeatureGrid(trafos)
        phi = self.feature_pipe_.fit_transform(x)
        y = utils.to_vec(y)
        self.regressor_ = Ridge(alpha=self.alpha, fit_intercept=False)
        self.regressor_.fit(phi, y)
        return self

    def transform(self, x, y=None):
        return self.feature_pipe_.transform(x)

    def predict(self, x, y=None):
        phi = self.feature_pipe_.transform(x)
        return self.regressor_.predict(phi)

    def score(self, x, y, sample_weight=None):
        yhat = self.predict(x)
        y = utils.to_vec(y)
        return r2_score(y, yhat, sample_weight=sample_weight,
                        multioutput='variance_weighted')

    @staticmethod
    def select_column(x, col=None):
        """Pick one or multiple columns from a tensor.

        Args:
            x: np.ndarray with 2 dimensions
            col: (int, np.ndarray) If int, return the chosen column (again as 2D
                array with second dimension 1). If np.ndarray use it as index in
                second dimension of x.
        Returns:
            The chosen column(s) of x.
        """
        if col is None:
            raise RuntimeError("Argument `col` is required.")
        if isinstance(col, int):
            return x[:, col].reshape(-1, 1)
        else:
            return x[:, col]
# endregion


# -------------------------------------------------------------------------
# region FeatureGrid extension of FeatureUnion
# -------------------------------------------------------------------------
class FeatureGrid(FeatureUnion):
    """Stack features from various transformers into grid."""

    def __init__(self, transformer_list, n_jobs=None,
                 transformer_weights=None):
        """Initialize the feature grid.

        Args:
            transformer_list: The list of transformers to grid.
            n_jobs: Number of parallel jobs.
        """
        super().__init__(transformer_list, n_jobs=n_jobs,
                         transformer_weights=transformer_weights)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = block_diag(*Xs)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs = block_diag(*Xs)
        return Xs
# endregion


# -------------------------------------------------------------------------
# region General helper functions copied from sklearn required for FeatureGrid
# -------------------------------------------------------------------------

# weight and fit_params are not used but it allows _fit_one_transformer,
# _transform_one and _fit_transform_one to have the same signature to
#  factorize the code in ColumnTransformer
def _fit_one_transformer(transformer, X, y, weight=None, **fit_params):
    return transformer.fit(X, y)


def _transform_one(transformer, X, y, weight, **fit_params):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(transformer, X, y, weight, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer
# endregion
