"""This module evaluates CFU on a grid of correlation matrices."""

from typing import Text, Dict, Any, Tuple, Sequence, Union

import numpy as np
import torch
from logzero import logger
from scipy.optimize import minimize
from scipy.special import comb
from tqdm import tqdm

import utils


class GridCFU:
    """Compute counterfactual unfairness in model B for a parameter grid."""

    def __init__(self,
                 model_a: Any,
                 theta: torch.Tensor,
                 config: Dict[Text, Any]):
        """Initialize brute force grid CFU evaluation."""
        self.model_a = model_a
        self.g_noy = model_a.g_noy
        self.dim = len(self.g_noy.non_roots())
        self.n_param = int(comb(self.dim, 2))
        self.theta = theta
        self.config = config
        self.alpha = model_a.alpha
        self.wdagger = torch.tensor(model_a.weights.reshape(-1, 1)).double()
        self.stddev_init_guess = np.array([0.] * self.dim)

    def evaluate(self,
                 inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 yhat: torch.Tensor) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
        """Evaluate counterfactual unfairness for grid of correlation matrices.

        Args:
            inputs: A tuple of inputs (phi, a, x) (torch.Tensor s) where
                phi: The feature matrices 3 dimensional (n, m, d)
                a: The protected attributes 2 dimensional (n, k)
                x: The feature variable values 2 dimensional (n, m)
            yhat: The estimates under the original counterfactually fair
                predictor.
        Returns:
            pvals, cfu: The values for p and the corresponding counterfactual
                unfairness
        """
        logger.info(f"Setup correlation matrices...")
        grid = self._get_grid()
        logger.info(f"Constructed {len(grid)} correlation matrices: {grid}")
        cfu = np.zeros(len(grid))
        corrmats = np.array(np.nan)
        logger.info(f"Compute {len(grid)} CFUs...")
        for i, ps in enumerate(tqdm(grid)):
            corrmat = self._get_corrmat(ps)
            logger.info(f"corrmat: {ps}...")
            yhat_cf = self._predict(*inputs, corrmat)
            if yhat_cf is not None:
                cfu[i] = np.mean((yhat_cf.numpy() - np.array(yhat))**2)
            else:
                cfu[i] = np.nan
            logger.info(f"cfu: {cfu[i]}")
        return grid, cfu, corrmats

    def _get_grid(self) -> np.ndarray:
        """Get the list of pairwise correlations to run."""
        grid_type = self.config['grid']
        if self.config['pmax_vals'] is None:
            pvals = np.linspace(0.0, 0.99, self.config['p_steps'])
            logger.info(f"Constructed linspace of pvals: {pvals}")
        else:
            pvals = np.array(self.config['pmax_vals'])
            self.config['p_steps'] = len(pvals)
            logger.info(f"Specified pvals: {pvals}")
        if grid_type == 'full_grid':
            logger.info(f"Construct full grid...")
            grid = np.array(np.meshgrid(*([pvals] * self.n_param)))
            grid = grid.T.reshape(-1, self.dim)
        elif "extremes" in grid_type:
            signature = grid_type.split("_")[1:]
            assert len(signature) == self.n_param,\
                f"incorrect signature in {grid_type}, need {self.n_param}"
            logger.info(f"Construct grid with only extremes: {signature}...")
            grid = np.stack([pvals] * self.n_param).T
            for i, sign in enumerate(signature):
                if sign not in 'pP+':
                    grid[:, i] *= - 1.
        else:
            raise RuntimeError(f"Unknown grid type {grid_type}.")
        return grid

    def _get_corrmat(self,
                     ps: Union[Sequence[float], np.ndarray]) -> torch.Tensor:
        """Construct correlation matrix from list of values."""
        corrmat = torch.eye(self.dim)
        k = 0
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                corrmat[i, j] = ps[k]
                corrmat[j, i] = ps[k]
                k += 1
        return corrmat

    def _predict(self,
                 phi: torch.Tensor,
                 a: torch.Tensor,
                 targets: torch.Tensor,
                 corrmat: torch.Tensor) -> torch.Tensor:
        """Forward pass for finding actual and counterfactual predicitons.

        Args:
            phi: The feature matrices 3 dimensional (n, m, d) (torch.tensor)
            a: The protected attributes 2 dimensional (n, k) (torch.tensor)
            targets: The feature values 2 dimensional (n, m) (torch.tensor)
            corrmat: The correlation matrix (torch.tensor)
        """

        optim_method = self.config['stddev_optim']
        if optim_method == 'powell':
            method = 'Powell'
            options = {
                'disp': True,
                'direc': np.eye(self.dim),
                'xtol': 0.001,
                'ftol': 0.001,
            }
        else:
            method = 'Nelder-Mead'
            options = {
                'disp': True,
            }

        # -----------------------------------------------------------------
        # FIND STANDARD DEVIATIONS
        try:
            logger.info(f"Use {optim_method} for sigmas...")

            def func(optarg: np.ndarray) -> float:
                stddevs = np.exp(optarg)
                stdmat = torch.diag(torch.tensor(stddevs))
                sigma = stdmat @ corrmat @ stdmat
                _, eps = utils.weighted_ridge(phi, targets, sigma, self.alpha)
                loss = (torch.transpose(eps, 1, 2) @ torch.inverse(sigma)
                        @ eps).sum(dim=0).numpy()[0, 0]
                loss += 2. * targets.shape[0] * np.log(np.prod(stddevs))
                return loss

            res = minimize(func, self.stddev_init_guess, method=method,
                           options=options)
            if not res.success:
                logger.warn("Optimization for stddevs did not converge")

            self.stddev_init_guess = res.x
            stddevs = np.exp(res.x)
            stdmat = torch.diag(torch.tensor(stddevs))
            sigma = stdmat @ corrmat @ stdmat
            logger.info(f"found stddev: {stddevs}...")

            # -----------------------------------------------------------------
            # FIT MODEL B and COMPUTE RESIDUALS
            logger.info("Final fit...")
            wstar, eps = utils.weighted_ridge(phi, targets, sigma, self.alpha)

            # -----------------------------------------------------------------
            # COMPUTE THE COUNTERFACTUALS IN MODEL B
            logger.info("Counterfactual data...")
            data_cf, x_cf, phi_cf = utils.counterfactual_data(self.model_a,
                                                              wstar, eps, a)

            # -----------------------------------------------------------------
            # COMPUTE FALSE COUNTERFACTUAL RESIDUALS IN MODEL A
            # (n,m,1) - (n,m,d) times (d,1) --> (n,m,1)
            logger.info("False counterfactual residuals...")
            vareps_cf = torch.squeeze(x_cf[:, :, None] -
                                      torch.matmul(phi_cf, self.wdagger))

            # -----------------------------------------------------------------
            # COMPUTE YHAT_CF
            logger.info("Counterfactual predictions...")
            yhat_cf = utils.torch_predict(self.theta, vareps_cf)
        except RuntimeError as e:
            logger.warn(f"{type(e)}: {e}\n{e.args}")
            logger.warn("Couldn't fit, return nothing")
            yhat_cf = None

        return yhat_cf
