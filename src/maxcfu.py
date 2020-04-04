"""This module maximizes CFU within a correlation constraints."""

from typing import Text, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from scipy.optimize import minimize
from tqdm import tqdm

import utils


class MaximizeCFU:
    """Maximize CFU within correlation constraints."""

    def __init__(self,
                 model_a: Any,
                 theta: torch.Tensor,
                 config: Dict[Text, Any]) -> None:
        """Initialize multivariate cfu evaluation driver class."""
        self.model_a = model_a
        self.theta = theta
        self.config = config
        self.dim = len(self.model_a.g_noy.non_roots())

    def evaluate(self,
                 inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 yhat: torch.Tensor) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
        """Maximize counterfactual unfairness for a range of maximum possible
        correlations.

        Args:
            inputs: A tuple (phi, a, targets)
            yhat: The estimates under the original counterfactually fair
                predictor.

        Returns:
            the cfu values for each epoch and each maximum correlation value
        """
        if self.config['grid']['pmax_vals'] is None:
            pmaxvals = np.linspace(0.0, 0.99, self.config['grid']['p_steps'])
            logger.info(f"Constructed linspace of pmaxvals: {pmaxvals}")
        else:
            pmaxvals = np.array(self.config['grid']['pmax_vals'])
            self.config['grid']['p_steps'] = len(pmaxvals)
            logger.info(f"Specified pmax_vals: {pmaxvals}")
        cfus = []
        corrmats = []
        l_params = self.get_init_params()
        for i, pmax in enumerate(pmaxvals):
            logger.info(f"Maximize CFU for pmax = {pmax}...")
            maxcfu = CFUOptimizer(self.model_a, self.theta, self.config, pmax)
            if not self.config['optimize']['curriculum']:
                l_params = self.get_init_params()
            maxcfu.set_parameters(l_params)
            # run multiple epochs for first run
            if i < 2 and pmax != 0:
                epochs = 1 * self.config['optimize']['epochs']
            else:
                epochs = None
            cfu, yhat_cf = maxcfu.maximize_cfu(inputs, yhat, epochs=epochs)
            l_params = maxcfu.l_params.detach().clone()
            cfus.append(cfu)
            corrmats.append(maxcfu.tanh_scaling(l_params @
                                                torch.t(l_params)).numpy())
        return pmaxvals, np.array(cfus), np.array(corrmats)

    def get_init_params(self) -> torch.Tensor:
        """Standard initializiation for parameters.

        Returns:
            matrix l_params of parameters
        """
        l_params = torch.eye(self.dim)
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                l_params[j, i] = 0.05
        return l_params


class CFUOptimizer(nn.Module):
    """Maximize counterfactual unfairness in model A."""

    def __init__(self,
                 model_a: Any,
                 theta: torch.Tensor,
                 config: Dict[Text, Any],
                 pmax: float) -> None:
        """Initialize a maximization class for counterfactual unfairness.

        Args:
            model_a: The weights of the (falsely) assumed model A.
            theta: The counterfactually fair predictor trained in model A.
            config: Configuration dictionary.
            pmax: Maximum allowed correlation value.
        """
        super(CFUOptimizer, self).__init__()

        self.model_a = model_a
        self.g_noy = model_a.g_noy
        self.theta = theta
        self.config = config
        self.pmax = pmax
        self.dim = len(self.g_noy.non_roots())
        self.alpha = model_a.alpha
        self.wdagger = torch.tensor(model_a.weights.reshape(-1, 1)).double()
        self.stddev_init_guess = np.array([0.] * self.dim)
        self.l_params = None

    def set_parameters(self, l_params: torch.Tensor) -> None:
        """Set the parameters l, e.g., for curriculum learning.

        Args:
            l_params: The parameter matrix l.
        """
        logger.info('Overwrite the parameters l...')
        self.l_params = nn.Parameter(l_params, requires_grad=True)
        init_l = self.l_params.detach().clone()
        init_corrmat = self.tanh_scaling(init_l @ torch.t(init_l))
        logger.info(f"Initial correlation matrix:\n{init_corrmat}")

    def forward(self,
                phi: torch.Tensor,
                a: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for finding actual and counterfactual predicitons.

        n is the number of examples in the data set
        m is the dimensionality of inputs x
        d is the dimensionality of the features Phi
        k is the dimensionality of the protected attributes a

        Args:
            phi: The feature matrices 3 dimensional (n, m, d) (torch.tensor)
            a: The protected attributes 2 dimensional (n, k) (torch.tensor)
            targets: The feature values 2 dimensional (n, m) (torch.tensor)
        """
        corrmat = torch.matmul(self.l_params, torch.t(self.l_params))
        n_original = self.config['n_original']

        # -----------------------------------------------------------------
        # FIND STANDARD DEVIATIONS
        # try:
        tmp_corrmat = corrmat.detach().clone()

        # -----------------------------------------------------------------
        # ALTERNATE OPTIMIZATION OF STDDEVS AND PARAMETERS
        def func(optarg):
            tmp_stddevs = np.exp(optarg)
            tmp_stdmat = torch.diag(torch.tensor(tmp_stddevs))
            tmp_sigma = tmp_stdmat @ self.tanh_scaling(tmp_corrmat) @\
                tmp_stdmat
            _, tmp_eps = utils.weighted_ridge(phi, targets, tmp_sigma,
                                              self.alpha,
                                              n_original=n_original)
            tmp_loss = (torch.transpose(tmp_eps, 1, 2) @
                        torch.inverse(tmp_sigma) @ tmp_eps).sum(dim=0)
            tmp_loss = tmp_loss.numpy()[0, 0]
            tmp_loss += 2. * targets.shape[0] * np.log(np.prod(tmp_stddevs))
            return tmp_loss

        res = minimize(func,
                       self.stddev_init_guess,
                       method='Powell',
                       options={
                           'disp': False,
                           'direc': np.eye(self.dim)
                       })
        if not res.success:
            logger.warn("Optimization for variance did not converge")

        # -----------------------------------------------------------------
        # CONSTRUCT FINAL COVARIANCE MATRIX
        self.stddev_init_guess = res.x
        stddevs = np.exp(res.x)
        stdmat = torch.diag(torch.tensor(stddevs))
        corrmat = self.tanh_scaling(corrmat)
        sigma = stdmat @ corrmat @ stdmat

        # -----------------------------------------------------------------
        # FIT MODEL B and COMPUTE RESIDUALS
        wstar, eps = utils.weighted_ridge(phi, targets, sigma, self.alpha,
                                          n_original=n_original)

        # -----------------------------------------------------------------
        # COMPUTE THE COUNTERFACTUALS IN MODEL B
        data_cf, x_cf, phi_cf = utils.counterfactual_data(self.model_a,
                                                          wstar, eps, a)

        # -----------------------------------------------------------------
        # COMPUTE FALSE COUNTERFACTUAL RESIDUALS IN MODEL A
        # (n,m,1) - (n,m,d) times (d,1) --> (n,m,1)
        vareps_cf = torch.squeeze(x_cf[:, :, None] -
                                  torch.matmul(phi_cf, self.wdagger))

        # -----------------------------------------------------------------
        # COMPUTE YHAT_CF
        yhat_cf = utils.torch_predict(self.theta, vareps_cf)
        # except RuntimeError as e:
        #     logger.warn("Couldn't fit, return nothing")
        #     logger.warn(e)
        #     logger.warn(e.args)
        #     yhat_cf = None

        return yhat_cf

    def maximize_cfu(self,
                     inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     yhat: torch.Tensor,
                     epochs: Optional[int] = None) -> Tuple[np.ndarray,
                                                            np.ndarray]:
        """Train parameters to maximize CFU.

        Args:
            inputs: A tuple (phi, a, targets)
            yhat: The estimates under the original counterfactually fair
                predictor.
            epochs: Override the number of epochs externally, e.g.,
                for first step curriculum learning

        Returns:
            the cfu values after each epoch
        """
        config = self.config

        # epochs
        epochs = config['optimize']['epochs'] if epochs is None else epochs

        # batchsize
        batchsize = config['optimize']['batchsize']
        if batchsize < 1:
            batchsize = yhat.shape[0]

        # optimizer
        optimizer = config['optimize']['optimizer']

        lr = config['optimize']['lr']
        if config['optimize']['scale_lr_by_pmax']:
            try:
                lr = min(abs(lr / self.pmax), config['optimize']['lr_maximum'])
            except ZeroDivisionError:
                lr = config['optimize']['lr_maximum']

        if optimizer == 'sgd':
            opt = torch.optim.SGD(self.parameters(),
                                  lr=lr,
                                  momentum=config['optimize']['momentum'],
                                  nesterov=config['optimize']['nesterov'])
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=config['optimize']['lr_step_size'],
                gamma=config['optimize']['lr_gamma'])
            logger.info("Using SGD...")
        elif optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            logger.info("Using Adam...")
        else:
            raise RuntimeError(f"Unkown optimizer {optimizer}.")

        # loss function
        loss_function = config['loss']
        if loss_function == 'mse':
            loss_func = torch.nn.MSELoss()
            logger.info("Using MSE loss...")
        else:
            raise RuntimeError(f"Unkown loss {loss_function}.")

        # collect results and optimize
        n_samples = yhat.shape[0]
        cfu_history = np.zeros(epochs)
        yhat_cfs = np.nan
        update_count = 0
        l_prev = self.l_params.detach().clone()
        cor_old = self.tanh_scaling(l_prev @ torch.t(l_prev))
        nochange = False if self.pmax != 0 else True
        for epoch in tqdm(range(epochs)):
            # Shuffle training data
            p = torch.randperm(n_samples).long()
            inputsp = tuple([inp[p] for inp in inputs])
            yhatp = yhat[p]

            for i1 in range(0, n_samples, batchsize):
                # Extract a batch
                i2 = min(i1 + batchsize, n_samples)
                inputsi = tuple([inp[i1:i2] for inp in inputsp])
                yhati = yhatp[i1:i2]

                # Reset gradients
                opt.zero_grad()
                # Forward pass
                yhat_cf = self(*inputsi)

                if yhat_cf is not None:
                    # maximize cfu = minimize negative squared difference
                    loss = -loss_func(yhat_cf, yhati)
                    # Backward pass
                    loss.backward()
                    # Parameter update
                    opt.step()

                    update_count += 1
                    if update_count % 5 == 4:
                        l_cur = self.l_params.detach().clone()
                        cor_new = self.tanh_scaling(l_cur @ torch.t(l_cur))
                        logger.info("Has correlation matrix changed...?")
                        if torch.allclose(cor_old, cor_new, rtol=0., atol=1e-5):
                            nochange = True
                            logger.info("No change. Wrap up.")
                            break
                        else:
                            logger.info("Still change.")
                            logger.info(f"correlation matrix:\n{cor_new}")

            if optimizer == 'sgd':
                scheduler.step(epoch)
            with torch.no_grad():
                logger.info(f"validation on all data...")
                yhat_all = self(*inputs)
                logger.info(f"done.")
                if yhat_all is not None:
                    loss = - loss_func(yhat_all, yhat)
                    cfu_history[epoch] = -float(loss.detach().clone().numpy())
                else:
                    cfu_history[epoch] = np.nan
            if nochange:
                break
        return cfu_history, np.array(yhat_cfs)

    def tanh_scaling(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pmax * torch.tanh(x)
        mask = torch.eye(x.shape[0])
        return mask + (1. - mask) * x
