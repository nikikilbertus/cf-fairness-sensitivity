#!/usr/bin/env python3
"""The main script evaluating counterfactual unfairness given a config file."""

import argparse
import json
import os

import logzero
import numpy as np
import torch
from logzero import logger
from sklearn.metrics import r2_score

import data as data_loader
import gridcfu
import maxcfu
import models
import plotters
import utils


# -------------------------------------------------------------------------
# Read configuration
# -------------------------------------------------------------------------
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class)

parser.add_argument('--path',
                    help='The path to the config file.',
                    type=str,
                    default='experiments/config_lawschool.json')

args = parser.parse_args()
config_path = os.path.abspath(args.path)

# -------------------------------------------------------------------------
# Load config file
# -------------------------------------------------------------------------
logger.info(f"Read config file from {config_path}...")
with open(config_path, 'r') as f:
    config = json.load(f)

# -------------------------------------------------------------------------
# Run on GPU or CPU
# -------------------------------------------------------------------------
if config["gpu"]:
    if torch.cuda.is_available():
        logger.info("Running on GPU...")
        device = "cuda:0"
    else:
        raise RuntimeError("Requested GPU, but can not find one.")
else:
    logger.info("Running on CPU...")
    device = "cpu"

# -------------------------------------------------------------------------
# Setup up directory for results
# -------------------------------------------------------------------------
result_dir, fig_dir = utils.setup_directories(config)

# -------------------------------------------------------------------------
# Set seeds and default values
# -------------------------------------------------------------------------
debug = config['debug']
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

# -------------------------------------------------------------------------
# Setup logger
# -------------------------------------------------------------------------
logger.info(f"Add file handle to logger...")
logzero.logfile(os.path.join(result_dir, 'logs.log'))

# -------------------------------------------------------------------------
# Construct graph from config file
# -------------------------------------------------------------------------
logger.info("Construct graph...")
g, g_noy = utils.construct_graph(config)

# -------------------------------------------------------------------------
# Load data according to config
# -------------------------------------------------------------------------
data_type = config['data']['type']
logger.info(f"Load {data_type} data; A: {config['data']['protected']} ...")
data = data_loader.get_data(data_type, config['data'], graph=g)
a, y = data['A'], data['Y']
config['data']['samples'] = len(y)
config['max_cfu']['n_original'] = len(y)

if debug > 0:
    logger.info("Create and save scatter plot of features...")
    plotters.plot_scatter_matrix(data, g, fig_dir, save=True)
    logger.info("Create conditional histograms...")
    for target in g.vertices():
        if target != 'A':
            plotters.plot_conditional_histograms(data, target, 'A', fig_dir)

# -------------------------------------------------------------------------
# Classification or regression problem?
# -------------------------------------------------------------------------
if utils.is_binary(y):
    config['cf_fair']['type'] = 'classification'
else:
    config['cf_fair']['type'] = 'regression'

# -------------------------------------------------------------------------
# Fit assumed model A via cross validation
# -------------------------------------------------------------------------
logger.info("Fit model A via CV, compute phi and residuals...")
model_a = models.ModelA(g_noy)
_, phi, vareps = model_a.fit(data, config['cf_fair'])
logger.info(f"Best parameters: {model_a.best_parameters}")

# Refit as torch with weighted ridge
logger.info("Refit model analytically...")
targets = utils.data_to_tensor(data, list(model_a.targets.keys()), numpy=True)
phi, a, targets = [torch.tensor(_) for _ in (phi, a, targets)]
sigma = torch.diag(torch.tensor(vareps.std(axis=0)**2))
wdagger, vareps = utils.weighted_ridge(phi, targets, sigma, model_a.alpha)

model_a.model.regressor_.coef_ = wdagger.clone().numpy().squeeze()
vareps = vareps.clone().numpy().squeeze()

if debug > 0:
    logger.info("Plot conditional histograms...")
    for i, target in enumerate(g_noy.vertices()[1:]):
        plotters.plot_conditional_histograms({f'resid_{target}': vareps[:, i],
                                              'A': a},
                                             f'resid_{target}',
                                             'A',
                                             fig_dir)

# -------------------------------------------------------------------------
# Compute the counterfactually fair predictor in model A via CV
# -------------------------------------------------------------------------
logger.info("Compute counterfactually fair predictor assuming model A...")
theta, yhat_cff, resid_cff = utils.simple_cv_fit_numpy(vareps,
                                                       y,
                                                       config['cf_fair'])

if debug > 0:
    plotters.plot_conditional_histograms({'resid_Y': resid_cff, 'A': a},
                                         'resid_Y', 'A', fig_dir)

# -------------------------------------------------------------------------
# Train baslines
# -------------------------------------------------------------------------
logger.info("Train baselines...")
logger.info("Train fully unconstrained baseline...")
x = utils.data_to_tensor(data, g_noy.vertices(), numpy=True)
theta_uc, yhat_uc, eps_uc = utils.simple_cv_fit_numpy(x,
                                                      y,
                                                      config['cf_fair'])
cfu_uc = utils.compute_cfu(yhat_cff, yhat_uc, config['cf_fair'])

logger.info("Train blind unconstrained baseline...")
x = utils.data_to_tensor(data, g_noy.non_roots(), numpy=True)
theta_buc, yhat_buc, eps_buc = utils.simple_cv_fit_numpy(x,
                                                         y,
                                                         config['cf_fair'])
cfu_buc = utils.compute_cfu(yhat_cff, yhat_buc, config['cf_fair'])

logger.info(f"R2 of counterfactually fair: {r2_score(y, yhat_cff)}")
logger.info(f"R2 of blind: {r2_score(y, yhat_buc)}")
logger.info(f"R2 of unconstrained: {r2_score(y, yhat_uc)}")
logger.info(f"CFU for unconstrained model: {cfu_uc}")
logger.info(f"CFU for blind unconstrained model: {cfu_buc}")

# -------------------------------------------------------------------------
# MAIN PART: Maximize counterfactual unfairness
# -------------------------------------------------------------------------
cfu_config = config['max_cfu']
inputs = (phi, a, targets)

if cfu_config["type"] == "grid":
    logger.info("Use grid approach...")
    cf = gridcfu.GridCFU(model_a, theta.best_estimator_, cfu_config)
    logger.info("Compute CFU for different correlation matrices...")
    pvals, cfu, corrmats = cf.evaluate(inputs, torch.tensor(yhat_cff))
elif cfu_config["type"] == "optimize":
    logger.info("Use optimization approach...")
    cf = maxcfu.MaximizeCFU(model_a, theta.best_estimator_, cfu_config)
    logger.info("Maximize CFU for different pmax...")
    pvals, cfu, corrmats = cf.evaluate(inputs, torch.tensor(yhat_cff))
else:
    raise RuntimeError(f"Unknown mode for CUF {cfu_config['type']}")

# -------------------------------------------------------------------------
# Output results
# -------------------------------------------------------------------------
logger.info("Write out results...")
datadump = os.path.join(result_dir, "datadump")
np.savez(datadump,
         cfus=cfu,
         cfu_uc=cfu_uc,
         cfu_buc=cfu_buc,
         pvals=pvals,
         corrmats=corrmats,
         yhat_cff=yhat_cff,
         yhat_uc=yhat_uc,
         yhat_buc=yhat_buc,
         wdagger=wdagger,
         theta=theta,
         theta_uc=theta_uc,
         theta_buc=theta_buc,
         data=data)

logger.info("Write out used config file...")
res_config_path = os.path.join(result_dir, 'config.json')
with open(res_config_path, 'w') as f:
    json.dump(config, f, indent=2)

# -------------------------------------------------------------------------
# Plot some results
# -------------------------------------------------------------------------
logger.info("Plot some results...")
if cfu_config["type"] == "optimize":
    plotters.plot_training_progress(pvals, cfu, fig_dir)
    cfu = cfu.max(axis=1)

cfus = np.stack([cfu,
                 np.ones_like(cfu) * cfu_uc,
                 np.ones_like(cfu) * cfu_buc]).T
plotters.plot_pvals_cfus(pvals, cfus, fig_dir, suff='compare', log=False)
plotters.plot_pvals_cfus(pvals, cfus, fig_dir, suff='compare_log', log=True)
plotters.plot_pvals_cfus(pvals, cfus[:, 0], fig_dir, suff='single', log=False)

logger.info("Finished run.")
logger.info("DONE")
