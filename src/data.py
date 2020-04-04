"""This module is used to load different datasets or synthesize data."""

import os

import numpy as np
import pandas as pd

import sem


# -------------------------------------------------------------------------
# region Load data from file
# -------------------------------------------------------------------------
def get_data(dataset, config, graph=None):
    """Retrieve a dataset.

    Args:
        dataset: The name of the data to retrieve (see config).
        config: Configuration dictionary.
        graph: For synthetic datasets, the causal graph is required.

    Returns:
        data, dictionary containing the data.
    """

    identifier_to_filename = {
        'nhs': 'nhs.npz',
        'lawschool': 'lawschool.csv',
        'custom': '',
    }

    data_path = os.path.join(os.path.abspath(config['path']),
                             identifier_to_filename[dataset])

    if dataset == 'lawschool':
        data = format_lawschool_data(data_path, config['protected'])
    elif dataset == 'custom':
        data = get_custom(graph, config['samples'], config['custom_type'])
    elif dataset == 'nhs':
        data = format_nhs_data(data_path, config['protected'])
    else:
        raise RuntimeError(f"Unknown dataset {dataset}.")

    whiten(data, config['whiten'])
    if config['samples'] is not None:
        data = subsample(data, config['samples'])

    return data
# endregion


# -------------------------------------------------------------------------
# region Preprocessing and formatting for different datasets
# -------------------------------------------------------------------------
def format_lawschool_data(data_path, protected):
    """Bring the raw law school dataset into the required format.

    For reference: race_values = [
        'Amerindian',
        'Asian',
        'Black',
        'Hispanic',
        'Mexican',
        'Other',
        'Puertorican',
        'White']

    Args:
        data_path: The path to the dataset file.
        protected: Which variable to use as the protected attribute
            (race, gender).

    Returns:
        data: A preprocessed data dictionary.
    """
    raw_data = pd.read_csv(data_path, index_col=0)
    data = {}
    if protected == 'race':
        data['A'] = raw_data['race'].apply(lambda x:
                                           2 * int(x.lower() == 'white')
                                           - 1).values
    elif protected == 'gender':
        data['A'] = 2 * (raw_data['sex'].values - 1) - 1
    else:
        raise RuntimeError(f'Unknown protected attribute {protected}.')
    data['G'] = raw_data['UGPA'].values
    data['L'] = raw_data['LSAT'].values
    data['Y'] = raw_data['ZFYA'].values
    return data


def format_nhs_data(data_path, protected):
    """Bring the raw nhs dataset into the required format.

    Args:
        data_path: The path to the dataset file.
        protected: Which column to use as the protected attribute.

    Returns:
        data: A preprocessed data dictionary.
    """
    raw_data = np.load(data_path)

    col_name_dict = {
        'race': 'A',
        'gender': 'A',
        'sex_orient': 'A',
        'disability': 'A',
        'age': 'A',
        'health': 'Y',
        'job_sat': 'J',
        'manager_sat': 'M',
        'org_sat': 'O'
    }
    to_drop = [
        'race',
        'gender',
        'sex_orient',
        'disability',
        'age']
    to_drop.remove(protected)
    df = pd.DataFrame(raw_data['data'], columns=raw_data['colnames'])
    df = df.drop(columns=to_drop).rename(index=str, columns=col_name_dict)
    data = pandas_to_dict(df)
    data['A'] = 2 * data['A'] - 3
    return data


def get_custom(graph, n_samples, type):
    """Get samples from an additive noise model with Gaussian noise and
    polynomial structural equations.

    Args:
        graph: The graph on top of which to build the generative model
            (including the target)
        n_samples: Number of samples

    Returns:
        data dict with vertex identifier as keys and torch.tensor as values
    """
    model = sem.SEM(graph)
    if type == 'binary1':
        model.attach_equation(
            'A',
            lambda n: 2 * np.random.randint(0, 2, (n, 1)) - 1
        )
        model.attach_equation(
            'G',
            lambda d: d['A'] + ((np.random.beta(2, 5, (len(d['A']), 1)) * 4 - 1) *
                                (np.ceil(d['A']) == 1).astype(float)) +
                      ((np.random.beta(9, 4, (len(d['A']), 1)) * 4 - 2.5) *
                       (np.ceil(d['A']) != 1).astype(float))
        )
        model.attach_equation(
            'L',
            lambda d: 0.2 * d['A'] + (0.4 * d['G']) ** 3 + 0.1 * d['A'] * d['G'] +
                      (np.random.beta(2, 4, (len(d['A']), 1)) - 1)**2
        )
        model.attach_equation('Y',
                              lambda d: 0.5 * d['G'] ** 2 * d['L'] +
                                        (0.7 * d['L']) ** 2 +
                                        0.3 * np.random.randn(len(d['A']), 1)
                              )
    elif type == 'binary2':
        model.attach_equation(
            'A',
            lambda n: 2 * np.random.randint(0, 2, (n, 1)) - 1
        )
        model.attach_equation(
            'G',
            lambda d: d['A'] + ((np.random.beta(2, 5, (len(d['A']), 1)) * 4 - 1) *
                                (np.ceil(d['A']) == 1).astype(float)) +
                      ((np.random.beta(9, 4, (len(d['A']), 1)) * 4 - 2.5) *
                       (np.ceil(d['A']) != 1).astype(float))
        )
        model.attach_equation(
            'L',
            lambda d: 0.2 * d['A'] + (0.4 * d['G']) ** 3 + 0.1 * d['A'] * d['G'] +
                      (np.random.beta(2, 4, (len(d['A']), 1)) - 1)**2 +
                      np.random.beta(4, 4, (len(d['A']), 1)) * 0.4 + 2
        )
        model.attach_equation('Y',
                              lambda d: 0.5 * d['G'] ** 2 * d['L'] +
                                        (0.7 * d['L']) ** 2 +
                                        0.3 * np.random.randn(len(d['A']), 1)
                              )
    data = model.sample(n_samples)
    return data
# endregion


# -------------------------------------------------------------------------
# region General helper functions
# -------------------------------------------------------------------------
def whiten(data, keys=None, conditioning=1e-8):
    """Whiten various datasets in data dictionary.

    Args:
        data: Data dictionary.
        keys: The dictionary entries to whiten. If `None`, whiten all.
        conditioning: Added to the denominator to avoid divison by zero.
    """
    if keys is None:
        keys = data.keys()
    for key in keys:
        mu = data[key].mean(axis=0)
        std = data[key].std(axis=0)
        data[key] = (data[key] - mu) / (std + conditioning)


def subsample(data, n_samples):
    """Subsample a fixed number of examples uniformly at random.

    Args:
        data: Data dicitonary which to subsample from.
        n_samples: Number of samples to take.

    Returns:
        subsampled data dictionary
    """
    n_samples = min(n_samples, len(data['Y']))
    idx = np.random.choice(len(data['Y']), n_samples, replace=False)
    for key in data:
        data[key] = data[key][idx]
    return data


def pandas_to_dict(df):
    """Convert a pandas dataframe into a dictionary of torch tensors."""
    data = {}
    for col in df.columns:
        data[col] = df[col].values
    return data
# endregion
