import numpy as np
import pandas as pd
import os 
import pickle 
from scipy.stats import rv_discrete
from typing import Optional

#DATA_PATH = '/data'
#DATA_PATH = '/data2'
DATA_PATH = '/data/scan'

def save_obj(obj, name): 
    filename = os.path.join(DATA_PATH, f'{name}_obj.pkl')
    with open(filename, 'wb') as fout: 
        pickle.dump(obj, fout)

def load_obj(name): 
    filename = os.path.join(DATA_PATH, f"{name}_obj.pkl")
    if not os.path.exists(filename):
        return None
    with open(filename, 'rb') as fin:
        res = pickle.load(fin)
    return res

def save_df(df, name): 
    filename = os.path.join(DATA_PATH, f'{name}_df.pkl') 
    df.to_pickle(filename)

def load_df(name): 
    filename = os.path.join(DATA_PATH, f'{name}_df.pkl') 
    return pd.read_pickle(filename)


"""
Data that we are interested in: 
0. over the ranks: 5, 20, 500 (500 is effectively not low rank) 
1. Just random.
2. Power law with params: 1.5, 2. 
"""

def _generate_power_law(n: int, k: int, pl_param: float = None, psd: bool = False):
    """
    generate a matrix so that its singular values follow  power law distribution.  
    :param n: the side length of the matrix. 
    :param k: the rank of the matrix. 
    """
    sigmas = [i ** (-pl_param) for i in range(1, k + 1)]
    rd_m1 = np.random.normal(0, 1, [n, k])
    u1, _, _ = np.linalg.svd(rd_m1, full_matrices=False)
    rd_m2 = np.random.normal(0, 1, [n, k])
    if not psd:
        u2, _, _ = np.linalg.svd(rd_m2, full_matrices=False)
    else:
        u2 = u1
    #TODO: produce details. 
    detail = {'U': u1@np.sqrt(np.diag(sigmas)), 'V': u2@np.sqrt(np.diag(sigmas))}
    
    return u1@np.diag(sigmas)@(u2.T), detail 

def generate_latent(n: int, k: int, pl_param: Optional[float] = None, psd: bool = False):
    if pl_param is None: 
        return _generate_latent(n, k, psd)
    return _generate_power_law(n, k, pl_param, psd)

def _generate_latent(n: int, k: int, psd: bool = False): 
    """
    Generate low rank latent matrix and observed matrices. 
    
    I also need to determine the density (maybe controlled by alpha). 
    """
    detail = {}
    if psd: 
        latent = np.random.normal(0, 1, (n, k)) 
        detail['U'] = latent
        latent = latent@latent.T 
    else: 
        left_latent = np.random.normal(0, 1, (n, k))
        right_latent = np.random.normal(0, 1, (n, k)) 
        latent = left_latent@right_latent.T 
        detail['U'] = left_latent 
        detail['V'] = right_latent 
    return latent, detail 
    
def generate_binary(latent: np.ndarray):
    rnd = np.random.uniform(0, 1, latent.shape)
    threshold = sigmoid(latent)
    return (rnd < threshold).astype(float) 

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def gen_logistic_data(n, k, beta = None): 
    if beta is not None: 
        assert k == beta.reshape(-1).shape[0]
    else: 
        beta = np.random.normal(0, 1, (k, 1)) 
    
    obs = np.random.normal(0, 1, (n, k))
    prob = sigmoid(obs@beta)
    y = generate_binary(prob) 
    return {'x': obs, 'y': y, 'beta': beta}


def rectangular_partition_train_val_test(n: int, m: int, train_p: Optional[float] = None, val_p: Optional[float] = None):
    """
    Partition a matrix into training, validation, and test set.
    :param n: the number of rows in the matrix.
    :param m: the number of columns in the matrix. 
    :param train_p: the probability that an element is in the training set.
    :param val_p: the probability that an element is in the validation set.
    """
    assert (train_p is None and val_p is None) or (train_p is not None and val_p is not None)
    if train_p is None:
        return {"train": None, "val": None, "test": None}

    assert 0 <= train_p + val_p <= 1
    test_p = 1 - train_p - val_p
    all_p = [train_p, val_p, test_p]
    r = rv_discrete(values=([0, 1, 2], all_p))
    sampled = r.rvs(size=[n, m])
    trainOmega = (sampled == 0).astype(float)
    valOmega = (sampled == 1).astype(float)
    testOmega = (sampled == 2).astype(float)
    return {"train": trainOmega, "val": valOmega, "test": testOmega}


def partition_train_val_test(n: int, train_p: Optional[float] = None, val_p: Optional[float] = None):
    return rectangular_partition_train_val_test(n, n, train_p, val_p)
    