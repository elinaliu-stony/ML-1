from itertools import product
from multiprocessing import Pool
import cvxpy as cp
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from attr import frozen
from tqdm import tqdm
from util import (
    generate_binary,
    generate_latent,
    load_obj,
    partition_train_val_test,
    save_obj,
    sigmoid,
)

_VAR_PREFIX = "v5"

@frozen
class MCDataParam:
    n: int = 1000
    k: int = 5
    std: float = 5
    # fraction of observations that are disclosed.
    p: float = 1

    # probability that each observation is in train or val
    # (if not in either train or val, then in test.)
    p_train: float = .4 
    p_val: float = .3
    
    #TODO: check that power law is also coded. 
    power_law: float = .75
    
    p_train: float = 0.4
    p_val: float = 0.3

    # TODO: check that power law is also coded.
    power_law: float = 0.75

    def _str(self):
        szstd = "{0:3f}".format(self.std)
        szp = "{0:3f}".format(self.p)
        sz_p_train = (
            "{0:3f}".format(self.p_train) if self.p_train is not None else "none"
        )
        sz_p_val = "{0:3f}".format(self.p_val) if self.p_val is not None else "none"
        sz_power_law = (
            "{0:3f}".format(self.power_law) if self.power_law is not None else "none"
        )
        return f"{_VAR_PREFIX}.{self.n}_{self.k}_{szstd}_{szp}_{sz_p_train}_{sz_p_val}_{sz_power_law}"


def generate_data(params: MCDataParam):
    latent, detail = generate_latent(params.n, params.k, params.power_law)
    std_lt = latent.reshape(-1).std()
    scale = params.std / std_lt
    latent = latent * scale
    x = generate_binary(latent)
    res = {"x": x, "latent": latent, "detail": detail}
    # keys in data_partion: train, val, test.
    data_partition = partition_train_val_test(params.n, params.p_train, params.p_val)
    res.update(data_partition)
    return res


@frozen
class GDSolverParam:
    """
    Parameters related to factored gradient descent
    """

    estk: int = 5
    lr: float = 1e-3
    n_iter: int = 1000
    scale_init: float = 0.2
    random_seed: int = 0

    def _str(self):
        sz_lr = "{0:7f}".format(self.lr)
        sz_scale_init = "{0:7f}".format(self.scale_init)
        return f"{_VAR_PREFIX}.solver_{self.estk}_{sz_lr}_{self.n_iter}_{sz_scale_init}_{self.random_seed}"


def solver_params():
    ls_estk = [5, 10, 100]
    ls_lr = [3e-3, 1e-3, 3e-4]
    ls_n_iter = [200]
    ls_scale_init = [1, 0.1]
    ls_random_seed = [0, 1, 2]
    ls_p = []
    for estk, lr, n_iter, scale_init, random_seed in product(
        ls_estk, ls_lr, ls_n_iter, ls_scale_init, ls_random_seed
    ):
        ls_p.append(
            GDSolverParam(
                estk=estk,
                lr=lr,
                n_iter=n_iter,
                scale_init=scale_init,
                random_seed=random_seed,
            )
        )
    return ls_p

def generate_experiment_data(no_generate=False): 
    ls_power = [.75, None]
    ls_n = [1000]

def generate_experiment_data(no_generate=False):
    ls_power = [0.75, None]
    ls_n = [1000]
    ls_k = [5, 50, 500]
    ls_std = [1, 10]
    #ls_train_val = [(None, None), (.7, .3), (.2, .3), (.1, .1)] #, (.2, .3)]
    ls_train_val = [(.7, .3), (.2, .3), (.1, .1)] 

    # ls_train_val = [(None, None), (.7, .3), (.2, .3), (.1, .1)] #, (.2, .3)]
    ls_train_val = [(0.7, 0.3), (0.2, 0.3), (0.1, 0.1)]  # , (.2, .3)]

    ls_p = []
    
    for (
        power_law,
        n,
        k,
        std,
        train_val,
    ) in tqdm(list(product(ls_power, ls_n, ls_k, ls_std, ls_train_val))):
        p_train = train_val[0]
        p_val = train_val[1]
        p = MCDataParam(
            n=n, k=k, std=std, p_train=p_train, p_val=p_val, power_law=power_law
        )
        if not no_generate:
            data = generate_data(p)
            save_obj(data, p._str())
        ls_p.append(p)
    return ls_p


def mc_loss_func(X, U, V, Omega=None):
    """
    If Omega is not None, just find the loss that is from Omega
    """
    estX = sigmoid(U @ V.T)
    # convert logistic language.
    y = X.reshape(-1)
    y = y * 2 - 1
    predy = estX.reshape(-1)
    if Omega is not None:
        mask = Omega.reshape(-1)
    else:
        mask = np.ones(predy.shape)
    return -np.sum(np.log(sigmoid(y * (predy))) * mask) / np.sum(mask)


def mc_grad(X, U, V, Omega_train=None, mu=0.01):
    # TODO: do we also need to scan mu?
    estX = U @ V.T
    gradX = sigmoid(estX) - X
    if Omega_train is not None:
        gradX = Omega_train * gradX

    gradU = gradX @ V
    gradV = gradX.T @ U
    reg = U.T @ U - V.T @ V
    reg_gradU = gradU + mu * U @ reg
    reg_gradV = gradV - mu * V @ reg
    return reg_gradU, reg_gradV


def factored_gd_mc(p_data: MCDataParam, p: GDSolverParam, disable_tqdm=False):
    """
    Rewrite factored_gd.

    Need to pull out the following information:
    (i) With respect to ground-truth: mse, correlation, loss.
    (ii) With respect to training data: loss.
    (iii) With respect to validation data: loss, misclassifcation rate.
    (iv) U and V: std

    """
    data = load_obj(p_data._str())
    x = data["x"]
    hints = data["latent"]
    Omega_train = data["train"]
    Omega_val = data["val"]
    n = x.shape[0]
    U = np.random.normal(0, p.scale_init, (n, p.estk))
    V = np.random.normal(0, p.scale_init, (n, p.estk))

    # ground-truth.
    gt_mse = []
    gt_cor = []
    gt_loss = []

    # training data
    train_loss = []
    # valiation data
    val_loss = []
    val_misclassify = []
    u_std = []
    v_std = []

    lst_U = []
    lst_V = []

    def _misclassify(_estU, _estV, Omega):
        classify = (_estU @ _estV.T >= 0).astype(float)
        if Omega is None:
            Omega = np.ones(classify.shape)

        cnt = np.sum(Omega.reshape(-1))
        return np.sum((Omega * np.abs(x - classify)).reshape(-1)) / cnt

    def _add_accu(_estU, _estV):
        if hints is None:
            return
        estX = _estU @ _estV.T
        gt_mse.append(np.mean((hints.reshape(-1) - estX.reshape(-1)) ** 2))
        gt_cor.append(np.corrcoef(hints.reshape(-1), estX.reshape(-1))[0, 1])
        gt_loss.append(mc_loss_func(x, U, V))

        train_loss.append(mc_loss_func(x, U, V, Omega_train))
        val_loss.append(mc_loss_func(x, U, V, Omega_val))
        val_misclassify.append(_misclassify(U, V, Omega_val))
        u_std.append(np.std(_estU.reshape(-1)))
        v_std.append(np.std(_estV.reshape(-1)))

    for i in tqdm(range(p.n_iter), disable=disable_tqdm): 
        
        #gradU, gradV = mc_grad(x, U, V) 
        gradU, gradV = mc_grad(x, U, V, Omega_train) 
    for i in tqdm(range(p.n_iter), disable=disable_tqdm):

        gradU, gradV = mc_grad(x, U, V, Omega_train)
        U = U - p.lr * gradU
        V = V - p.lr * gradV
        _add_accu(U, V)
        if i % 10 == 9:
            lst_U.append(U)
            lst_V.append(V)

    res = {"U": lst_U, "V": lst_V}
    res["gt_mse"] = gt_mse
    res["gt_corr"] = gt_cor
    res["gt_loss"] = gt_loss
    res["train_loss"] = train_loss
    res["val_loss"] = val_loss
    res["val_misclassify"] = val_misclassify
    res["u_std"] = u_std
    res["v_std"] = v_std

    return res

def wrap_experiments(pdata_list, psolver_list): 
    pbar = tqdm(total = len(pdata_list) * len(psolver_list))

    global _run

    def _run(_pdata: MCDataParam, _psolver: GDSolverParam):
        res = factored_gd_mc(_pdata, _psolver, disable_tqdm=True)
        name = "_".join([_pdata._str(), _psolver._str()])
        save_obj(res, name)
        pbar.update(1)
        return name

    with Pool(5) as pool: 
        res = pool.starmap(_run, list(product(pdata_list, psolver_list)))
    return res

def mosek_loss_func(y, latent, Omega = None): 
    y = y.reshape(-1)
    predy = sigmoid(latent).shape(-1) 
    y = y * 2 - 1
    if Omega is not None:
        mask = Omega.reshape(-1)
    else:
        mask = np.ones(predy.shape)
    return -np.sum(np.log(sigmoid(y * (predy))) * mask) / np.sum(mask)



def _mosek_solve(x: np.ndarray, lambd_val: float, Omega: Optional[np.ndarray], hints = None, verbose=True):
    n = x.shape[0] 
    lambd = cp.Parameter(nonneg=True)
    est_latent = cp.Variable((n,n))
    mask = (Omega > 0)
    log_likelihood = cp.sum(
        (cp.multiply(x, est_latent) - cp.logistic(est_latent))[mask]
    )
    problem = cp.Problem(cp.Maximize(log_likelihood - lambd * cp.norm(est_latent, "nuc")))
    lambd.value = lambd_val
    problem.solve(verbose=verbose)
    sol = est_latent.value
    acc_mse = np.mean((hints.reshape(-1) - sol.reshape(-1)) ** 2)
    acc_cor = np.corrcoef(hints.reshape(-1), sol.reshape(-1))[0, 1]
    acc_loss = mosek_loss_func(x, sol)
    return {
        "est_latent": sol,
        "acc_mse": acc_mse,
        "acc_cor": acc_cor,
        "acc_loss": acc_loss,
    }


@frozen
class MosekParam:
    """
    Parameter related to Mosek solver.
    """

    lambd_val: float = 1

    def _str(self):
        sz_lambd_val = "{0:7f}".format(self.lambd_val)
        return f'{_VAR_PREFIX}.msolver_{sz_lambd_val}'

def mosek_solver_params():
    trials = 20
    ls_lambd_vals = np.linspace(0, 20, trials)
    return [MosekParam(lambd_val) for lambd_val in ls_lambd_vals]

def mosek_solve(p_data: MCDataParam, p: MosekParam, verbose=True):
    data = load_obj(p_data._str())
    x = data['x']
    hints = data['latent']
    Omega = data['train']
    if Omega is None:
        Omega = np.ones(x.shape)
    return _mosek_solve(x, lambd_val=p.lambd_val, Omega=Omega, hints=hints, verbose=verbose)  


def wrap_mosek_experiments(pdata_list, psolver_list):
    pbar = tqdm(total=len(pdata_list) * len(psolver_list))
    global _run

    def _run(_pdata: MCDataParam, _psolver: MosekParam):
        res = mosek_solve(_pdata, _psolver, verbose=False)
        name = "_".join([_pdata._str(), _psolver._str()])
        save_obj(res, name)
        pbar.update(1)
        return name

    with Pool(4) as pool:
        res = pool.starmap(_run, list(product(pdata_list, psolver_list)))
    return res


def to_frame(p: MCDataParam, psolver: GDSolverParam, res: Dict, idx_tag: str = "0"):
    """
    Given a data param (MCDataParam), a solver param (GDSolverParam), and the corresponding
    result, produce
    """
    res = res.copy()
    res.pop("U")
    res.pop("V")
    df = pd.DataFrame(res)
    stat_cols = list(df.columns)
    df["n"] = p.n
    df["k"] = p.k
    df["std"] = p.std
    df["p_train"] = p.p_train
    df["p_val"] = p.p_val
    df["power law"] = p.power_law
    data_cols = ["n", "k", "std", "p_train", "p_val", "power law"]
    df["estk"] = psolver.estk
    df["lr"] = psolver.lr
    df["scale_init"] = psolver.scale_init
    df["seed"] = psolver.random_seed
    df["idx_tag"] = idx_tag
    df["iters"] = list(range(df.shape[0]))
    solver_cols = ["estk", "lr", "scale_init", "iters", "seed"]
    cols = data_cols + solver_cols + stat_cols + ["idx_tag"]
    return df[cols]


def build_opt_p(use_p: MCDataParam, use_solvers: List[GDSolverParam], res_map: Dict):
    df_m = pd.concat(
        [
            to_frame(use_p, aps, res_map[(use_p, aps)], i)
            for i, aps in tqdm(enumerate(use_solvers), disable=True)
        ],
        axis=0,
        ignore_index=True,
    )

    # find the optimal one
    df_m = df_m[df_m["iters"] % 10 == 9]
    # idx = df_m["val_loss"].argmin()
    idx = df_m["val_misclassify"].argmin()
    res = {
        "n": use_p.n,
        "k": use_p.k,
        "std": use_p.std,
        "p_train": use_p.p_train,
        "p_val": use_p.p_val,
        "power_law": use_p.power_law is not None,
    }
    res2 = {
        "estk": df_m.iloc[idx, :].estk,
        "lr": df_m.iloc[idx, :].lr,
        "scale_init": df_m.iloc[idx, :].scale_init,
        "mse": df_m.iloc[idx, :].gt_mse,
        "corr": df_m.iloc[idx, :].gt_corr,
        "val_loss": df_m.iloc[idx, :].val_loss,
        "val_misclass": df_m.iloc[idx, :].val_misclassify,
        "seed": df_m.iloc[idx, :].seed,
        "iters": df_m.iloc[idx, :].iters,
        "n_iter": 200,
    }

    # find the result using the last iteration.
    idx_tag = df_m.iloc[idx, :].idx_tag
    stopped_cor = df_m[df_m.idx_tag == idx_tag].iloc[-1, :].gt_corr
    if stopped_cor < 0 or stopped_cor <= df_m.iloc[idx, :].gt_corr * 0.9:
        res2["early_stop"] = True
    else:
        res2["early_stop"] = False

    res.update(res2)
    rrr = pd.DataFrame(res, index=[0])

    use_solver = GDSolverParam(
        estk=res2["estk"],
        lr=res2["lr"],
        scale_init=res2["scale_init"],
        n_iter=res2["n_iter"],
        random_seed=res2["seed"],
    )
    use_iter = int(df_m.iloc[idx, :].iters)

    useU = res_map[(use_p, use_solver)]["U"][use_iter // 10]
    useV = res_map[(use_p, use_solver)]["V"][use_iter // 10]

    estG = useU @ useV.T

    avg_estG = estG.copy()
    solvers_with_other_seeds = [
        GDSolverParam(
            estk=res2["estk"],
            lr=res2["lr"],
            scale_init=res2["scale_init"],
            n_iter=res2["n_iter"],
            random_seed=i,
        )
        for i in range(3)
        if i != res2["seed"]
    ]
    all_cors = []
    all_pred_corr = []

    for tmp in solvers_with_other_seeds:
        if (use_p, tmp) in res_map:
            all_cors.append(res_map[(use_p, tmp)]["gt_corr"][use_iter])
            nextU = res_map[(use_p, tmp)]["U"][use_iter // 10]
            nextV = res_map[(use_p, tmp)]["V"][use_iter // 10]
            nextG = nextU @ (nextV.T)
            avg_estG += nextG
            predcor = np.corrcoef((nextG).reshape(-1), estG.reshape(-1))
            all_pred_corr.append(predcor[0, 1])

    mean_corr = np.mean(all_cors)
    mean_predcorr = np.mean(all_pred_corr)
    raw_data = load_obj(use_p._str())

    rrr["mean_corr"] = mean_corr
    rrr["pred_corr"] = mean_predcorr
    rrr["improved_corr"] = np.corrcoef(
        raw_data["latent"].reshape(-1), avg_estG.reshape(-1)
    )[0, 1]
    return rrr
