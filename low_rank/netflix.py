import numpy as np
from mc import mc_loss_func, GDSolverParam, mc_grad
from tqdm import tqdm
from itertools import product
from util import save_obj
from multiprocessing import Pool


def _misclassify(
    x: np.ndarray, u: np.ndarray, v: np.ndarray, omega: np.ndarray, threshold: float
):
    c = (u @ v.T > threshold).astype(int)
    return np.sum(np.abs(omega * (c - x))) / np.sum(omega)


def logistic_misclassify(
    x: np.ndarray, u: np.ndarray, v: np.ndarray, omega: np.ndarray
):
    threshold = 0
    return _misclassify(x, u, v, omega, threshold=threshold)


def mse_misclassify(x: np.ndarray, u: np.ndarray, v: np.ndarray, omega: np.ndarray):
    """
    Expect to use MSE to fit variables between 0 and 1.
    """
    threshold = 0.5
    return _misclassify(x, u, v, omega, threshold=threshold)


def hit_rate(
    x: np.ndarray, u: np.ndarray, v: np.ndarray, omega: np.ndarray, top: float = 200
):
    """
    For each user, among top 200 scored items, find the number
    of items that are viewed.
    """
    est = u @ v.T
    n_row = est.shape[0]
    rg_hit = []
    for i in range(n_row):
        u = est[i, :][omega[i, :] == 1]
        threshold = sorted(u)[-top:][0]
        u = (u >= threshold).astype(int)
        tsm = (x[i, :][omega[i, :] == 1] > 0).astype(int)
        rg_hit.append(np.sum(u * tsm))
    return np.mean(rg_hit)


def mse_loss_func(X, U, V, Omega=None):
    estX = U @ V.T
    # convert logistic language.
    y = X.reshape(-1)
    predy = estX.reshape(-1)
    if Omega is not None:
        mask = Omega.reshape(-1)
    else:
        mask = np.ones(predy.shape)
    return np.sum(mask * ((y - predy) ** 2)) / np.sum(mask)


def mse_grad(X, U, V, Omega_train=None, mu=0.01):
    estX = U @ V.T
    gradX = estX - X
    if Omega_train is not None:
        gradX = Omega_train * gradX

    gradU = gradX @ V
    gradV = gradX.T @ U
    reg = U.T @ U - V.T @ V
    reg_gradU = gradU + mu * U @ reg
    reg_gradV = gradV - mu * V @ reg
    return reg_gradU, reg_gradV

def solver_params():
    ls_estk = [5, 10, 100]
    ls_lr = [3e-3, 1e-3]
    ls_n_iter = [200]
    ls_scale_init = [0.01]
    ls_random_seed = [0,1]
    ls_p = []
    for estk, lr, n_iter, scale_init, random_seed in product(ls_estk, ls_lr, ls_n_iter, ls_scale_init, ls_random_seed): 
        ls_p.append(GDSolverParam(estk=estk,lr=lr, n_iter=n_iter, scale_init=scale_init, random_seed=random_seed))
    return ls_p


#some params that could work. 
#res = factored_grad(logistic_obs, estk=100, lr=0.005, n_iter=50, scale_init=0.1/5, Omegas=Omegas)

def netflix_factored_grad(X, Omegas, p: GDSolverParam, disable_tqdm: bool = False):
    n_row = X.shape[0]
    n_col = X.shape[1]
    U = np.random.normal(0, p.scale_init, (n_row, p.estk))
    V = np.random.normal(0, p.scale_init, (n_col, p.estk))
    Omega_train = Omegas["train"]
    Omega_val = Omegas["val"]
    Omega_test = Omegas["test"]

    train_loss = []
    train_misclassify = []
    val_loss = []
    val_misclassify = []
    test_loss = []
    test_misclassify = []
    lst_U = []
    lst_V = []
    lst_test_top = []

    def hit_rate(_estU, _estV, _Omega, top=200):
        est = _estU @ _estV.T
        rg_hit = []
        for i in range(n_row):
            u = est[i, :]
            u = u[_Omega[i, :] == 1]
            tsm = X[i, :]
            tsm = tsm[_Omega[i, :] == 1]
            tsm = (tsm > 0).astype(int)
            threshold = sorted(u)[-top:][0]
            u = (u >= threshold).astype(int)
            rg_hit.append(np.sum(u * tsm))
        return np.mean(rg_hit)

    def _add_accu(_estU, _estV):
        train_loss.append(mc_loss_func(X, _estU, _estV, Omega_train))
        train_misclassify.append(logistic_misclassify(X, _estU, _estV, Omega_train))

        val_loss.append(mc_loss_func(X, _estU, _estV, Omega_val))
        val_misclassify.append(logistic_misclassify(X, _estU, _estV, Omega_val))

        test_loss.append(mc_loss_func(X, _estU, _estV, Omega_test))
        test_misclassify.append(logistic_misclassify(X, _estU, _estV, Omega_test))

    for i in tqdm(range(p.n_iter), disable=disable_tqdm):
        gradU, gradV = mc_grad(X, U, V, Omega_train)
        U = U - p.lr * gradU
        V = V - p.lr * gradV
        _add_accu(U, V)
        if i % 10 == 9:
            lst_U.append(U)
            lst_V.append(V)
            lst_test_top.append(hit_rate(U, V, Omega_test))
    res = {"U": lst_U, "V": lst_V, "test_top": lst_test_top}
    res["train_loss"] = train_loss
    res["train_misclassify"] = train_misclassify
    res["val_loss"] = val_loss
    res["val_misclassify"] = val_misclassify
    res["test_loss"] = test_loss
    res["test_misclassify"] = test_misclassify
    return res

def wrap_experiments(X, Omegas, psolver_list, is_server=False):
    pbar = tqdm(total = len(psolver_list))

    global _run 
    def _run(_psolver: GDSolverParam): 
        res = netflix_factored_grad(X, Omegas, _psolver, disable_tqdm=True)
        name = "_".join(['netflix_logistic', _psolver._str()])
        save_obj(res, name)
        pbar.update(1)
        return name

    if is_server:
        with Pool(2) as pool: 
            res = pool.starmap(_run,  psolver_list)
    else:
        res = list(_run(psolver) for psolver in tqdm(psolver_list))
    return res

def netflix_mse_factored_grad(X, Omegas, p: GDSolverParam, disable_tqdm: bool = False): 
    """
    Use MSE to solve the problem. This is quite similar to netflix_factored_grad and needs refactoring. 
    """
    n_row = X.shape[0]
    n_col = X.shape[1]
    U = np.random.normal(0, p.scale_init, (n_row, p.estk))
    V = np.random.normal(0, p.scale_init, (n_col, p.estk))
    Omega_train = Omegas["train"]
    Omega_val = Omegas["val"]
    Omega_test = Omegas["test"]

    train_loss = []
    train_misclassify = []
    val_loss = []
    val_misclassify = []
    test_loss = []
    test_misclassify = []
    lst_U = []
    lst_V = []
    lst_test_top = []

    def hit_rate(_estU, _estV, _Omega, top=200):
        est = _estU @ _estV.T
        rg_hit = []
        for i in range(n_row):
            u = est[i, :]
            u = u[_Omega[i, :] == 1]
            tsm = X[i, :]
            tsm = tsm[_Omega[i, :] == 1]
            tsm = (tsm > 0).astype(int)
            threshold = sorted(u)[-top:][0]
            u = (u >= threshold).astype(int)
            rg_hit.append(np.sum(u * tsm))
        return np.mean(rg_hit)

    def _add_accu(_estU, _estV):
        train_loss.append(mse_loss_func(X, _estU, _estV, Omega_train))
        train_misclassify.append(mse_misclassify(X, _estU, _estV, Omega_train))

        val_loss.append(mse_loss_func(X, _estU, _estV, Omega_val))
        val_misclassify.append(mse_misclassify(X, _estU, _estV, Omega_val))

        test_loss.append(mse_loss_func(X, _estU, _estV, Omega_test))
        test_misclassify.append(mse_misclassify(X, _estU, _estV, Omega_test))

    for i in tqdm(range(p.n_iter), disable=disable_tqdm):
        gradU, gradV = mse_grad(X, U, V, Omega_train)
        U = U - p.lr * gradU
        V = V - p.lr * gradV
        _add_accu(U, V)
        if i % 10 == 9:
            lst_U.append(U)
            lst_V.append(V)
            lst_test_top.append(hit_rate(U, V, Omega_test))
    res = {"U": lst_U, "V": lst_V, "test_top": lst_test_top}
    res["train_loss"] = train_loss
    res["train_misclassify"] = train_misclassify
    res["val_loss"] = val_loss
    res["val_misclassify"] = val_misclassify
    res["test_loss"] = test_loss
    res["test_misclassify"] = test_misclassify
    return res


def wrap_mse_experiments(X, Omegas, psolver_list, is_server=False):
    pbar = tqdm(total = len(psolver_list))

    global _run 
    def _run(_psolver: GDSolverParam): 
        res = netflix_mse_factored_grad(X, Omegas, _psolver, disable_tqdm=True)
        name = "_".join(['netflix_mse', _psolver._str()])
        save_obj(res, name)
        pbar.update(1)
        return name

    if is_server:
        with Pool(2) as pool: 
            res = pool.map(_run,  psolver_list)
    else:
        res = list(_run(psolver) for psolver in tqdm(psolver_list))
    return res