from util import * 
import numpy as np 
from tqdm import tqdm
from attr import frozen
from itertools import product
from multiprocessing import Pool


_VAR_PREFIX = 'vp2'

@frozen 
class LogisticDataParam:
    d: int = 100
    k: int = 5
    std: float = .1

    n_train: int = 1000
    n_val: int = 1000
    power_law: float = .75

    def _str(self): 
        szstd = "{0:3f}".format(self.std) 
        sz_power_law = "{0:3f}".format(self.power_law) if self.power_law is not None else 'none'
        return f"{_VAR_PREFIX}_logistic_{self.d}_{self.k}_{szstd}_{self.n_train}_{self.n_val}_{sz_power_law}"

def logistic_generate_data(params: LogisticDataParam): 
    latent, detail = generate_latent(params.d, params.k, params.power_law)
    std_lt = latent.reshape(-1).std()
    scale = params.std / std_lt
    latent = latent * scale
    
    def _get_data(n: int): 
        x = np.random.normal(0, 1, [n, params.d])
        y = generate_binary(x@latent)
        return x, y
    x_train, y_train = _get_data(params.n_train)
    x_val, y_val = _get_data(params.n_val)
    res = {'latent': latent, 'detail': detail, 'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val}
    return res


@frozen 
class LogisticGDSolverParam: 
    """
    Parameters related to factored gradient descent 
    """
    estk : int = 5 
    lr : float = 1e-3
    n_iter: int = 400
    scale_init: float = .2 
    random_seed: int = 0
    def _str(self): 
        sz_lr = "{0:7f}".format(self.lr)
        sz_scale_init = "{0:7f}".format(self.scale_init)
        return f"{_VAR_PREFIX}.lgsolver_{self.estk}_{sz_lr}_{self.n_iter}_{sz_scale_init}_{self.random_seed}"

def logistic_solver_params():
    ls_estk = [5, 10]
    ls_lr = [1e-2, 1e-3] #, 3e-4]
    ls_n_iter = [800]
    ls_scale_init = [.5, .05]
    ls_random_seed = [0,1,2]
    ls_p = []
    for estk, lr, n_iter, scale_init, random_seed in product(ls_estk, ls_lr, ls_n_iter, ls_scale_init, ls_random_seed): 
        ls_p.append(LogisticGDSolverParam(estk=estk,lr=lr, n_iter=n_iter, scale_init=scale_init, random_seed=random_seed))
    return ls_p


def logistic_generate_experiment_data(no_generate=False):
    ls_power = [.75, None]
    ls_d = [500, 1000]
    ls_k = [5, 50]
    ls_std = [.05, .5]
    ls_train_val = [(1000, 1000), (4000, 4000)]
    ls_p = []
    for power_law, d, k, std, train_val in tqdm(list(product(ls_power, ls_d, ls_k, ls_std, ls_train_val))):
        n_train = train_val[0]
        n_val = train_val[1]
        p = LogisticDataParam(d=d, k=k, std=std, n_train=n_train, n_val=n_val, power_law=power_law)
        if not no_generate:
            data = logistic_generate_data(p)
            save_obj(data, p._str())
        ls_p.append(p)
    return ls_p


def logistic_loss_func(x, y, beta):
    predy = sigmoid(x@beta).reshape(-1)
    y = y.reshape(-1) * 2 - 1
    return -np.mean(np.log(sigmoid(y * (predy))))

def logistic_grad(x, y, U, V, mu = 0.01):
    beta = U@V.T
    predy = sigmoid(x@beta)
    n_obs = x.shape[0]
    gradbeta = (x.T@(predy - y) / n_obs)
    gradU = gradbeta@V
    gradV = gradbeta.T@U
    reg = U.T@U - V.T@V
    reg_gradU = gradU + mu * U@reg 
    reg_gradV = gradV - mu * V@reg
    return reg_gradU, reg_gradV 
    
def factored_gd_reg(p_data: LogisticDataParam, p: LogisticGDSolverParam, disable_tqdm=False):
    """
    Need to pull out the following information: 
    (i) With respect to ground-truth: mse, correlation. 
    (ii) With respect to training data: loss. 
    (iii) With respect to validation data: loss, misclassifcation rate. 
    (iv) U and V: std 
    """
    data = load_obj(p_data._str())
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    hints = data['latent']
    
    U = np.random.normal(0, p.scale_init, (p_data.d, p.estk))
    V = np.random.normal(0, p.scale_init, (p_data.d, p.estk))

    #ground-truth. 
    gt_mse = []
    gt_cor = []
    
    #training data
    train_loss = []
    #valiation data
    val_loss = []
    val_misclassify = [] 
    u_std = []
    v_std = []

    lst_U = []
    lst_V = []

    def _misclassify(_estBeta, _x, _y):
        predy = (_x@_estBeta>0).astype(float)
        return np.mean(np.abs(predy - _y).reshape(-1))

    def _add_accu(_estU, _estV):
        _estBeta = _estU@_estV.T
        gt_mse.append(np.mean((hints.reshape(-1) - _estBeta.reshape(-1))**2))
        gt_cor.append(np.corrcoef(hints.reshape(-1), _estBeta.reshape(-1))[0, 1])
        train_loss.append(logistic_loss_func(x_train, y_train, _estBeta))
        val_loss.append(logistic_loss_func(x_val, y_val, _estBeta))
        val_misclassify.append(_misclassify(_estBeta, x_val, y_val))
        u_std.append(np.std(_estU.reshape(-1)))
        v_std.append(np.std(_estV.reshape(-1)))

    for i in tqdm(range(p.n_iter), disable=disable_tqdm):
        gradU, gradV = logistic_grad(x_train, y_train, U, V)
        U = U - p.lr * gradU
        V = V - p.lr * gradV 
        _add_accu(U, V) 
        if i % 10 == 9:
            lst_U.append(U)
            lst_V.append(V)
    
    res = {"U": lst_U, "V": lst_V}    
    res['gt_mse'] = gt_mse
    res['gt_corr'] = gt_cor
    res['train_loss'] = train_loss
    res['val_loss'] = val_loss
    res['val_misclassify'] = val_misclassify
    res['u_std'] = u_std
    res['v_std'] = v_std

    return res 


def wrap_experiments(pdata_list, psolver_list): 
    pbar = tqdm(total = len(pdata_list) * len(psolver_list))

    global _run 
    def _run(_pdata: LogisticDataParam, _psolver: LogisticGDSolverParam): 
        res = factored_gd_reg(_pdata, _psolver, disable_tqdm=True)
        name = "_".join([_pdata._str(), _psolver._str()])
        save_obj(res, name)
        pbar.update(1)
        return name

    with Pool(5) as pool: 
        res = pool.starmap(_run, list(product(pdata_list, psolver_list)))
    return res


def to_frame(_p, _psolver, _res, idx_tag="0"):
    _res = _res.copy()
    _res.pop("U")
    _res.pop("V")
    _df = pd.DataFrame(_res)
    stat_cols = list(_df.columns)
    _df["d"] = _p.d
    _df["k"] = _p.k
    _df["std"] = _p.std
    _df["n_train"] = _p.n_train
    _df["n_val"] = _p.n_val
    _df["power law"] = _p.power_law
    data_cols = ["d", "k", "std", "n_train", "n_val", "power law"]
    _df["estk"] = _psolver.estk
    _df["lr"] = _psolver.lr
    _df["scale_init"] = _psolver.scale_init
    _df["seed"] = _psolver.random_seed
    _df["idx_tag"] = idx_tag
    _df["iters"] = list(range(_df.shape[0]))
    solver_cols = ["estk", "lr", "scale_init", "iters", "seed"]
    cols = data_cols + solver_cols + stat_cols + ["idx_tag"]
    return _df[cols]


def build_opt_p(use_p, use_solvers, res_map):
    to_concat = [
        to_frame(use_p, aps, res_map[(use_p, aps)], i)
        for i, aps in tqdm(enumerate(use_solvers), disable=True)
        if (use_p, aps) in res_map
    ]

    if len(to_concat) == 0:
        return None
    df_m = pd.concat(
        to_concat,
        axis=0,
        ignore_index=True,
    )

    # find the optimal one
    df_m = df_m[df_m["iters"] % 10 == 9]
    # idx = df_m["val_loss"].argmin()
    idx = df_m["val_misclassify"].argmin()
    res = {
        "d": use_p.d,
        "k": use_p.k,
        "std": use_p.std,
        "n_train": use_p.n_train,
        "n_val": use_p.n_val,
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
        "n_iter": 800,
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

    use_solver = LogisticGDSolverParam(
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
        LogisticGDSolverParam(
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

    mean_corr = np.mean(all_cors) if len(all_cors) > 0 else np.nan
    mean_predcorr = np.mean(all_pred_corr) if len(all_pred_corr) > 0 else np.nan
    raw_data = load_obj(use_p._str())

    rrr["mean_corr"] = mean_corr
    rrr["pred_corr"] = mean_predcorr
    rrr["improved_corr"] = np.corrcoef(
        raw_data["latent"].reshape(-1), avg_estG.reshape(-1)
    )[0, 1]
    return rrr

def filter_none(lst):
    return [e for e in lst if e is not None]


def sample_summary():
    p = logistic_generate_experiment_data(no_generate=True)
    p_solvers = logistic_solver_params()
    Res = {}
    for ap, apsolver in tqdm(product(p[:10], p_solvers[:5])):
        name = "_".join([ap._str(), apsolver._str()])
        Res[(ap, apsolver)] = load_obj(name)
    use_solvers = [psolver for psolver in p_solvers[:5]]  # if psolver.estk == 100]
    res_df = pd.concat(
        [build_opt_p(pp, use_solvers, Res) for pp in tqdm(p[:10])],
        axis=0,
        ignore_index=True,
    )
    return res_df
