import sys
path = "/home/yinan/research/ml_git/ml/low_rank"
sys.path.append(path)

from mc import generate_experiment_data, wrap_mosek_experiments, mosek_solver_params

p = generate_experiment_data(True)
mp_solvers = mosek_solver_params()
wrap_mosek_experiments(p, mp_solvers)