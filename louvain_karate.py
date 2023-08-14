import numpy as np

from paths import KARATE_PR_NAME, csv_path
from QStats.search.base import ParamGrid
from QStats.search.louvain_search import LouvainSearch
from QStats.solvers.louvain.louvain import Louvain
from util import G
from Utils.Printer.printer import Printer

ID = 0

C_RES = 0.5
M_RES = 1

RES_RUNS = 20
N_RUNS_PER_PARAM = 10
N_COMMUNITIES = 2


matrix_res_space = np.linspace(0.5, 1.0, RES_RUNS)
score_res_space = np.array([M_RES] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    score_resolutions=score_res_space,
)

search = LouvainSearch(id=ID, graph=G)
res: np.ndarray = search.search_grid(
    param_grid=param_grid, n_runs_per_param=N_RUNS_PER_PARAM
)
np.savez(f"{ID}_res_{KARATE_PR_NAME}_{Louvain.name}", res=res)

Printer.csv_from_array(
    res,
    "w",
    csv_path(id=ID, problem_name=KARATE_PR_NAME, solver_name=Louvain.name),
)
