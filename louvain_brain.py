import numpy as np

from paths import BRAIN_PR_NAME, csv_path
from QStats.search.base import ParamGrid
from QStats.search.louvain_search import LouvainSearch
from QStats.solvers.louvain.louvain import Louvain
from util import BRAIN_NETWORK_GRAPH
from Utils.Printer.printer import Printer


ID = 1
N_COMMUNITIES = 2

GAMMA_LOWER_BOUND = 0.01
GAMMA_UPPER_BOUND = 1.8

SCORE_RES = 1

RES_RUNS = 60
N_RUNS_PER_PARAM = 10


matrix_res_space = np.linspace(GAMMA_LOWER_BOUND, GAMMA_UPPER_BOUND, RES_RUNS)
score_res_space = np.array([SCORE_RES] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    score_resolutions=score_res_space,
)

search = LouvainSearch(id=ID, graph=BRAIN_NETWORK_GRAPH)
res: np.ndarray = search.search_grid(
    param_grid=param_grid, n_runs_per_param=N_RUNS_PER_PARAM
)

Printer.csv_from_array(
    res,
    "w",
    csv_path(id=ID, problem_name=BRAIN_PR_NAME, solver_name=Louvain.name),
)
