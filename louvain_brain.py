import numpy as np

from paths import BRAIN_PR_NAME, csv_path
from Printer.printer import Printer
from QStats.search.louvain_search import LouvainSearch, ParamGrid
from QStats.solvers.louvain.louvain import Louvain
from util import BRAIN_NETWORK_GRAPH

ID = 1

C_RES = 0.11
M_RES = 1

RES_RUNS = 60
N_RUNS_PER_PARAM = 10
N_COMMUNITIES = 2


matrix_res_space = np.linspace(0.01, 1.8, RES_RUNS)
score_res_space = np.array([M_RES] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    score_resolutions=score_res_space,
)

search = LouvainSearch(id=ID, graph=BRAIN_NETWORK_GRAPH)
res: np.ndarray = search.search_grid(
    param_grid=param_grid, n_runs_per_param=N_RUNS_PER_PARAM
)
np.savez(f"res_{Louvain.name}_{ID}", res=res)

Printer.csv_from_array(
    res,
    "w",
    csv_path(id=ID, problem_name=BRAIN_PR_NAME, solver_name=Louvain.name),
)
# Printer.draw_samples_modularities(
#     samples=res[SAMPLE],
#     mod_scores=res[MOD_SCORE],
#     matrix_res=res[MATRIX_RESOLUTION],
#     score_res=res[SCORE_RESOLUTION],
#     graph=BRAIN_NETWORK_GRAPH,
#     base_path=img_dir(
#         id=ID, problem_name=BRAIN_PR_NAME, solver_name=Louvain.name
#     ),
#     solver=Louvain.name,
# )
