import numpy as np

from paths import BRAIN_PR_NAME, csv_path, img_dir
from Printer.printer import Printer
from QStats.search.search import ParamGrid, Search
from QStats.solvers.advantage.advantage import Advantage
from util import (BRAIN_NETWORK_GRAPH, MATRIX_RESOLUTION, MOD_SCORE, SAMPLE,
                  SCORE_RESOLUTION)

ID = 6

RES_RUNS = 15
N_RUNS_PER_PARAM = 5
N_COMMUNITIES = 2


matrix_res_space = np.linspace(0.8, 1.4, RES_RUNS)
score_res_space = np.array([1] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    score_resolutions=score_res_space,
)

search = Search(id=ID)
res: np.ndarray = search.search_grid(
    param_grid=param_grid,
    n_runs_per_param=N_RUNS_PER_PARAM,
    n_communities=N_COMMUNITIES,
)

np.savez(f"res_{ID}", res=res)
# npzfile = np.load(f"res_{ID}.npz", allow_pickle=True)
# res = npzfile["res"]

Printer.csv_from_array(res, "w", csv_path(ID, BRAIN_PR_NAME, Advantage.name))
# Printer.draw_samples_modularities(
#     samples=res[SAMPLE],
#     mod_scores=res[MOD_SCORE],
#     matrix_res=res[MATRIX_RESOLUTION],
#     score_res=res[SCORE_RESOLUTION],
#     graph=BRAIN_NETWORK_GRAPH,
#     base_path=img_dir(ID, BRAIN_PR_NAME, Advantage.name),
#     solver=Advantage.name,
# )
