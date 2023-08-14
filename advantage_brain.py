import numpy as np

from paths import BRAIN_PR_NAME, csv_path
from Printer.printer import Printer
from QStats.provider.provider import ProblemInstance
from QStats.search.advantage_search import ParamGrid, Search
from QStats.solvers.advantage.advantage import Advantage

ID = 8

RES_RUNS = 2
N_RUNS_PER_PARAM = 2
N_COMMUNITIES = 2


matrix_res_space = np.linspace(0.4, 1.8, RES_RUNS)
score_res_space = np.array([1] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    score_resolutions=score_res_space,
)

search = Search(id=ID, problem_instance_callable=ProblemInstance.brain_problem)
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
