import numpy as np

from paths import BRAIN_PR_NAME, csv_path
from QStats.provider.provider import ProblemInstance
from QStats.search.advantage_search import AdvantageSearch
from QStats.search.base import ParamGrid
from QStats.solvers.advantage.advantage import Advantage
from Utils.Printer.printer import Printer

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

search = AdvantageSearch(
    id=ID,
    problem_instance_callable=ProblemInstance.brain_problem,
    problem_name=BRAIN_PR_NAME,
)
res: np.ndarray = search.search_grid(
    param_grid=param_grid,
    n_runs_per_param=N_RUNS_PER_PARAM,
    n_communities=N_COMMUNITIES,
)

np.savez(f"{ID}_res_{BRAIN_PR_NAME}_{Advantage.name}", res=res)
# npzfile = np.load(f"res_{ID}.npz", allow_pickle=True)
# res = npzfile["res"]

Printer.csv_from_array(res, "w", csv_path(ID, BRAIN_PR_NAME, Advantage.name))
