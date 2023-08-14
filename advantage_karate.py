import numpy as np

from paths import KARATE_PR_NAME, csv_path
from QStats.provider.provider import ProblemInstance
from QStats.search.advantage_search import AdvantageSearch
from QStats.search.base import ParamGrid
from QStats.solvers.advantage.advantage import Advantage
from Utils.Printer.printer import Printer

ID = 0

RES_RUNS = 2
N_RUNS_PER_PARAM = 2
N_COMMUNITIES = 2

ADV_C_RES = 1
ADV_M_RES = 0.5


matrix_res_space = np.linspace(0.5, 1.0, RES_RUNS)
score_res_space = np.array([1] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    score_resolutions=score_res_space,
)

search = AdvantageSearch(
    id=ID,
    problem_instance_callable=ProblemInstance.karate_problem,
    problem_name=KARATE_PR_NAME,
)
res: np.ndarray = search.search_grid(
    param_grid=param_grid,
    n_runs_per_param=N_RUNS_PER_PARAM,
    n_communities=N_COMMUNITIES,
)

np.savez(f"{ID}_res_{KARATE_PR_NAME}_{Advantage.name}", res=res)
Printer.csv_from_array(res, "w", csv_path(ID, KARATE_PR_NAME, Advantage.name))
