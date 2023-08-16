import numpy as np

from paths import KARATE_PR_NAME, csv_path, img_dir, solver_dir
from QStats.provider.provider import ProblemInstance
from QStats.search.advantage_search import AdvantageSearch
from QStats.search.base import ParamGrid
from QStats.solvers.advantage.advantage import Advantage
from Utils.Printer.printer import Printer
from util import G, MATRIX_RESOLUTION, MOD_SCORE, SAMPLE, SCORE_RESOLUTION


ID = 2
N_COMMUNITIES = 2

SCORE_RES = 1

RES_RUNS = 20
N_RUNS_PER_PARAM = 5


matrix_res_space = np.linspace(0.4, 1.5, RES_RUNS)
score_res_space = np.array([SCORE_RES] * RES_RUNS)

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
# npzfile = np.load(f"{ID}_res_{KARATE_PR_NAME}_{Advantage.name}.npz", allow_pickle=True)
# res = npzfile["res"]

Printer.csv_from_array(res, "w", csv_path(ID, KARATE_PR_NAME, Advantage.name))
Printer.draw_samples_modularities(
    samples=res[SAMPLE],
    mod_scores=res[MOD_SCORE],
    matrix_res=res[MATRIX_RESOLUTION],
    score_res=res[SCORE_RESOLUTION],
    graph=G,
    base_path=img_dir(id=ID, problem_name=KARATE_PR_NAME, solver_name="adv"),
    solver="adv"
)