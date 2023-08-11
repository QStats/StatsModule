import numpy as np
from QHyper.problems.community_detection import BrainNetwork

from paths import (IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE, csv_path,
                   img_dir)
from Printer.printer import Printer
from QStats.solvers.advantage.advantage import Advantage
from resolution_search import ParamGrid, Search
from util import MATRIX_RESOLUTION, MOD_SCORE, SAMPLE, SCORE_RESOLUTION


ID = 4

RES_RUNS = 15


graph = BrainNetwork(IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE).graph

matrix_res_space = np.linspace(1.05, 1.3, RES_RUNS)
score_res_space = np.array([1] * RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    modularity_score_resoltion=score_res_space,
)

res: np.ndarray = Search.run_grid(param_grid=param_grid, n_communities=2)
print("done")

np.savez(f"res_{ID}", res=res)

# npzfile = np.load("res_3.npz", allow_pickle=True)
# res = npzfile["res"]

Printer.csv_from_array(res, "w", csv_path(ID, "brain", "adv"))
Printer.draw_samples_modularities(
    samples=res[SAMPLE],
    mod_scores=res[MOD_SCORE],
    matrix_res=res[MATRIX_RESOLUTION],
    score_res=res[SCORE_RESOLUTION],
    graph=graph,
    base_path=img_dir(ID, "brain", Advantage.name),
    solver=Advantage.name,
)
