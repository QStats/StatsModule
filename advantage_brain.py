import numpy as np
from QStats.solvers.advantage.advantage import Advantage

from paths import IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE, csv_path, img_dir
from Printer.printer import Printer
from resolution_search import ParamGrid, Search
from util import MOD_SCORE, SAMPLE
from QHyper.problems.community_detection import BrainNetwork


C_RES = 1.2
M_RES = 1
RES_RUNS = 7


graph = BrainNetwork(IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE).graph

matrix_res_space = np.linspace(0.4, 10, RES_RUNS)
score_res_space = np.linspace(0.7, 24, RES_RUNS)

param_grid = ParamGrid(
    resolution_grid=matrix_res_space,
    modularity_score_resoltion=score_res_space,
)

res: np.ndarray = Search.run_grid(param_grid=param_grid)
print("---------------------------------")
print(f"RES: {res}")

Printer.csv_from_array(res, "w", csv_path("brain", "adv"))
Printer.draw_samples_modularities(
    samples=res[SAMPLE],
    modularities=res[MOD_SCORE],
    graph=graph,
    base_path=img_dir("brain", Advantage.name),
    solver=Advantage.name,
)
