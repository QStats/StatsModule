import numpy as np

from paths import csv_path
from Printer.printer import Printer
from resolution_search import ParamGrid, Search

C_RES = 1.2
M_RES = 1
N_RUNS_PER_PARAM = 2
RES_RUNS = 7

resolution_space = np.linspace(0.8, 1, 2)

param_grid = ParamGrid(
    resolution_grid=resolution_space,
    modularity_score_resoltion=resolution_space,
)

res = Search.run_grid(param_grid=param_grid)
print("---------------------------------")
print(f"RES: {res}")

Printer.csv_from_array(res, "w", csv_path("brain", "adv"))
# Printer.draw_samples_modularities(
#     samples=samples,
#     modularities=modularity_scores,
#     graph=self.problem.G,
#     base_path=img_dir(self.problem_name, Advantage.name),
#     solver=Advantage.name,
# )
