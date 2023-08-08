from Printer.printer import Printer
from QStats.solvers.louvain.louvain import Louvain

solver = Louvain.solver
name = "karate"
folder = f"demo/network_community_detection/demo_output/{solver}"
solution_file = f"{folder}/csv_files/{name}_{solver}_solution.csv"
path = f"{folder}/{name}_{solver}"


COMMUNITIES_RES = 0.5
MODULARITY_RES = 0.5
N_RUNS = 10

res = Louvain.run_parallel(N_RUNS, COMMUNITIES_RES, MODULARITY_RES)

samples = res["sample"]
modularities = res["mod_score"]

Printer.csv_from_array(res, solution_file)
Printer.draw_samples_modularities(samples, modularities, path, solver)
