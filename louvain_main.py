from Printer.printer import Printer
from QStats.louvain.louvain import Louvain


solver = Louvain.solver
name = "karate"
folder = f"demo/network_community_detection/demo_output/{solver}"
solution_file = f"{folder}/csv_files/{name}_{solver}_solution.csv"
path = f"{folder}/{name}_{solver}"


C_RES = 0.5
M_RES = 0.5
N_RUNS = 10

res = Louvain.run_parallel(N_RUNS, C_RES, M_RES)
Printer.csv_from_array(res, solution_file)

samples = res["sample"]
modularities = res["mod_score"]
Printer.draw_samples_modularities(samples, modularities, path, solver)
