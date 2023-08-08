from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)

from Printer.printer import Printer
from QStats.models.bqm import BQM
from QStats.solvers.advantage.advantage import Advantage

solver = Advantage.solver
name = "karate"
folder = f"demo/network_community_detection/demo_output/{solver}"
solution_file = f"{folder}/csv_files/{name}_{solver}_solution.csv"
path = f"{folder}/{name}_{solver}"


COMMUNITIES_RES = 1
MODULARITY_RES = 0.5
N_RUNS = 10

problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=COMMUNITIES_RES), communities=1
)
bqm = BQM.bqm(problem, 10)

res = Advantage.run(
    bqm=bqm,
    runs=N_RUNS,
    modularity_resolution=MODULARITY_RES,
    problem=problem,
    communities=2,
)

samples = res["sample"]
modularities = res["mod_score"]

Printer.csv_from_array(res, solution_file)
Printer.draw_samples_modularities(samples, modularities, path, solver)
