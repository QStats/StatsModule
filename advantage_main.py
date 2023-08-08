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
path = f"{folder}/{name}_{solver}.png"


resolution = 0.5
RESOLUTION = 1
problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=RESOLUTION), communities=1
)

bqm = BQM.bqm(problem, 10)

res = Advantage.run(bqm, 1, 0.5, problem, 2)

Printer.csv_from_array(res, solution_file)

samples = res["sample"]
modularities = res["mod_score"]
Printer.draw_samples_modularities(samples, modularities, path, solver)
