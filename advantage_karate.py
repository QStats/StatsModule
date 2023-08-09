import dimod
from QHyper.problems.community_detection import (CommunityDetectionProblem,
                                                 KarateClubNetwork)
from QHyper.solvers.converter import Converter

from paths import KARATE_PR_NAME
from QStats.solutions.advantage_solution import AdvantageSolution


ADV_C_RES = 1
ADV_M_RES = 0.5
ADV_RUNS = 1


problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=ADV_C_RES), communities=1
)
qubo = Converter.create_qubo(problem, [1])
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

advantage = AdvantageSolution(
    problem=problem, n_communities=2, problem_name=KARATE_PR_NAME
)
adv_res = advantage.compute(
    bqm=bqm, n_runs=ADV_RUNS, modularity_resolution=ADV_M_RES
)
