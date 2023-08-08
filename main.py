from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)

from QStats.models.bqm import BQM
from QStats.solutions.advantage_solution import AdvantageSolution
from QStats.solutions.louvain_solution import LouvainSolution

LAGRANGE_MULT = 10

LOUVAIN_C_RES = 0.5
LOUVAIN_M_RES = 0.5
LOUVAIN_RUNS = 10

ADV_C_RES = 1
ADV_M_RES = 0.5
ADV_RUNS = 10

problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=ADV_C_RES), communities=1
)
bqm = BQM.bqm(problem, LAGRANGE_MULT)

adv_res = AdvantageSolution.compute(
    bqm=bqm,
    problem=problem,
    n_runs=ADV_RUNS,
    modularity_res=ADV_M_RES,
    communities=2,
)
louvain_res = LouvainSolution.compute(
    n_runs=LOUVAIN_RUNS,
    communities_res=LOUVAIN_C_RES,
    modularity_res=LOUVAIN_M_RES,
)


print(f"adv_res:\n{adv_res}\n\n")
print(f"louvain_res:\n{louvain_res}\n\n")
