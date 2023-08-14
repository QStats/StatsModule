from QHyper.problems.community_detection import (CommunityDetectionProblem,
                                                 KarateClubNetwork)

from paths import KARATE_PR_NAME
from QStats.solutions.louvain_solution import LouvainSolution

ID = 0

C_RES = 0.5
M_RES = 0.5
N_RUNS = 10


problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=C_RES), communities=1
)

louvain = LouvainSolution(problem=problem, problem_name=KARATE_PR_NAME)
louvain_res = louvain.compute(
    n_runs=N_RUNS, communities_res=C_RES, modularity_res=M_RES, id=ID
)
