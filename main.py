import dimod
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
    BrainNetwork,
)
from QHyper.solvers.converter import Converter

from QStats.solutions.advantage_solution import AdvantageSolution
from QStats.solutions.louvain_solution import LouvainSolution
from paths import IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE, BRAIN_PR_NAME


LAGRANGE_MULT = 10

LOUVAIN_C_RES = 0.5
LOUVAIN_M_RES = 0.5
LOUVAIN_RUNS = 10

ADV_C_RES = 1
ADV_M_RES = 0.5
ADV_RUNS = 1


problem = CommunityDetectionProblem(
    BrainNetwork(
        input_data_dir=IN_BRAIN_NETWORK_DIR,
        input_data_name=IN_BRAIN_NETWORK_FILE,
        resolution=LOUVAIN_C_RES,
    )
)

# problem = CommunityDetectionProblem(
#     network_data=KarateClubNetwork(resolution=ADV_C_RES), communities=1
# )
# qubo = Converter.create_qubo(problem, [1])
# bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

# advantage = AdvantageSolution(problem=problem, n_communities=2)
# adv_res = advantage.compute(
#     bqm=bqm, n_runs=ADV_RUNS, modularity_resolution=ADV_M_RES
# )


louvain = LouvainSolution(problem=problem, problem_name=BRAIN_PR_NAME)
louvain_res = louvain.compute(
    n_runs=LOUVAIN_RUNS,
    communities_res=LOUVAIN_C_RES,
    modularity_res=LOUVAIN_M_RES,
)
