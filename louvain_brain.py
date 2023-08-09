from QHyper.problems.community_detection import (BrainNetwork,
                                                 CommunityDetectionProblem)

from paths import BRAIN_PR_NAME, IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE
from QStats.solutions.louvain_solution import LouvainSolution


C_RES = 0.11
M_RES = 1
N_RUNS = 10


problem = CommunityDetectionProblem(
    BrainNetwork(
        input_data_dir=IN_BRAIN_NETWORK_DIR,
        input_data_name=IN_BRAIN_NETWORK_FILE,
        resolution=C_RES,
    )
)

louvain = LouvainSolution(problem=problem, problem_name=BRAIN_PR_NAME)
louvain_res = louvain.compute(
    n_runs=N_RUNS,
    communities_res=C_RES,
    modularity_res=M_RES,
)
