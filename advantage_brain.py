import dimod
from QHyper.problems.community_detection import (BrainNetwork,
                                                 CommunityDetectionProblem)
from QHyper.solvers.converter import Converter

from paths import BRAIN_PR_NAME, IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE
from QStats.solutions.advantage_solution import AdvantageSolution


C_RES = 1.2
M_RES = 1
N_RUNS = 1


problem = CommunityDetectionProblem(
    BrainNetwork(
        input_data_dir=IN_BRAIN_NETWORK_DIR,
        input_data_name=IN_BRAIN_NETWORK_FILE,
        resolution=C_RES,
    ), communities=1
)

qubo = Converter.create_qubo(problem, [1])
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

advantage = AdvantageSolution(
    problem=problem, n_communities=2, problem_name=BRAIN_PR_NAME
)
adv_res = advantage.compute(
    bqm=bqm, n_runs=N_RUNS, modularity_resolution=M_RES
)
