import dimod
from joblib import Parallel, delayed
import networkx as nx
import networkx.algorithms.community as nx_comm
from dwave.system import DWaveSampler, EmbeddingComposite
from matplotlib import pyplot as plt
import numpy as np
from demo.network_community_detection.utils.utils import (
    COLORS,
    communities_from_sample,
    write_to_file,
)
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)

name = "karate"
folder = "demo/network_community_detection/demo_output"
solution_file = f"{folder}/csv_files/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/csv_files/{name}_adv_decoded_solution.csv"
# img_solution_path = f"{folder}/{name}_adv.png"

k = 10
d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
types = np.dtype([(a, d) for a, d in zip(d_alias, d_types)])
dims = (k, 1)
arr: np.ndarray = np.zeros(dims, dtype=types)

resolution = 0.5
problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=1), communities=1
)

adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))
sampler = adv_sampler
solver = "adv"

binary_polynomial = dimod.BinaryPolynomial(
    problem.objective_function.as_dict(), dimod.BINARY
)
cqm = dimod.make_quadratic_cqm(binary_polynomial)
bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)


def decode_solution(sample: dict) -> dict:
    return {
        int(str(key)[len("x"):]): val
        for key, val in problem.sort_encoded_solution(sample).items()
    }


for i in range(k):
    sampleset = EmbeddingComposite(adv_sampler).sample(bqm)
    sample = sampleset.first.sample
    energy = sampleset.first.energy
    run_time = 10
    solution = decode_solution(sample)

    try:
        communities = communities_from_sample(solution, problem.cases + 1)
        mod = nx_comm.modularity(
            problem.G,
            communities=communities,
            resolution=resolution,
        )
    except Exception:
        mod = 0

    arr[i] = i, len(communities), solution, mod, energy, run_time


print(np.array(arr["mod_score"]).mean())
print(np.array(arr["k"]).mean())
