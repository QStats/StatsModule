import dimod
import networkx.algorithms.community as nx_comm
import numpy as np
from demo.network_community_detection.utils.utils import (
    communities_from_sample,
)
from dwave.system import DWaveSampler, EmbeddingComposite
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.problems.base import Problem
from QHyper.util import QUBO

name = "karate"
folder = "demo/network_community_detection/demo_output"
solution_file = f"{folder}/csv_files/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/csv_files/{name}_adv_decoded_solution.csv"
# img_solution_path = f"{folder}/{name}_adv.png"


problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=resolution), communities=1
)
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


class Advantage:
    def __init__(self) -> None:
        self.d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
        self.d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
        self.solver = "adv"

    @staticmethod
    def run(self, bqm: QUBO, runs: int, resolution: float, topology_type: str = "pegasus"):
        types = np.dtype([(a, d) for a, d in zip(self.d_alias, self.d_types)])
        dims = (runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        sampler = DWaveSampler(solver=dict(topology__type=topology_type))
        for i in range(runs):
            sampleset = EmbeddingComposite(sampler).sample(bqm)
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

        return arr
