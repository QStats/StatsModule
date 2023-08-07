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
from typing import Callable
from networkx.algorithms.community import louvain_communities, modularity
import networkx as nx


name = "karate"
folder = "demo/network_community_detection/demo_output"
solution_file = f"{folder}/csv_files/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/csv_files/{name}_adv_decoded_solution.csv"
# img_solution_path = f"{folder}/{name}_adv.png"


class Advantage:
    d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
    solver = "adv"

    def decode_solution(self, problem, sample: dict) -> dict:
        return {
            int(str(key)[len("x"):]): val
            for key, val in problem.sort_encoded_solution(sample).items()
        }

    @staticmethod
    def score_modularity(graph: nx.Graph, community_partition: list, resolution: float = 0.5) -> float:
        try:
            mod = modularity(
                graph,
                communities=community_partition,
                resolution=resolution,
            )
        except Exception:
            ERROR_MOD = -1
            mod = ERROR_MOD

        return mod

    @staticmethod
    def run(self, bqm: QUBO, runs: int, communities: int = 2, topology_type: str = "pegasus"):
        types = np.dtype([(a, d) for a, d in zip(self.d_alias, self.d_types)])
        dims = (runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        sampler = DWaveSampler(solver=dict(topology__type=topology_type))
        for i in range(runs):
            sampleset = EmbeddingComposite(sampler).sample(bqm)
            sample = sampleset.first.sample
            energy = sampleset.first.energy
            run_time = 10

            solution = self.decode_solution(sample)
            communities_partition = communities_from_sample(solution, communities)
            mod = scor

            arr[i] = i, len(communities_partition), solution, mod, energy, run_time

        return arr
