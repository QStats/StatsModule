import networkx as nx
import numpy as np
from demo.network_community_detection.utils.utils import (
    communities_from_sample,
)
from dwave.system import DWaveSampler, EmbeddingComposite
from QHyper.util import QUBO
from scorer.scorer import Scorer

from util import G


class Advantage:
    d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
    solver = "adv"

    def decode_solution(self, problem, sample: dict) -> dict:
        return {
            int(str(key)[len("x") :]): val
            for key, val in problem.sort_encoded_solution(sample).items()
        }

    @staticmethod
    def run(
        self,
        bqm: QUBO,
        runs: int,
        resolution: float,
        communities: int = 2,
        topology_type: str = "pegasus",
        graph: nx.Graph = G,
    ):
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
            communities_partition = communities_from_sample(
                solution, communities
            )
            mod = Scorer.score_modularity(
                graph, communities_partition, resolution
            )

            arr[i] = (
                i,
                len(communities_partition),
                solution,
                mod,
                energy,
                run_time,
            )

        return arr
