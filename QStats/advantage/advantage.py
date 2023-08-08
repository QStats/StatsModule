import networkx as nx
import numpy as np
from demo.network_community_detection.utils.utils import (
    communities_from_sample,
)
from dwave.system import DWaveSampler, EmbeddingComposite
from QHyper.problems.base import Problem
from QHyper.util import QUBO

from util import G

from ..scorer.scorer import Scorer


class Advantage:
    d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
    solver = "adv"

    @staticmethod
    def decode_solution(problem: Problem, sample: dict) -> dict:
        return {
            int(str(key)[len("x") :]): val
            for key, val in problem.sort_encoded_solution(sample).items()
        }

    @staticmethod
    def run(
        bqm: QUBO,
        runs: int,
        resolution: float,
        problem: Problem,
        communities: int = 2,
        topology_type: str = "pegasus",
        graph: nx.Graph = G,
    ):
        types = np.dtype(
            [(a, d) for a, d in zip(Advantage.d_alias, Advantage.d_types)]
        )
        dims = (runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        # sampler = DWaveSampler(solver=dict(topology__type=topology_type))
        for i in range(runs):
            sampler = DWaveSampler(solver=dict(topology__type=topology_type))
            sampleset = EmbeddingComposite(sampler).sample(bqm)
            sample = sampleset.first.sample
            energy = sampleset.first.energy
            run_time = 10

            print(f"sample: {sample}")
            solution = Advantage.decode_solution(problem, sample)
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
