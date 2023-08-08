import networkx as nx
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from QHyper.problems.base import Problem
from QHyper.util import QUBO

from QStats.utils.converter.converter import Converter as C
from QStats.utils.scorer.scorer import Scorer
from util import G


class Advantage:
    d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
    solver = "adv"

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

        for i in range(runs):
            sampler = DWaveSampler(solver=dict(topology__type=topology_type))
            sampleset = EmbeddingComposite(sampler).sample(bqm)
            sample = sampleset.first.sample
            energy = sampleset.first.energy
            run_time = 10

            solution = C.AdvantageHelper.decode_solution(problem, sample)
            communities_partition = C.AdvantageHelper.communities_from_sample(
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
