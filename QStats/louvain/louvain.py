import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.community import louvain_communities

from util import G

from ..scorer.scorer import Scorer


class LouvainHelper:
    @staticmethod
    def louvain_communities_to_sample_like(lcda: list) -> dict:
        sample_like = {
            node_i: comm_i
            for comm_i, comms_set in enumerate(lcda)
            for node_i in comms_set
        }
        return dict(sorted(sample_like.items()))


class Louvain:
    d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
    solver = "louvain"

    @staticmethod
    def run(idx: int, c_resolution: float, m_resolution: float, graph: nx.Graph = G):
        start = time.time()
        lcda = louvain_communities(graph, resolution=c_resolution)
        end = time.time()
        total_time = end - start

        mod = Scorer.score_modularity(graph, lcda, resolution=m_resolution)
        sample = LouvainHelper.louvain_communities_to_sample_like(lcda)
        return idx, len(lcda), sample, mod, 34.55, total_time

    @staticmethod
    def run_parallel(n_runs: int, c_resolution: float, m_resolution: float):
        types = np.dtype(
            [(a, d) for a, d in zip(Louvain.d_alias, Louvain.d_types)]
        )
        dims = (n_runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        res_lcda = Parallel(n_jobs=4)(
            delayed(Louvain.run)(i, c_resolution, m_resolution) for i in range(n_runs)
        )

        for idx, el in enumerate(res_lcda):
            arr[idx] = el

        return arr
