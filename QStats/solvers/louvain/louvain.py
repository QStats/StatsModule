import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.community import louvain_communities

from QStats.utils.converter.converter import Converter as C
from QStats.utils.scorer.scorer import Scorer
from util import G


class Louvain:
    d_alias = ["ord", "k", "sample", "mod_score", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_]
    solver = "louvain"

    @staticmethod
    def run(
        idx: int, c_resolution: float, m_resolution: float, graph: nx.Graph = G
    ):
        start = time.time()
        lcda = louvain_communities(graph, resolution=c_resolution)
        end = time.time()
        total_time = end - start

        mod = Scorer.score_modularity(graph, lcda, resolution=m_resolution)
        sample = C.LouvainHelper.louvain_communities_to_sample_like(lcda)
        return idx, len(lcda), sample, mod, total_time

    @staticmethod
    def run_parallel(
        n_runs: int,
        communities_resolution: float,
        modularity_resolution: float,
    ):
        types = np.dtype(
            [(a, d) for a, d in zip(Louvain.d_alias, Louvain.d_types)]
        )
        dims = (n_runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        res_lcda = Parallel(n_jobs=4)(
            delayed(Louvain.run)(
                i, communities_resolution, modularity_resolution
            )
            for i in range(n_runs)
        )

        for idx, el in enumerate(res_lcda):
            arr[idx] = el

        return arr
