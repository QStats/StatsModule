import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.community import louvain_communities

from QStats.utils.converter.converter import Converter as C
from QStats.utils.scorer.scorer import Scorer
from util import MOD_SCORE, R_TIME, SAMPLE, K


class Louvain:
    d_aliases = [K, SAMPLE, MOD_SCORE, R_TIME]
    d_types = [np.float_, np.object_, np.float64, np.float_]
    solver = "louvain"

    @staticmethod
    def run(
        graph: nx.Graph, c_resolution: float, m_resolution: float
    ) -> tuple[float, dict, float, float]:
        start = time.time()
        lcda = louvain_communities(graph, resolution=c_resolution)
        end = time.time()
        total_time = end - start

        mod = Scorer.score_modularity(graph, lcda, resolution=m_resolution)
        sample = C.LouvainHelper.louvain_communities_to_sample_like(lcda)
        return len(lcda), sample, mod, total_time

    @staticmethod
    def run_parallel(
        n_runs: int,
        graph: nx.Graph,
        communities_resolution: float,
        modularity_resolution: float,
        n_jobs: int = 4,
    ) -> np.ndarray:
        types = np.dtype(
            [(a, d) for a, d in zip(Louvain.d_aliases, Louvain.d_types)]
        )
        dims = (n_runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        res_lcda = Parallel(n_jobs=n_jobs)(
            delayed(Louvain.run)(
                graph, communities_resolution, modularity_resolution
            )
            for _ in range(n_runs)
        )

        for idx, el in enumerate(res_lcda):
            arr[idx] = el

        return arr
