import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.community import louvain_communities

from QStats.utils.converter.converter import Converter as C
from QStats.utils.scorer.scorer import Scorer
from util import LOU_RES_TYPES


class Louvain:
    name = "louvain"

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
        dims = (n_runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=LOU_RES_TYPES)

        res_lcda = Parallel(n_jobs=n_jobs)(
            delayed(Louvain.run)(
                graph, communities_resolution, modularity_resolution
            )
            for _ in range(n_runs)
        )

        for idx, el in enumerate(res_lcda):
            arr[idx] = el

        print(arr)
        return arr
