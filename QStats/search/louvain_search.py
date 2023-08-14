import networkx as nx
import numpy as np

from QStats.search.base import ParamGrid, Search
from QStats.solvers.louvain.louvain import Louvain
from util import (LOU_RES_ALIASES, LOU_RES_DTYPES, MATRIX_RESOLUTION,
                  MOD_SCORE, R_TIME, SAMPLE, SCORE_RESOLUTION, K)


class LouvainSearch(Search):
    def __init__(self, id: int, graph: nx.Graph) -> None:
        self.id = id
        self.graph = graph

    def search_grid(
        self, param_grid: ParamGrid, n_runs_per_param: int, n_jobs: int = 4
    ) -> np.ndarray:
        score_resolutions = param_grid["resolution_grid"]
        modularity_resolutions = param_grid["score_resolutions"]
        self._check_param_grid_len(score_resolutions, modularity_resolutions)
        n_params = len(score_resolutions)

        grid_types = np.dtype(
            [
                (a, t)
                for a, t in zip(
                    [MATRIX_RESOLUTION, SCORE_RESOLUTION] + LOU_RES_ALIASES,
                    [np.float_, np.float_] + LOU_RES_DTYPES,
                )
            ]
        )
        dims = (n_params * n_runs_per_param, 1)
        result: np.ndarray = np.zeros(dims, dtype=grid_types)

        for i, (resolution_val, score_res) in enumerate(
            zip(score_resolutions, modularity_resolutions)
        ):
            runs_res = Louvain.run_parallel(
                n_runs=n_runs_per_param,
                graph=self.graph,
                communities_resolution=resolution_val,
                modularity_resolution=score_res,
                n_jobs=n_jobs,
            )

            k = runs_res[K]
            samples = runs_res[SAMPLE]
            modularity_scores = runs_res[MOD_SCORE]
            r_times = runs_res[R_TIME]

            rv = np.array([[resolution_val] * n_runs_per_param]).transpose()
            sr = np.array([[score_res] * n_runs_per_param]).transpose()

            col_stack = np.hstack(
                [rv, sr, k, samples, modularity_scores, r_times]
            )

            resolution_results = np.array(
                list([[tuple(row)] for row in col_stack]), dtype=grid_types
            )

            start_idx = i * n_runs_per_param
            end_idx = start_idx + n_runs_per_param
            result[start_idx:end_idx, :] = resolution_results[
                0:n_runs_per_param, :
            ]

        return result
