from typing import TypedDict

import numpy as np

from paths import BRAIN_PR_NAME
from QStats.solutions.louvain_solution import LouvainSolution
from util import (BRAIN_NETWORK_GRAPH, LOU_RES_ALIASES, LOU_RES_DTYPES,
                  MATRIX_RESOLUTION, MOD_SCORE, R_TIME, SAMPLE,
                  SCORE_RESOLUTION, K)

ParamGrid = TypedDict(
    "ParamGrid",
    {"resolution_grid": np.ndarray, "score_resolutions": np.ndarray},
)


class LouvainSearch:
    def __init__(self, id: int) -> None:
        self.id = id

    def search_grid(
        self, param_grid: ParamGrid, n_runs_per_param: int
    ) -> np.ndarray:
        score_resolutions = param_grid["resolution_grid"]
        modularity_resolutions = param_grid["score_resolutions"]
        if len(score_resolutions) != len(modularity_resolutions):
            raise Exception(
                "Param grid objects must be of the same length,"
                + f"got of length {len(score_resolutions)}"
                + f"and {len(modularity_resolutions)} instead"
            )
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
            problem_graph = BRAIN_NETWORK_GRAPH
            lou_sol = LouvainSolution(
                graph=problem_graph, problem_name=BRAIN_PR_NAME
            )
            runs_res = lou_sol.compute(
                n_runs=n_runs_per_param,
                communities_res=resolution_val,
                modularity_res=score_res,
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
