from typing import Callable

import numpy as np

from paths import solver_dir
from QStats.provider.provider import BQM
from QStats.search.base import ParamGrid, Search
from QStats.solutions.advantage_solution import AdvantageSolution
from util import (ADV_RES_ALIASES, ADV_RES_DTYPES, EN, MATRIX_RESOLUTION,
                  MOD_SCORE, R_TIME, SAMPLE, SCORE_RESOLUTION, K)
from Utils.Printer.printer import Printer


class AdvantageSearch(Search):
    def __init__(
        self,
        id: int | str,
        problem_instance_callable: Callable,
        problem_name: str,
    ) -> None:
        self.id = id
        self.problem_instance_callable = problem_instance_callable
        self.problem_name = problem_name

    def search_grid(
        self, param_grid: ParamGrid, n_runs_per_param: int, n_communities: int
    ) -> np.ndarray:
        score_resolutions = param_grid["resolution_grid"]
        modularity_resolutions = param_grid["score_resolutions"]
        self._check_param_grid_len(score_resolutions, modularity_resolutions)
        n_params = len(score_resolutions)

        grid_types = np.dtype(
            [
                (a, t)
                for a, t in zip(
                    [MATRIX_RESOLUTION, SCORE_RESOLUTION] + ADV_RES_ALIASES,
                    [np.float_, np.float_] + ADV_RES_DTYPES,
                )
            ]
        )
        dims = (n_params * n_runs_per_param, 1)
        result: np.ndarray = np.zeros(dims, dtype=grid_types)
        BIN_OFFSET = -1 if n_communities == 2 else 0

        for i, (resolution_val, score_res) in enumerate(
            zip(score_resolutions, modularity_resolutions)
        ):
            problem = self.problem_instance_callable(
                n_communities=n_communities + BIN_OFFSET,
                resolution=resolution_val,
            )
            adv_sol = AdvantageSolution(
                problem=problem, n_communities=n_communities
            )
            # TODO weights
            bqm = BQM.bqm(problem=problem, weights=[1])
            runs_res = adv_sol.compute(
                bqm=bqm,
                n_runs=n_runs_per_param,
                score_mod_resolution=score_res,
            )

            save_dir = solver_dir(self.id, self.problem_name, "adv")
            file_path = f"{save_dir}/run_res_{i}"
            with Printer.safe_open(file_path, "w"):
                try:
                    np.savez(file_path, res=runs_res)
                except Exception:
                    np.savez("res", res=runs_res)

            k = runs_res[K]
            samples = runs_res[SAMPLE]
            modularity_scores = runs_res[MOD_SCORE]
            energies = runs_res[EN]
            r_times = runs_res[R_TIME]

            rv = np.array([[resolution_val] * n_runs_per_param]).transpose()
            sr = np.array([[score_res] * n_runs_per_param]).transpose()

            col_stack = np.hstack(
                [rv, sr, k, samples, modularity_scores, energies, r_times]
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
