from typing import TypeAlias, TypedDict

import numpy as np
from QStats.solutions.advantage_solution import AdvantageSolution
from bqm_factory import BQMFactory, BrainProblemInstance as BPI
from paths import solver_dir

from util import (
    ADV_RES_ALIASES,
    ADV_RES_DTYPES,
    ADV_RES_TYPES,
    EN,
    MATRIX_RESOLUTION,
    MOD_SCORE,
    R_TIME,
    SAMPLE,
    SCORE_RESOLUTION,
    K,
)

from Printer.printer import Printer

ResolutionGrid: TypeAlias = np.ndarray
RunsPerValue: TypeAlias = np.ndarray
ModularityScoreResolution: TypeAlias = np.ndarray

ParamGrid = TypedDict(
    "ParamGrid",
    {
        "resolution_grid": ResolutionGrid,
        "modularity_score_resoltion": ModularityScoreResolution,
    },
)


class Search:
    @staticmethod
    def run_grid(
        param_grid: ParamGrid, n_runs_per_param: int, n_communities: int
    ) -> np.array:
        score_resolutions = param_grid["resolution_grid"]
        modularity_resolutions = param_grid["modularity_score_resoltion"]
        if len(score_resolutions) != len(modularity_resolutions):
            raise Exception(
                "Param grid objects must be of the same length,"
                + f"got of length {len(score_resolutions)}"
                + f"and {len(modularity_resolutions)} instead"
            )
        n_params = len(score_resolutions)

        types = np.dtype(
            [
                (a, t)
                for a, t in zip(
                    [MATRIX_RESOLUTION, SCORE_RESOLUTION] + ADV_RES_ALIASES,
                    [np.float_, np.float_] + ADV_RES_DTYPES,
                )
            ]
        )
        dims = (n_params * n_runs_per_param, 1)
        res: np.ndarray = np.zeros(dims, dtype=types)

        for i, (res_val, mod_res) in enumerate(
            zip(score_resolutions, modularity_resolutions)
        ):
            problem_instance = BPI.get(resolution=res_val, n_communities=1)

            bqm = BQMFactory.bqm(problem=problem_instance, weights=[1])

            runs_res = AdvantageSolution(
                problem=problem_instance, n_communities=n_communities
            ).compute(
                bqm=bqm, n_runs=n_runs_per_param, score_mod_resolution=mod_res
            )

            save_dir = solver_dir(4, "brain", "adv")
            with Printer.safe_open(f"{save_dir}/run_res_{i}", "w"):
                try:
                    np.savez(f"{save_dir}/run_res_{i}", runs_res=runs_res)
                except Exception:
                    np.savez("arr", runs_res=runs_res)

            k = runs_res[K]
            samples = runs_res[SAMPLE]
            modularity_scores = runs_res[MOD_SCORE]
            samples = runs_res[SAMPLE]
            energies = runs_res[EN]
            r_times = runs_res[R_TIME]

            rv = np.array([[res_val] * n_runs_per_param]).transpose()
            mr = np.array([[mod_res] * n_runs_per_param]).transpose()

            res_stacked = np.hstack(
                [rv, mr, k, samples, modularity_scores, energies, r_times]
            )
            res_s = np.array(
                list([[tuple(row)] for row in res_stacked]), dtype=types
            )

            start_idx = i * n_runs_per_param
            end_idx = start_idx + n_runs_per_param
            res[start_idx:end_idx, :] = res_s[0:n_runs_per_param, :]

        return res
