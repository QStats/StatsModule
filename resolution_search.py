from typing import TypeAlias, TypedDict

import numpy as np

from util import (ADV_RES_ALIASES, ADV_RES_DTYPES, EN, MATRIX_RESOLUTION,
                  MOD_SCORE, R_TIME, SAMPLE, SCORE_RESOLUTION, K)

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
    def run_grid(param_grid: ParamGrid) -> np.array:
        da = [MATRIX_RESOLUTION, SCORE_RESOLUTION]
        dt = [np.float_, np.float_]
        n_params = len(param_grid["resolution_grid"])
        types = np.dtype(
            [(a, t) for a, t in zip(da + ADV_RES_ALIASES, dt + ADV_RES_DTYPES)]
        )
        res: np.ndarray = np.ndarray((n_params, 1), dtype=types)

        for i, (res_val, mod_res) in enumerate(zip(*param_grid.values())):
            # problem_instance = BPI.get(
            #     resolution=res_val, n_communities=1
            # )

            # bqm = BQMFactory.bqm(problem=problem_instance, weights=[1])

            # runs_res = AdvantageSolution(
            #     problem=problem_instance, n_communities=n_communtities
            # ).compute(
            #     bqm=bqm, n_runs=n_runs_per_param, score_mod_resolution=mod_res
            # )

            # np.savez("arr", runs_res=runs_res)
            npzfile = np.load("arr.npz", allow_pickle=True)
            runs_res = npzfile["runs_res"]

            tt = np.dtype([(a, d) for a, d in zip(da, dt)])
            modularity_cols: np.ndarray = np.ndarray((1, 1), dtype=tt)
            modularity_cols[::] = res_val, mod_res

            k = runs_res[K]
            samples = runs_res[SAMPLE]
            modularity_scores = runs_res[MOD_SCORE]
            samples = runs_res[SAMPLE]
            energies = runs_res[EN]
            r_times = runs_res[R_TIME]

            rv = modularity_cols[MATRIX_RESOLUTION]
            mr = modularity_cols[SCORE_RESOLUTION]

            s = np.hstack([rv, mr])

            stacked = np.hstack(
                [k, samples, modularity_scores, energies, r_times]
            )
            res_stacked = np.hstack([s, stacked])

            res_s = np.array(
                list(map(tuple, res_stacked[::])), dtype=types
            ).reshape(samples.shape[0], 1)

            res[i : i + 1] = res_s

        return res
