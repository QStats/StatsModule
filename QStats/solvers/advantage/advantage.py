from time import time

import numpy as np
from dimod import BinaryQuadraticModel as BQM
from dwave.system import DWaveSampler, EmbeddingComposite

from util import EN, MOD_SCORE, R_TIME, SAMPLE, K


class Advantage:
    d_aliases = [K, SAMPLE, MOD_SCORE, EN, R_TIME]
    d_types = [np.float_, np.object_, np.float64, np.float_, np.float_]
    name = "adv"

    @staticmethod
    def run(
        bqm: BQM, n_runs: int, topology_type: str = "pegasus"
    ) -> np.ndarray:
        da = [SAMPLE, EN, R_TIME]
        dt = [np.object_, np.float64, np.float_]
        types = np.dtype([(a, t) for a, t in zip(da, dt)])
        dims = (n_runs, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        for i in range(n_runs):
            tic = time()
            sampler = DWaveSampler(solver=dict(topology__type=topology_type))
            sampleset = EmbeddingComposite(sampler).sample(bqm)
            toc = time()

            sample = sampleset.first.sample
            energy = sampleset.first.energy
            run_time = toc - tic

            arr[i] = sample, energy, run_time

        return arr
