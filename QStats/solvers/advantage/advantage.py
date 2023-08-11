from time import time

import numpy as np
from dimod import BinaryQuadraticModel as BQM
from dwave.system import DWaveSampler, EmbeddingComposite

from util import EN, R_TIME, SAMPLE


class Advantage:
    name = "adv"

    @staticmethod
    def run(
        bqm: BQM, n_runs: int, topology_type: str = "pegasus"
    ) -> np.ndarray:
        da = [SAMPLE, EN, R_TIME]
        dt = [np.object_, np.float64, np.float_]
        types = np.dtype([(a, t) for a, t in zip(da, dt)])
        dims = (n_runs, 1)

        res: np.ndarray = np.zeros(dims, dtype=types)

        for i in range(n_runs):
            tic = time()
            sampler = DWaveSampler(solver=dict(topology__type=topology_type))
            sampleset = EmbeddingComposite(sampler).sample(bqm)
            toc = time()

            sample = sampleset.first.sample
            energy = sampleset.first.energy
            run_time = toc - tic

            res[i] = sample, energy, run_time

        return res
