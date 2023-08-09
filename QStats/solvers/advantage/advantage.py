from time import time

import numpy as np
from dimod import BinaryQuadraticModel as BQM
from dwave.system import DWaveSampler, EmbeddingComposite


class Advantage:
    d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
    d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
    solver = "adv"

    @staticmethod
    def run(
        bqm: BQM, n_runs: int, topology_type: str = "pegasus"
    ) -> np.ndarray:
        aa = ["ord", "sample", "energy", "run_time"]
        tt = [np.int_, np.object_, np.float64, np.float_]
        types = np.dtype([(a, d) for a, d in zip(aa, tt)])
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

            arr[i] = i, sample, energy, run_time

        return arr
