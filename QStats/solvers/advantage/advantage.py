from time import time

import numpy as np
from dimod import BinaryQuadraticModel as BQM
from dwave.system import DWaveSampler, EmbeddingComposite

from util import EN, R_TIME, SAMPLE


class IdGenerator:
    next_id: int = 0

    @staticmethod
    def get_next_id() -> int:
        IdGenerator.next_id += 1
        return IdGenerator.next_id - 1


class Advantage:
    name = "adv"

    @staticmethod
    def run(
        bqm: BQM, n_runs: int, topology_type: str = "pegasus"
    ) -> np.ndarray:
        aliases = [SAMPLE, EN, R_TIME]
        dtypes = [np.object_, np.float64, np.float_]
        types = np.dtype([(a, t) for a, t in zip(aliases, dtypes)])
        res: np.ndarray = np.zeros((n_runs, 1), dtype=types)
        savez_samplesets: dict = {}

        for i in range(n_runs):
            tic = time()
            sampler = DWaveSampler(solver=dict(topology__type=topology_type))
            sampleset = EmbeddingComposite(sampler).sample(bqm)
            toc = time()

            try:
                savez_samplesets[
                    "sampleset_" + str(IdGenerator.get_next_id() * i)
                ] = sampleset
            except Exception:
                pass

            sample = sampleset.first.sample
            energy = sampleset.first.energy
            run_time = toc - tic

            res[i] = sample, energy, run_time

        np.savez(f"samplesets_{IdGenerator.next_id}", **savez_samplesets)
        del savez_samplesets
        return res
