import dimod
import numpy as np
from QHyper.problems.community_detection import \
    CommunityDetectionProblem as CDP

from QStats.utils.converter.converter import Converter as C
from QStats.utils.scorer.scorer import Scorer
from util import ADV_RES_TYPES, EN, MOD_SCORE, R_TIME, SAMPLE, K


class AdvantageSolution:
    def __init__(self, problem: CDP, n_communities: int) -> None:
        self.problem = problem
        self.n_communities = n_communities

    def compute(
        self, bqm: dimod.BQM, n_runs: int, score_mod_resolution: float
    ) -> np.ndarray:
        # adv_res = Advantage.run(bqm=bqm, n_runs=n_runs)
        adv_res = np.array([])

        raw_samples = adv_res[SAMPLE]
        energies = adv_res[EN]
        r_times = adv_res[R_TIME]

        scores = self.process_samples(raw_samples, score_mod_resolution)
        k = scores[K]
        samples = scores[SAMPLE]
        modularity_scores = scores[MOD_SCORE]

        res_stacked = np.hstack(
            [k, samples, modularity_scores, energies, r_times]
        )

        res = np.array(
            list(map(tuple, res_stacked[::])), dtype=ADV_RES_TYPES
        ).reshape(samples.shape[0], 1)

        return res

    def process_samples(
        self, adv_samples: np.ndarray, modularity_resolution: float
    ) -> np.ndarray:
        da = [K, SAMPLE, MOD_SCORE]
        dt = [np.float_, np.object_, np.float_]
        types = np.dtype([(a, d) for a, d in zip(da, dt)])
        n_samples = adv_samples.shape[0]
        dims = (n_samples, 1)
        arr: np.ndarray = np.zeros(dims, dtype=types)

        for i in range(n_samples):
            sample = adv_samples[i][0]
            solution = C.AdvantageHelper.decode_solution(self.problem, sample)
            communities_partition = C.AdvantageHelper.communities_from_sample(
                solution, self.n_communities
            )
            mod = Scorer.score_modularity(
                self.problem.G, communities_partition, modularity_resolution
            )
            arr[i] = (len(communities_partition), solution, mod)

        return arr
