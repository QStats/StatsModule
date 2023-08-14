import numpy as np
from QHyper.problems.community_detection import \
    CommunityDetectionProblem as CDP

from paths import csv_path, img_dir
from Printer.printer import Printer
from QStats.solvers.louvain.louvain import Louvain
from util import MOD_SCORE, SAMPLE


class LouvainSolution:
    def __init__(self, problem: CDP, problem_name: str) -> None:
        self.problem = problem
        self.problem_name = problem_name

    def compute(
        self,
        n_runs: int,
        communities_res: float,
        modularity_res: float,
        id: int | str,
        n_jobs: int = 4,
    ) -> np.ndarray:
        res = Louvain.run_parallel(
            n_runs,
            self.problem.G,
            communities_res,
            modularity_res,
            n_jobs=n_jobs,
        )

        Printer.csv_from_array(
            res, "w", csv_path(id, self.problem_name, Louvain.name)
        )
        Printer.draw_samples_modularities(
            samples=res[SAMPLE],
            mod_scores=res[MOD_SCORE],
            matrix_res=np.array([[0] * len(res[SAMPLE])]),
            score_res=np.array([[0] * len(res[SAMPLE])]),
            graph=self.problem.G,
            base_path=img_dir(id, self.problem_name, Louvain.name),
            solver=Louvain.name,
        )

        return res
