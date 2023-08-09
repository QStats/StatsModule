import numpy as np
from QHyper.problems.base import Problem

from Printer.printer import Printer
from QStats.solvers.louvain.louvain import Louvain
from util import MOD_SCORE, SAMPLE

solver = Louvain.solver
name = "karate"
folder = f"demo/network_community_detection/demo_output/{solver}"
solution_file = f"{folder}/csv_files/{name}_{solver}_solution.csv"
path = f"{folder}/{name}_{solver}"


class LouvainSolution:
    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def compute(
        self,
        n_runs: int,
        communities_res: float,
        modularity_res: float,
        n_jobs: int = 4,
    ) -> np.ndarray:
        res = Louvain.run_parallel(
            n_runs,
            self.problem.G,
            communities_res,
            modularity_res,
            n_jobs=n_jobs,
        )

        Printer.csv_from_array(res, solution_file)
        Printer.draw_samples_modularities(
            res[SAMPLE], res[MOD_SCORE], path, solver
        )

        return res
