import numpy as np
from QHyper.problems.base import Problem

from Printer.printer import Printer
from QStats.solvers.louvain.louvain import Louvain

solver = Louvain.solver
name = "karate"
folder = f"demo/network_community_detection/demo_output/{solver}"
solution_file = f"{folder}/csv_files/{name}_{solver}_solution.csv"
path = f"{folder}/{name}_{solver}"


class LouvainSolution:
    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def compute(
        self, n_runs: int, communities_res: float, modularity_res: float
    ) -> np.ndarray:
        res = Louvain.run_parallel(
            n_runs, self.problem.G, communities_res, modularity_res
        )

        samples = res["sample"]
        modularities = res["mod_score"]

        Printer.csv_from_array(res, solution_file)
        Printer.draw_samples_modularities(samples, modularities, path, solver)

        return res
