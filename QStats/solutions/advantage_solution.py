import dimod
import numpy as np
from QHyper.problems.base import Problem

from Printer.printer import Printer
from QStats.solvers.advantage.advantage import Advantage

solver = Advantage.solver
name = "karate"
folder = f"demo/network_community_detection/demo_output/{solver}"
solution_file = f"{folder}/csv_files/{name}_{solver}_solution.csv"
path = f"{folder}/{name}_{solver}"


class AdvantageSolution:
    @staticmethod
    def compute(
        bqm: dimod.BQM,
        problem: Problem,
        n_runs: int,
        modularity_res: float,
        communities: int,
    ) -> np.ndarray:
        res = Advantage.run(
            bqm=bqm,
            runs=n_runs,
            modularity_resolution=modularity_res,
            problem=problem,
            communities=communities,
        )

        samples = res["sample"]
        modularities = res["mod_score"]

        Printer.csv_from_array(res, solution_file)
        Printer.draw_samples_modularities(samples, modularities, path, solver)

        return res
