import networkx as nx
import numpy as np

from QStats.solvers.louvain.louvain import Louvain


class LouvainSolution:
    def __init__(self, graph: nx.Graph, problem_name: str) -> None:
        self.graph = graph
        self.problem_name = problem_name

    def compute(
        self,
        n_runs: int,
        communities_res: float,
        modularity_res: float,
        n_jobs: int = 4,
    ) -> np.ndarray:
        res = Louvain.run_parallel(
            n_runs,
            self.graph,
            communities_res,
            modularity_res,
            n_jobs=n_jobs,
        )

        return res
