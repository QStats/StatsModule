from QHyper.problems.community_detection import CommunityDetectionProblem
from networkx.algorithms.community import louvain_communities, modularity


class ModularityFunction:
    def __init__(self, problem: CommunityDetectionProblem) -> None:
        self.problem = problem


