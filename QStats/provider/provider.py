import dimod
from QHyper.problems.community_detection import BrainNetwork
from QHyper.problems.community_detection import \
    CommunityDetectionProblem as CDP
from QHyper.problems.community_detection import KarateClubNetwork
from QHyper.solvers.converter import Converter

from paths import IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE


class BQM:
    @staticmethod
    def bqm(problem: CDP, weights: list[float]):
        qubo = Converter.create_qubo(problem, weights)
        return dimod.BinaryQuadraticModel.from_qubo(qubo)


class ProblemInstance:
    @staticmethod
    def brain_problem(n_communities: int, resolution: float) -> CDP:
        return CDP(
            network_data=BrainNetwork(
                IN_BRAIN_NETWORK_DIR,
                IN_BRAIN_NETWORK_FILE,
                resolution=resolution,
            ),
            communities=n_communities,
        )

    @staticmethod
    def karate_problem(n_communities: int, resolution: float) -> CDP:
        return CDP(
            network_data=KarateClubNetwork(resolution=resolution),
            communities=n_communities,
        )
