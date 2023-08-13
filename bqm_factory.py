import dimod
from QHyper.problems.community_detection import BrainNetwork
from QHyper.problems.community_detection import CommunityDetectionProblem
from QHyper.problems.community_detection import (
    CommunityDetectionProblem as CDP,
)
from QHyper.solvers.converter import Converter

from paths import IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE


class BQMFactory:
    @staticmethod
    def bqm(problem: CDP, weights: list[float]):
        qubo = Converter.create_qubo(problem, weights)
        return dimod.BinaryQuadraticModel.from_qubo(qubo)


class BrainProblemInstance:
    @staticmethod
    def get(resolution: float, n_communities: int) -> CDP:
        return CommunityDetectionProblem(
            BrainNetwork(
                input_data_dir=IN_BRAIN_NETWORK_DIR,
                input_data_name=IN_BRAIN_NETWORK_FILE,
                resolution=resolution,
            ),
            communities=n_communities,
        )
