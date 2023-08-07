from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)

problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=resolution), communities=1
)
