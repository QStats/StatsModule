import networkx as nx
from networkx.algorithms.community import modularity


class Scorer:
    @staticmethod
    def score_modularity(
        graph: nx.Graph, community_partition: list, resolution: float
    ) -> float:
        try:
            mod = modularity(
                graph,
                communities=community_partition,
                resolution=resolution,
            )
        except Exception:
            ERROR_MOD = -1
            mod = ERROR_MOD

        return mod
