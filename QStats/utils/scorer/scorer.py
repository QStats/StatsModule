import networkx as nx
from networkx.algorithms.community import modularity


class Scorer:
    @staticmethod
    def score_modularity(
        graph: nx.Graph, communities_partitions: list, resolution: float
    ) -> float:
        try:
            mod = modularity(
                graph,
                communities=communities_partitions,
                resolution=resolution,
            )
        except Exception as e:
            ERROR_MOD = -1
            mod = ERROR_MOD
            print(f"[{__file__} exception] {e}")

        return mod
