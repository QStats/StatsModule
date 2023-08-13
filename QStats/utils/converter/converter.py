from QHyper.problems.community_detection import (
    CommunityDetectionProblem as CDP,
)


class Converter:
    class AdvantageHelper:
        @staticmethod
        def decode_solution(problem: CDP, sample: dict) -> dict:
            return {
                int(str(key)[len("x") :]): val
                for key, val in problem.sort_encoded_solution(sample).items()
            }

        @staticmethod
        def communities_from_sample(sample: dict, n_communities: int) -> list:
            communities: list = []
            for k in range(n_communities):
                comm = []
                for i in sample:
                    if sample[i] == k:
                        comm.append(i)
                communities.append(set(comm))

            return communities

    class LouvainHelper:
        @staticmethod
        def louvain_communities_to_sample_like(lcda: list) -> dict:
            sample_like = {
                node_i: comm_i
                for comm_i, comms_set in enumerate(lcda)
                for node_i in comms_set
            }
            return dict(sorted(sample_like.items()))
