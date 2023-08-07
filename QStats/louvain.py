import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.community import louvain_communities, modularity

n = 100

d_alias = ["ord", "k", "sample", "mod_score", "energy", "run_time"]
d_types = [np.int_, np.int_, np.object_, np.float64, np.float_, np.float_]
types = np.dtype([(a, d) for a, d in zip(d_alias, d_types)])
dims = (n, 1)
arr: np.ndarray = np.zeros(dims, dtype=types)
G = nx.karate_club_graph()


def louvain_run(idx: int, resolution: float = 0.5):
    start = time.time()
    lcda = louvain_communities(G, resolution=resolution)
    end = time.time()
    total_time = end - start
    mod = modularity(G, lcda, resolution=resolution)
    return idx, len(lcda), lcda, mod, 34.55, total_time


res_lcda = Parallel(n_jobs=4)(delayed(louvain_run)(i) for i in range(n))
for idx, el in enumerate(res_lcda):
    arr[idx] = el

print(np.array(arr["mod_score"]).mean())
print(np.array(arr["k"]).mean())
