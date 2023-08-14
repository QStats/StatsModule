import networkx as nx
import numpy as np
from QHyper.problems.community_detection import BrainNetwork

from paths import IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE

MATRIX_RESOLUTION = "matrix_res"
SCORE_RESOLUTION = "score_res"
K = "k"
SAMPLE = "sample"
MOD_SCORE = "mod_score"
EN = "energy"
R_TIME = "run_time"

ADV_RES_ALIASES = [K, SAMPLE, MOD_SCORE, EN, R_TIME]
ADV_RES_DTYPES = [np.float_, np.object_, np.float64, np.float_, np.float_]
ADV_RES_TYPES = np.dtype(
    [(a, t) for a, t in zip(ADV_RES_ALIASES, ADV_RES_DTYPES)]
)

LOU_RES_ALIASES = [K, SAMPLE, MOD_SCORE, R_TIME]
LOU_RES_DTYPES = [np.float_, np.object_, np.float64, np.float_]
LOU_RES_TYPES = np.dtype(
    [(a, t) for a, t in zip(LOU_RES_ALIASES, LOU_RES_DTYPES)]
)

G = nx.karate_club_graph()
BRAIN_NETWORK_GRAPH = BrainNetwork(
    IN_BRAIN_NETWORK_DIR, IN_BRAIN_NETWORK_FILE
).graph
