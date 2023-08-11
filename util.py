import networkx as nx
import numpy as np

G = nx.karate_club_graph()


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

# EXP_ALIASES = [MATRIX_RESOLUTION, ]
