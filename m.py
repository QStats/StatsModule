import numpy as np


res_val = 1.2
mod_res = 1.0

n_runs_per_param = 2

rv = np.array([[res_val] * n_runs_per_param])
rvv = rv.transpose()
mr = np.array([[mod_res] * n_runs_per_param])
mrr = mr.transpose()

print(rv, rv.shape)
print(rvv, rvv.shape)


hs = np.hstack((rvv, mrr))
print(hs, hs.shape)

vs = np.vstack((rvv, mrr))
print(vs, vs.shape)
