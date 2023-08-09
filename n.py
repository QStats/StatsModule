import numpy as np

N = 4

d_alias = ["ord", "k", "sample", "mod_score", "run_time"]
d_types = [np.int_, np.int_, np.object_, np.float64, np.float_]
types = np.dtype(
    [(a, d) for a, d in zip(d_alias, d_types)]
)
dims = (N, 1)
arr: np.ndarray = np.zeros(dims, dtype=types)

x = np.array([(1, 2), (3, 4)], dtype=[('foo', np.int_), ('bar', np.int_)])

print(x['foo'])


for i in range(N):
    arr[i] = i, i+1, {"d": 3, 4: "k"}, 0.6573853, 1232
    print(arr[i])
print(f"ARR: {arr}\n\n\n")

ids = arr["ord"]
print(ids)
k = arr["k"]
samples = arr["sample"]
modularity_scores = arr["mod_score"]
run_times = arr["run_time"]


res = np.stack(
    (ids, k, samples, modularity_scores, run_times), axis=-1
)


res = np.hstack(
    (ids, k, samples, modularity_scores, run_times)
)


print(f"RES: {res}")